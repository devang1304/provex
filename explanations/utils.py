import hashlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader
from tqdm import tqdm

try:
    from ..config import ARTIFACT_DIR, GRAPHS_DIR, MODELS_DIR, neighbor_size, node_embedding_dim
    from ..kairos_utils import tensor_find
    from .. import kairos_tgnn_model
except ImportError:  # pragma: no cover - fallback when run as script
    from config import ARTIFACT_DIR, GRAPHS_DIR, MODELS_DIR, neighbor_size, node_embedding_dim
    from kairos_utils import tensor_find
    import kairos_tgnn_model

# Controls where event context tensors are stored. Use "gpu", "cpu", or "cpu_pin";
# default "auto" keeps contexts on CPU and streams them to GPU for explanation.
_CONTEXT_STORAGE_MODE = os.environ.get("PROVEX_CONTEXT_DEVICE", "auto").lower()

_LOGGER = logging.getLogger(__name__)
_GPU_FALLBACK_WARNED = False
_GPU_FALLBACK_ACTIVE = False
_GPU_BUFFER_BYTES = int(os.environ.get(
    "PROVEX_CONTEXT_GPU_BUFFER", 512 * 1024**2))
_CUDA_LOG_EVERY = int(os.environ.get("PROVEX_CUDA_LOG_EVERY", 0))
_CACHE_DIR = os.path.join(ARTIFACT_DIR, "explanations", "cache")


def _move_to_cuda(tensor: Tensor) -> Tensor:
    global _GPU_FALLBACK_WARNED, _GPU_FALLBACK_ACTIVE
    if _GPU_FALLBACK_ACTIVE or not torch.cuda.is_available():
        return tensor.to("cpu")
    try:
        return tensor.to("cuda", non_blocking=True)
    except RuntimeError as exc:  # pragma: no cover - only triggers on OOM
        message = str(exc).lower()
        if "out of memory" in message:
            if not _GPU_FALLBACK_WARNED:
                print(
                    "[warn] GPU memory exhausted while storing contexts; falling back to CPU.")
                _LOGGER.warning(
                    "GPU out of memory when storing context tensor; falling back to CPU storage.")
                _GPU_FALLBACK_WARNED = True
            _GPU_FALLBACK_ACTIVE = True
            torch.cuda.empty_cache()
            return tensor.to("cpu")
        raise


def _store_tensor(tensor: Optional[Tensor]) -> Optional[Tensor]:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    result = tensor.detach()
    target_mode = _CONTEXT_STORAGE_MODE
    if target_mode == "auto":
        target_mode = "cpu"
    if target_mode == "gpu" and not _GPU_FALLBACK_ACTIVE:
        return _move_to_cuda(result)
    if target_mode == "cpu_pin":
        result = result.to("cpu")
        try:
            result = result.pin_memory()
        except RuntimeError:
            pass
        return result
    # default: keep on CPU for streaming to the accelerator when needed.
    return result.to("cpu")


def ensure_gpu_space(buffer_bytes: Optional[int] = None) -> bool:
    """
    Ensure at least `buffer_bytes` free on the active CUDA device.
    Returns True when enough memory is available (after optionally clearing cache),
    False otherwise. When CUDA is unavailable the function always returns True.
    """
    if not torch.cuda.is_available():
        return True

    device = torch.device("cuda")
    threshold = buffer_bytes if buffer_bytes is not None else _GPU_BUFFER_BYTES
    if threshold <= 0:
        return True

    total = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)
    free = total - reserved
    if free >= threshold:
        return True

    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved(device)
    free = total - reserved
    if free >= threshold:
        return True

    print(
        f"[warn] Available CUDA memory {free/1024**2:.1f} MiB below buffer "
        f"({threshold/1024**2:.1f} MiB). Deferring GPU work."
    )
    _LOGGER.warning(
        "CUDA free memory %.1f MiB below buffer %.1f MiB. Deferring GPU work.",
        free / 1024**2,
        threshold / 1024**2,
    )
    return False


def log_cuda_memory(stage: str, step: Optional[int] = None) -> None:
    """Print current CUDA memory stats for visibility."""
    if not torch.cuda.is_available():
        return
    if _CUDA_LOG_EVERY <= 0:
        return
    if step is not None and step % _CUDA_LOG_EVERY != 0:
        return
    device = torch.device("cuda")
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    free = total - reserved
    print(
        f"[info] CUDA memory before {stage}: "
        f"allocated={allocated/1024**2:.1f} MiB, reserved={reserved/1024**2:.1f} MiB, "
        f"free={free/1024**2:.1f} MiB."
    )


@dataclass
class EventContext:
    """Snapshot of the TGN state around a single temporal event."""

    event_index: int
    src_node: int
    dst_node: int
    timestamp: int
    label: int
    memory_inputs: Tensor
    last_update: Tensor
    edge_index: Tensor
    edge_messages: Tensor
    edge_times: Tensor
    base_embeddings: Tensor
    logits: Tensor
    probabilities: Tensor
    prob_label: float
    loss: float
    src_local_index: int
    dst_local_index: int
    node_ids: Tensor
    raw_message: Tensor
    build_time: float = 0.0
    is_attack: bool = False

    def copy(self) -> "EventContext":
        return EventContext(
            event_index=self.event_index,
            src_node=self.src_node,
            dst_node=self.dst_node,
            timestamp=self.timestamp,
            label=self.label,
            memory_inputs=self.memory_inputs.clone(),
            last_update=self.last_update.clone(),
            edge_index=self.edge_index.clone(),
            edge_messages=self.edge_messages.clone(),
            edge_times=self.edge_times.clone(),
            base_embeddings=self.base_embeddings.clone(),
            logits=self.logits.clone(),
            probabilities=self.probabilities.clone(),
            prob_label=self.prob_label,
            loss=self.loss,
            src_local_index=self.src_local_index,
            dst_local_index=self.dst_local_index,
            node_ids=self.node_ids.clone(),
            raw_message=self.raw_message.clone(),
            is_attack=self.is_attack,
        )


class TemporalLinkWrapper(torch.nn.Module):
    """Wraps the PROVEX GNN for compatibility with PyG explainer APIs."""

    def __init__(
        self,
        gnn: torch.nn.Module,
        link_pred: torch.nn.Module,
        context: EventContext,
        device: torch.device,
    ):
        super().__init__()
        self.gnn = gnn
        self.link_pred = link_pred
        self.src_idx = int(context.src_local_index)
        self.dst_idx = int(context.dst_local_index)

        self.register_buffer(
            "last_update", context.last_update.detach().to(device))
        self.register_buffer(
            "edge_times", context.edge_times.detach().to(device))
        self.register_buffer(
            "edge_messages", context.edge_messages.detach().to(device))
        self.register_buffer("node_ids", context.node_ids.detach().to(device))

        self.gnn.eval()
        self.link_pred.eval()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        edge_t: Optional[Tensor] = None,
    ) -> Tensor:
        device = self.last_update.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        msg = (edge_attr if edge_attr is not None else self.edge_messages).to(device)
        times = (edge_t if edge_t is not None else self.edge_times).to(device)
        if edge_index.numel() == 0:
            edge_index = edge_index.new_empty((2, 0))
        if msg.numel() == 0:
            msg = msg.new_empty((0, self.edge_messages.size(-1)))
        if times.numel() == 0:
            times = times.new_empty((0,))
        z = self.gnn(x, self.last_update, edge_index, times, msg)
        return self.link_pred(z[[self.src_idx]], z[[self.dst_idx]])


def _slice_indices(timestamps: Tensor, start_ns: int, end_ns: int) -> Tuple[int, int]:
    start_idx = int(torch.searchsorted(
        timestamps, torch.tensor(start_ns, device=timestamps.device)))
    end_idx = int(torch.searchsorted(timestamps, torch.tensor(
        end_ns, device=timestamps.device), right=True))
    return max(0, start_idx), min(timestamps.numel(), end_idx)


def load_temporal_graph(label: str, root: Optional[str] = None) -> TemporalData:
    """Loads a TemporalData object for the requested day/window label."""
    root = root or GRAPHS_DIR
    path = os.path.join(root, f"graph_{label}.TemporalData.simple")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Temporal graph not found at {path}")
    data: TemporalData = torch.load(path)
    return data


def slice_temporal_graph(data: TemporalData, start_ns: int, end_ns: int) -> Tuple[TemporalData, int, int]:
    """Return a sliced TemporalData and the inclusive index bounds for start/end."""
    start_idx, end_idx = _slice_indices(data.t, start_ns, end_ns)
    sliced = TemporalData(
        src=data.src[start_idx:end_idx],
        dst=data.dst[start_idx:end_idx],
        t=data.t[start_idx:end_idx],
        msg=data.msg[start_idx:end_idx],
        y=data.y[start_idx:end_idx] if hasattr(data, "y") else None,
    )
    return sliced, start_idx, end_idx


def _cache_signature(windows: List[Tuple[int, int, str]]) -> str:
    ordered = sorted((int(start), int(end)) for start, end, _ in windows)
    payload = ";".join(f"{start}-{end}" for start,
                       end in ordered).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _cache_path(label: str, windows: List[Tuple[int, int, str]]) -> str:
    signature = _cache_signature(windows)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{label}_{signature}.pt")


def load_context_cache(label: str, windows: List[Tuple[int, int, str]]) -> Optional[Dict[Tuple[int, int, str], List["EventContext"]]]:
    path = _cache_path(label, windows)
    if os.path.exists(path):
        return torch.load(path)
    return None


def save_context_cache(label: str, windows: List[Tuple[int, int, str]], contexts: Dict[Tuple[int, int, str], List["EventContext"]]) -> None:
    path = _cache_path(label, windows)
    torch.save(contexts, path)


def load_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """Loads the trained PROVEX model components."""
    model_path = model_path or os.path.join(MODELS_DIR, "models.pt")
    device = device or model.device
    memory, gnn, link_pred, _ = torch.load(model_path, map_location=device)
    memory = memory.to(device)
    gnn = gnn.to(device)
    link_pred = link_pred.to(device)

    memory.eval()
    gnn.eval()
    link_pred.eval()

    if hasattr(memory, "num_nodes"):
        model.configure_node_capacity(int(memory.num_nodes))

    return memory, gnn, link_pred


def _init_neighbor_loader(device: torch.device) -> LastNeighborLoader:
    return LastNeighborLoader(
        model.max_node_num,
        size=neighbor_size,
        device=device,
    )


def build_event_context(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    event_index: int,
    device: Optional[torch.device] = None,
) -> EventContext:
    """Replays the temporal stream until `event_index` and captures state."""
    device = device or model.device
    memory.reset_state()
    loader = _init_neighbor_loader(device)
    loader.reset_state()

    temporal_loader = TemporalDataLoader(data, batch_size=64, shuffle=False)
    processed = 0

    with torch.no_grad():
        for batch in tqdm(temporal_loader, desc="Replaying temporal stream", leave=False):
            src_cpu = batch.src
            dst_cpu = batch.dst
            t_cpu = batch.t
            msg_cpu = batch.msg

            src = src_cpu.to(device)
            dst = dst_cpu.to(device)
            t = t_cpu.to(device)
            msg = msg_cpu.to(device)

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = loader(n_id)
            model.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            memory_inputs, last_update = memory(n_id)
            e_id_cpu = e_id.cpu()
            edge_times = data.t[e_id_cpu].to(device)
            edge_messages = data.msg[e_id_cpu].to(device)

            embeddings = gnn(memory_inputs, last_update,
                             edge_index, edge_times, edge_messages)
            logits = link_pred(
                embeddings[model.assoc[src]], embeddings[model.assoc[dst]])
            probabilities = logits.softmax(dim=-1)

            if processed == event_index:
                label = tensor_find(
                    msg_cpu[0][node_embedding_dim:-node_embedding_dim], 1
                ) - 1
                prob_label = probabilities[0, label].item()
                loss = - \
                    float(torch.log(probabilities[0, label] + 1e-12).item())
                context = EventContext(
                    event_index=event_index,
                    src_node=int(src_cpu.item()),
                    dst_node=int(dst_cpu.item()),
                    timestamp=int(t_cpu.item()),
                    label=int(label),
                    memory_inputs=_store_tensor(memory_inputs),
                    last_update=_store_tensor(last_update),
                    edge_index=_store_tensor(edge_index),
                    edge_messages=_store_tensor(edge_messages),
                    edge_times=_store_tensor(edge_times),
                    base_embeddings=_store_tensor(embeddings),
                    logits=_store_tensor(logits),
                    probabilities=_store_tensor(probabilities),
                    prob_label=prob_label,
                    loss=loss,
                    src_local_index=int(model.assoc[src].item()),
                    dst_local_index=int(model.assoc[dst].item()),
                    node_ids=_store_tensor(n_id),
                    raw_message=_store_tensor(msg_cpu[0]),
                )
                return context

            memory.update_state(src, dst, t, msg)
            loader.insert(src, dst)
            processed += 1

    raise IndexError(
        f"Event index {event_index} out of range (processed {processed}).")


def stream_event_contexts(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    *,
    device: Optional[torch.device] = None,
    start_offset: int = 0,
    end_offset: Optional[int] = None,
    predicate=None,
):
    """Yields EventContext objects for every event while streaming once."""
    device = device or model.device
    memory.reset_state()
    loader = _init_neighbor_loader(device)
    loader.reset_state()

    temporal_loader = TemporalDataLoader(data, batch_size=1, shuffle=False)

    with torch.no_grad():
        for event_index, batch in enumerate(tqdm(temporal_loader, desc="Streaming event contexts", leave=False)):
            if end_offset is not None and event_index >= end_offset:
                break

            src_cpu = batch.src
            dst_cpu = batch.dst
            t_cpu = batch.t
            msg_cpu = batch.msg

            src = src_cpu.to(device)
            dst = dst_cpu.to(device)
            t = t_cpu.to(device)
            msg = msg_cpu.to(device)

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = loader(n_id)
            model.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            memory_inputs, last_update = memory(n_id)
            e_id_cpu = e_id.cpu()
            edge_times = data.t[e_id_cpu].to(device)
            edge_messages = data.msg[e_id_cpu].to(device)

            embeddings = gnn(memory_inputs, last_update,
                             edge_index, edge_times, edge_messages)
            logits = link_pred(
                embeddings[model.assoc[src]], embeddings[model.assoc[dst]])
            probabilities = logits.softmax(dim=-1)

            label = tensor_find(
                msg_cpu[0][node_embedding_dim:-node_embedding_dim], 1) - 1
            prob_label = probabilities[0, label].item()
            loss = -float(torch.log(probabilities[0, label] + 1e-12).item())

            context = EventContext(
                event_index=event_index,
                src_node=int(src_cpu.item()),
                dst_node=int(dst_cpu.item()),
                timestamp=int(t_cpu.item()),
                label=int(label),
                memory_inputs=_store_tensor(memory_inputs),
                last_update=_store_tensor(last_update),
                edge_index=_store_tensor(edge_index),
                edge_messages=_store_tensor(edge_messages),
                edge_times=_store_tensor(edge_times),
                base_embeddings=_store_tensor(embeddings),
                logits=_store_tensor(logits),
                probabilities=_store_tensor(probabilities),
                prob_label=prob_label,
                loss=loss,
                src_local_index=int(model.assoc[src].item()),
                dst_local_index=int(model.assoc[dst].item()),
                node_ids=_store_tensor(n_id),
                raw_message=_store_tensor(msg_cpu[0]),
            )

            if event_index >= start_offset:
                if predicate is None or predicate(context):
                    yield context

            memory.update_state(src, dst, t, msg)
            loader.insert(src, dst)


def compute_event_contexts(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    indices: Iterable[int],
    device: Optional[torch.device] = None,
) -> Dict[int, EventContext]:
    """Materialize contexts for the requested indices."""
    device = device or model.device
    requested = set(int(i) for i in indices)
    contexts: Dict[int, EventContext] = {}
    if not requested:
        return contexts

    for context in stream_event_contexts(data, memory, gnn, link_pred, device=device):
        if context.event_index in requested:
            contexts[context.event_index] = context
            if len(contexts) == len(requested):
                break
    return contexts


def collect_all_event_contexts(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    *,
    device: Optional[torch.device] = None,
) -> Dict[int, EventContext]:
    """Collects contexts for every event in the stream."""
    contexts: Dict[int, EventContext] = {}
    for context in stream_event_contexts(data, memory, gnn, link_pred, device=device):
        contexts[context.event_index] = context
    return contexts


def collect_contexts(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    *,
    predicate=None,
    limit: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[int, EventContext]:
    """Collects contexts satisfying `predicate` until `limit`."""
    contexts: Dict[int, EventContext] = {}

    def should_include(ctx: EventContext) -> bool:
        return True if predicate is None else bool(predicate(ctx))

    for context in stream_event_contexts(data, memory, gnn, link_pred, device=device):
        if should_include(context):
            contexts[context.event_index] = context
            if limit is not None and len(contexts) >= limit:
                break
    return contexts


def aggregate_node_scores(contexts: Iterable[EventContext]) -> Dict[int, float]:
    """
    Sum losses for source/destination nodes across contexts.
    """
    scores: Dict[int, float] = defaultdict(float)
    for ctx in contexts:
        scores[ctx.src_node] += ctx.loss
        scores[ctx.dst_node] += ctx.loss
    return scores
