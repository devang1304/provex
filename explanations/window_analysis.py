"""
Explanation pipeline focused on attack windows for the PROVEX TGN model.

Workflow (no CLI arguments):
  * Load the pretrained model.
  * Locate attack windows using ``fetch_attack_list``.
  * For each window:
      - Aggregate high-loss events to produce a GraphMask-style graph story.
      - Select nodes whose cumulative loss exceeds the threshold.
      - Run GNNExplainer and VA-TGExplainer on edges touching those nodes.
  * Persist concise JSON outputs under ``artifact/explanations``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import torch
from tqdm import tqdm

try:
    from ..config import ARTIFACT_DIR, NODE_MAPPING_JSON, include_edge_type, node_embedding_dim
    from ..kairos_utils import datetime_to_ns_time_US, ns_time_to_datetime_US
    from .export_node_mapping import ensure_node_mapping  # type: ignore
except ImportError:  # pragma: no cover
    from config import ARTIFACT_DIR, NODE_MAPPING_JSON, include_edge_type, node_embedding_dim
    from kairos_utils import datetime_to_ns_time_US, ns_time_to_datetime_US
    from explanations.export_node_mapping import ensure_node_mapping  # type: ignore

from . import gnn_explainer, graphmask_explainer, utils, va_tg_explainer
from .utils import TemporalLinkWrapper, ensure_gpu_space, log_cuda_memory

DEFAULT_GRAPH_LABEL = "4_6"
# Set False to process the entire day (edit this flag manually).
USE_ATTACK_WINDOWS = True
MAX_EVENTS_PER_WINDOW = 50
GRAPHMASK_TOP_EVENTS = 25
MAX_NODES_PER_WINDOW = 20
THRESHOLD_MULTIPLIER = 1.5
TOP_K_EDGE_EXPLANATIONS = 10
MIN_EDGE_WEIGHT = 0.1
MIN_NODE_SCORE = 0.1
WARMUP_MARGIN_SECONDS = int(os.environ.get(
    "PROVEX_EXPLAIN_WARMUP_SEC", 2 * 3600))

HARD_CODED_ATTACK_WINDOWS = [
    ("2018-04-06 11:00:00", "2018-04-06 12:15:00"),
]

OUTPUT_DIR = os.path.join(ARTIFACT_DIR, "explanations")
NODE_MAPPING_PATH = os.getenv("PROVEX_NODE_MAPPING_JSON", NODE_MAPPING_JSON)
INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def _setup_logger() -> logging.Logger:
    """
    Sets up a file-based logger for the explanation pipeline.

    The logger writes to 'temporal_explanations.log' in the configured
    artifact output directory. It uses a specific format including timestamp,
    log level, and message.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger("explanations_logger")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    handler = logging.FileHandler(os.path.join(
        OUTPUT_DIR, "temporal_explanations.log"))
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _compute_threshold(losses: List[float]) -> float:
    """
    Computes a dynamic threshold for selecting high-loss events.

    The threshold is calculated as: mean + (multiplier * standard_deviation).
    This helps identify events that are statistically significant outliers
    in terms of loss.

    Args:
        losses: A list of loss values from the events in the window.

    Returns:
        float: The computed threshold value.
    """
    tensor = torch.tensor(losses)
    mu = float(tensor.mean().item())
    sigma = float(tensor.std(unbiased=False).item())
    return mu + THRESHOLD_MULTIPLIER * sigma


def _select_nodes(
    contexts: List[utils.EventContext],
    threshold: float,
) -> Tuple[List[int], Dict[int, float], Dict[int, int], Dict[int, float]]:
    """
    Selects the most relevant nodes for detailed explanation based on their cumulative loss.

    It aggregates the loss for each node across all events in the window. Nodes with
    a total score exceeding the threshold are selected. If no nodes meet the threshold,
    it falls back to selecting the top nodes by score, provided they meet a minimum score.

    Args:
        contexts: List of event contexts in the window.
        threshold: The loss threshold used for filtering events (used here as a baseline).

    Returns:
        Tuple containing:
            - List[int]: Selected node IDs.
            - Dict[int, float]: Map of node ID to total accumulated loss score.
            - Dict[int, int]: Map of node ID to count of events involving the node.
            - Dict[int, float]: Map of node ID to the maximum single event loss seen.
    """
    scores: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)
    peak_losses: Dict[int, float] = defaultdict(float)
    for ctx in contexts:
        loss_val = float(ctx.loss)
        scores[ctx.src_node] += loss_val
        scores[ctx.dst_node] += loss_val
        counts[ctx.src_node] += 1
        counts[ctx.dst_node] += 1
        peak_losses[ctx.src_node] = max(peak_losses[ctx.src_node], loss_val)
        peak_losses[ctx.dst_node] = max(peak_losses[ctx.dst_node], loss_val)

    selected = [node for node, score in scores.items() if score >= threshold]
    if selected:
        return selected[:MAX_NODES_PER_WINDOW], scores, counts, peak_losses

    fallback = [
        node for node, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if score >= MIN_NODE_SCORE
    ]
    return fallback[:MAX_NODES_PER_WINDOW], scores, counts, peak_losses


def _serialise_tensor(tensor: torch.Tensor | None) -> List[float]:
    """
    Helper to convert a PyTorch tensor to a standard Python list of floats.
    Handles None inputs and detaches from the computation graph.
    """
    if tensor is None:
        return []
    return tensor.detach().cpu().tolist()


def _serialise_edge_map(
    edge_map: Dict[Tuple[int, int, str], Dict[str, float | int | List[int]]]
) -> List[Dict[str, object]]:
    """
    Converts the internal edge map (dictionary with tuple keys) into a JSON-serializable
    list of dictionaries.

    Args:
        edge_map: Dictionary where keys are (src, dst, relation) tuples and values
                  are dictionaries of edge attributes.

    Returns:
        List[Dict[str, object]]: List of edge objects suitable for JSON output.
    """
    return [
        {
            "src": key[0],
            "dst": key[1],
            "relation": key[2],
            "weight": value.get("weight", 0.0),
            "count": value.get("count", 0),
            "timestamps": value.get("timestamps", []),
        }
        for key, value in edge_map.items()
    ]


def _safe_filename(text: str, default: str = "window") -> str:
    """
    Sanitizes a string to be safe for use as a filename.
    Replaces invalid characters and spaces with underscores.
    """
    cleaned = []
    for ch in text:
        if ch in INVALID_FILENAME_CHARS:
            cleaned.append("_")
        elif ch.isspace():
            cleaned.append("_")
        else:
            cleaned.append(ch)
    candidate = "".join(cleaned).strip("_").strip(".")
    return candidate or default


def _top_edge_explanations(
    mask_metrics: Dict[str, float],
    context: utils.EventContext,
) -> List[Dict[str, float]]:
    """
    Extracts the top contributing edges from the GraphMask explanation results.

    It looks at the 'edge_mask' weights returned by GraphMask, ranks them,
    and returns the top K edges that have a weight above the minimum threshold.

    Args:
        mask_metrics: Dictionary containing metrics from GraphMask, including 'edge_mask'.
        context: The event context associated with the explanation.

    Returns:
        List[Dict[str, float]]: List of top edges with their metadata and importance weights.
    """
    weights = mask_metrics.get("edge_mask", [])
    if not weights:
        return []
    ranked = sorted(
        zip(range(len(weights)), weights),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top = []
    for idx, weight in ranked:
        if len(top) >= TOP_K_EDGE_EXPLANATIONS:
            break
        if weight < MIN_EDGE_WEIGHT:
            continue
        src = int(context.edge_index[0, idx])
        dst = int(context.edge_index[1, idx])
        timestamp = int(context.edge_times[idx].item())
        msg_slice = context.edge_messages[idx]
        relation_idx = torch.argmax(
            msg_slice[node_embedding_dim:-node_embedding_dim]).item()
        relation = include_edge_type[relation_idx]
        top.append(
            {
                "src": src,
                "dst": dst,
                "relation": relation,
                "timestamp": timestamp,
                "weight": float(weight),
            }
        )
    return top


def _collect_windows(train_data, memory, gnn, link_pred, device):
    """
    Identifies time windows of interest and collects event contexts for them.

    If USE_ATTACK_WINDOWS is True, it uses the hardcoded attack windows.
    Otherwise, it defaults to the full duration of the dataset.
    It handles caching of contexts to speed up subsequent runs.

    Args:
        train_data: The temporal graph dataset.
        memory: The TGN memory module.
        gnn: The GNN module.
        link_pred: The link prediction module.
        device: The torch device (CPU or CUDA).

    Returns:
        Dict: A mapping from window identifier tuples to lists of EventContext objects.
    """
    if USE_ATTACK_WINDOWS:
        windows: List[Tuple[int, int, str]] = []
        for start_str, end_str in HARD_CODED_ATTACK_WINDOWS:
            start_ns = datetime_to_ns_time_US(start_str)
            end_ns = datetime_to_ns_time_US(end_str)
            identifier = f"{start_str.replace(' ', '_')}~{end_str.replace(' ', '_')}.txt"
            windows.append((start_ns, end_ns, identifier))
    else:
        start_ns = int(train_data.t.min().item())
        end_ns = int(train_data.t.max().item())
        windows = [(start_ns, end_ns, f"graph_{DEFAULT_GRAPH_LABEL}_full_day")]

    cached = utils.load_context_cache(DEFAULT_GRAPH_LABEL, windows)
    if cached is not None:
        print("[info] Loaded cached event contexts for selected windows.")
        return cached

    if not windows:
        return {}

    earliest_start = min(start for start, _, _ in windows)
    latest_end = max(end for _, end, _ in windows)

    margin_ns = WARMUP_MARGIN_SECONDS * 1_000_000_000
    data_min = int(train_data.t.min().item())
    data_max = int(train_data.t.max().item())
    # Ensure we start slightly before the window to provide context for the model (warmup)
    slice_start_ns = max(data_min, earliest_start - margin_ns)
    slice_end_ns = min(data_max, latest_end)

    print(
        f"[info] Streaming slice from {ns_time_to_datetime_US(slice_start_ns)} "
        f"to {ns_time_to_datetime_US(slice_end_ns)} for context collection."
    )

    sliced_data, _, _ = utils.slice_temporal_graph(
        train_data, slice_start_ns, slice_end_ns)
    # Find the index in the sliced data where the actual window of interest begins
    context_start_offset = int(
        torch.searchsorted(sliced_data.t, torch.tensor(
            earliest_start, device=sliced_data.t.device))
    )

    contexts_by_window: Dict[Tuple[int, int, str], List[utils.EventContext]] = {
        window: [] for window in windows}

    def _predicate(ctx: utils.EventContext) -> bool:
        included = False
        for window in windows:
            start_ns, end_ns, _ = window
            if start_ns <= ctx.timestamp <= end_ns:
                contexts_by_window[window].append(ctx)
                included = True
        return included

    # Exhaust generator to populate contexts_by_window; actual yielded values aren't needed.
    for _ in utils.stream_event_contexts(
        sliced_data,
        memory,
        gnn,
        link_pred,
        device=device,
        start_offset=context_start_offset,
        predicate=_predicate,
    ):
        pass

    print("[info] Persisting window contexts to cache for future runs.")
    utils.save_context_cache(DEFAULT_GRAPH_LABEL, windows, contexts_by_window)
    return contexts_by_window


def run_pipeline() -> Dict[str, object]:
    """
    Execute the full explanation pipeline for the configured graph and windows.

    Steps:
    1. Setup logging and device.
    2. Load the PROVEX model and graph data.
    3. Collect event contexts for the specified windows (attack windows or full day).
    4. Initialize GraphMask and VA-TG explainers (including CPU fallbacks).
    5. Iterate through each window:
        a. Filter for high-loss events based on a dynamic threshold.
        b. Run GraphMask on high-loss events to get a "graph story".
        c. Select key nodes based on cumulative loss.
        d. For each selected node, run GNNExplainer and VA-TGExplainer on related events.
        e. Aggregate results and save to a JSON file.
    6. Save a global summary JSON.

    Returns:
        Dict[str, object]: The summary dictionary containing paths to all generated explanations.
    """
    logger = _setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    configured_mode = getattr(utils, "_CONTEXT_STORAGE_MODE", "auto")
    effective_mode = "cpu" if configured_mode == "auto" else configured_mode
    print(
        f"[info] Context tensors will be cached on {effective_mode.upper()} "
        "(override via PROVEX_CONTEXT_DEVICE)."
    )
    # Ensure node mapping exists early (best-effort; pipeline should not crash if unavailable)
    try:
        ensure_node_mapping(Path(NODE_MAPPING_PATH))
    except Exception as _e:  # pylint: disable=broad-except
        print(f"[warn] Node mapping creation failed or skipped: {_e}")

    print(
        f"[info] Loading PROVEX model and graph for label '{DEFAULT_GRAPH_LABEL}'...")
    memory, gnn, link_pred = utils.load_model(device=device)
    train_data = utils.load_temporal_graph(DEFAULT_GRAPH_LABEL)
    print("[success] Model and graph loaded.")

    print("[info] Gathering attack windows and streaming contexts...")
    window_contexts = _collect_windows(
        train_data, memory, gnn, link_pred, device)
    num_windows = len(window_contexts)
    total_events = sum(len(ctxs) for ctxs in window_contexts.values())
    print(
        f"[info] Collected contexts for {num_windows} window(s); total events: {total_events}.")

    masker = graphmask_explainer.GraphMaskExplainer()
    temporal_explainer = va_tg_explainer.VATGExplainer()
    cpu_device = torch.device("cpu")
    gnn_cpu = copy.deepcopy(gnn).to(cpu_device)
    link_pred_cpu = copy.deepcopy(link_pred).to(cpu_device)
    temporal_explainer_cpu = va_tg_explainer.VATGExplainer()
    print("[info] Initialised GraphMask and VA-TG explainers.")

    outputs: List[Dict[str, object]] = []

    for window, contexts in tqdm(window_contexts.items(), desc="Windows", leave=True):
        if not contexts:
            print(f"[warn] Window {window[2]} has no contexts; skipping.")
            continue

        print(
            f"[info] Processing window {window[2]} with {len(contexts)} events...")
        contexts = sorted(contexts, key=lambda ctx: ctx.loss, reverse=True)
        losses = [ctx.loss for ctx in contexts]
        # Calculate dynamic threshold to identify outliers
        threshold = _compute_threshold(losses)

        high_loss_candidates = [(idx, ctx) for idx, ctx in enumerate(
            contexts) if ctx.loss >= threshold]
        candidate_count = len(high_loss_candidates)
        high_loss_events = high_loss_candidates[:MAX_EVENTS_PER_WINDOW]

        if not high_loss_events:
            # Fallback: if no events meet the threshold, take the top N events by loss
            high_loss_events = list(
                enumerate(contexts[:MAX_EVENTS_PER_WINDOW]))
            candidate_count = len(contexts)

        def _wrapper_factory(
            ctx: utils.EventContext, target_device: torch.device
        ) -> TemporalLinkWrapper:
            if target_device.type == "cpu":
                return TemporalLinkWrapper(gnn_cpu, link_pred_cpu, ctx, target_device)
            return TemporalLinkWrapper(gnn, link_pred, ctx, target_device)

        mask_results = masker.explain_window(
            high_loss_events,
            wrapper_factory=_wrapper_factory,
            device=device,
            top_k_events=GRAPHMASK_TOP_EVENTS,
            fallback_wrapper_factory=_wrapper_factory,
            fallback_device=cpu_device,
        )
        print(
            f"[info] GraphMask analysed {len(mask_results)} event(s) for window {window[2]}.")
        # Aggregate individual event explanations into a window-level summary
        aggregated_map = graphmask_explainer.GraphMaskExplainer.aggregate(
            mask_results)

        selected_nodes, node_scores, node_counts, node_peak_losses = _select_nodes(
            contexts, threshold)
        print(
            f"[info] Selected {len(selected_nodes)} node(s) for detailed explanations.")

        node_outputs = []
        for node in tqdm(selected_nodes, desc="Nodes", leave=False):
            related_contexts_all = [
                ctx for ctx in contexts if ctx.src_node == node or ctx.dst_node == node]
            related_contexts = related_contexts_all[:MAX_EVENTS_PER_WINDOW]
            max_event_loss = node_peak_losses.get(node, 0.0)

            gnn_results = []
            va_event_results: List[va_tg_explainer.VATGResult] = []
            va_serialised = []

            pending: Deque[utils.EventContext] = deque(related_contexts)
            gnn_event_counter = 0

            while pending:
                ctx = pending.popleft()
                target_device = device
                gnn_module = gnn
                link_module = link_pred
                explainer_instance = temporal_explainer

                if not ensure_gpu_space():
                    print(
                        f"[warn] Low GPU memory for event {ctx.event_index}; falling back to CPU.")
                    # Switch to CPU models if GPU memory is insufficient
                    target_device = cpu_device
                    gnn_module = gnn_cpu
                    link_module = link_pred_cpu
                    explainer_instance = temporal_explainer_cpu

                gnn_event_counter += 1
                log_cuda_memory(
                    f"GNNExplainer event {ctx.event_index}", step=gnn_event_counter)
                gnn_metrics = gnn_explainer.explain_event(
                    ctx, gnn_module, link_module, target_device)
                gnn_results.append(
                    {
                        "event_index": ctx.event_index,
                        "src_node": ctx.src_node,
                        "dst_node": ctx.dst_node,
                        "prob_full": gnn_metrics["prob_full"],
                        "prob_keep": gnn_metrics["prob_keep"],
                        "prob_drop": gnn_metrics["prob_drop"],
                        "comprehensiveness": gnn_metrics["comprehensiveness"],
                        "sufficiency": gnn_metrics["sufficiency"],
                        "sparsity": gnn_metrics["sparsity"],
                        "entropy": gnn_metrics["entropy"],
                        "runtime_sec": gnn_metrics["runtime_sec"],
                        "kept_edges": gnn_metrics["kept_edges"],
                        "top_edges": _top_edge_explanations(gnn_metrics, ctx),
                    }
                )

                wrapper = TemporalLinkWrapper(
                    gnn_module, link_module, ctx, target_device)
                log_cuda_memory(
                    f"VA-TG event {ctx.event_index}", step=gnn_event_counter)
                if target_device.type != "cpu" and not ensure_gpu_space():
                    print(
                        f"[warn] Low GPU memory before VA-TG for event {ctx.event_index}; "
                        "retrying on CPU."
                    )
                    # Double-check memory before running VA-TG, fallback if needed
                    wrapper = TemporalLinkWrapper(
                        gnn_cpu, link_pred_cpu, ctx, cpu_device)
                    explainer_instance = temporal_explainer_cpu
                    target_device = cpu_device

                va_result = explainer_instance.explain_event(
                    ctx, wrapper, target_device)
                va_event_results.append(va_result)
                va_serialised.append(
                    {
                        "event_index": va_result.event_index,
                        "edge_importance": _serialise_tensor(va_result.edge_importance),
                        "edges": [
                            {"src": edge[0], "dst": edge[1],
                                "relation": edge[2], "timestamp": edge[3]}
                            for edge in va_result.edges
                        ],
                        "kl_history": va_result.kl_history,
                        "loss_history": va_result.loss_history,
                    }
                )

            va_aggregate = va_tg_explainer.VATGExplainer.aggregate(
                va_event_results)

            avg_score = node_scores.get(
                node, 0.0) / max(node_counts.get(node, 0), 1)
            node_outputs.append(
                {
                    "node_id": node,
                    "score": node_scores.get(node, 0.0),
                    "avg_score": avg_score,
                    "event_count": node_counts.get(node, 0),
                    "max_event_loss": max_event_loss,
                    "gnn": gnn_results,
                    "va_tg": {
                        "events": va_serialised,
                        "aggregate": _serialise_edge_map(va_aggregate),
                    },
                }
            )

        window_output = {
            "window_path": window[2],
            "start_ns": window[0],
            "end_ns": window[1],
            "threshold": threshold,
            "num_events": len(contexts),
            "metrics": {
                "high_loss_candidates": candidate_count,
                "high_loss_used": len(high_loss_events),
                "graphmask_events": len(mask_results),
            },
            "graphmask": {
                "per_event": [
                    {
                        "event_index": res.event_index,
                        "edge_importance": _serialise_tensor(res.edge_importance),
                        "edges": [
                            {"src": edge[0], "dst": edge[1],
                                "relation": edge[2], "timestamp": edge[3]}
                            for edge in res.edges
                        ],
                        "loss_history": res.loss_history,
                    }
                    for res in mask_results
                ],
                "aggregate": _serialise_edge_map(aggregated_map),
            },
            "nodes": node_outputs,
        }

        outputs.append(window_output)
        logger.info("Window %s | events=%d | threshold=%.4f",
                    window[2], len(contexts), threshold)
        print(
            f"[success] Completed explanations for window {window[2]} (threshold={threshold:.4f}).")

        raw_base = os.path.splitext(os.path.basename(window[2]))[0]
        safe_base = _safe_filename(raw_base)
        out_path = os.path.join(OUTPUT_DIR, f"{safe_base}_explanations.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(window_output, fh, indent=2)
        logger.info("Wrote window explanations to %s", out_path)
        print(f"[success] Wrote window explanations to {out_path}")

    summary = {"graph_label": DEFAULT_GRAPH_LABEL, "windows": outputs}
    summary_path = os.path.join(
        OUTPUT_DIR, f"graph_{DEFAULT_GRAPH_LABEL}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary saved to %s", summary_path)
    print(f"[success] Summary saved to {summary_path}")

    return summary


def main() -> None:
    """
    Entry point for the explanation pipeline.
    """
    run_pipeline()


if __name__ == "__main__":
    main()
