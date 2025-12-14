"""
GraphMask-style temporal explanations for the PROVEX TGN model.

This implementation adapts the GraphMask idea (learning sparse edge masks
that preserve the model prediction) to the EventContext abstraction used in
PROVEX.  For each temporal event we optimise a sigmoid gate per historical
edge and penalise dense masks.  Running it across multiple events and
aggregating the scores yields a graph-level story for the attack window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor
from tqdm.auto import tqdm

try:
    from torch_geometric.explain.algorithm import GraphMaskExplainer as PyGGraphMaskExplainer
except ImportError:
    from .pyg_graphmask import GraphMaskExplainer as PyGGraphMaskExplainer

class DebugGraphMaskExplainer(PyGGraphMaskExplainer):
    def _train_explainer(self, model, x, edge_index, **kwargs):
        try:
            return super()._train_explainer(model, x, edge_index, **kwargs)
        except TypeError as e:
            if "Tensor, not type" in str(e):
                print("\n[DEBUG] Caught TypeError in GraphMaskExplainer._train_explainer.")
                print(f"[DEBUG] Model: {model}")
                print("[DEBUG] Iterating modules to check for potential mismatches:")
                for i, module in enumerate(model.modules()):
                    print(f"[DEBUG] Module {i}: {module._get_name()}")
                
                # Check for gates
                if hasattr(self, 'gates'):
                   print(f"[DEBUG] Gates length: {len(self.gates)}")
                   
                raise e
            raise e


from .utils import EventContext, TemporalLinkWrapper, ensure_gpu_space, log_cuda_memory
try:  # pragma: no cover
    from ..config import (
        include_edge_type as _DEFAULT_INCLUDE_EDGE_TYPE,
        node_embedding_dim as _DEFAULT_NODE_EMBEDDING_DIM,
    )
except ImportError:  # pragma: no cover
    try:
        from config import (  # type: ignore
            include_edge_type as _DEFAULT_INCLUDE_EDGE_TYPE,
            node_embedding_dim as _DEFAULT_NODE_EMBEDDING_DIM,
        )
    except ImportError:  # pragma: no cover
        _DEFAULT_INCLUDE_EDGE_TYPE = None
        _DEFAULT_NODE_EMBEDDING_DIM = None


@dataclass
class GraphMaskResult:
    """
    Container for the results of a GraphMask explanation on a single event.
    """
    event_index: int
    edge_importance: Tensor  # shape: (num_edges,)
    edges: List[Tuple[int, int, str, int]]
    loss_history: List[float]


class GraphMaskExplainer:
    """
    Lightweight GraphMask-style explainer tailored for EventContext objects.
    Wraps torch_geometric.explain.algorithm.GraphMaskExplainer.

    Parameters
    ----------
    sparsity_weight: float
        Coefficient encouraging sparse masks (default 1e-3).
    entropy_weight: float
        Coefficient discouraging ambiguous masks (default 1e-3).
    epochs: int
        Number of optimisation steps per event (default 200).
    lr: float
        Learning rate for Adam optimiser (default 0.01).
    """

    def __init__(
        self,
        sparsity_weight: float = 1e-3,
        entropy_weight: float = 1e-3,
        epochs: int = 200,
        lr: float = 0.01,
        include_edge_type=None,
        node_embedding_dim=None,
    ) -> None:
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight
        self.epochs = epochs
        self.lr = lr
        self.include_edge_type = include_edge_type or _DEFAULT_INCLUDE_EDGE_TYPE
        self.node_embedding_dim = node_embedding_dim or _DEFAULT_NODE_EMBEDDING_DIM
        if self.include_edge_type is None or self.node_embedding_dim is None:
            raise ValueError(
                "GraphMaskExplainer requires 'include_edge_type' and 'node_embedding_dim'. "
                "Pass them to the constructor or ensure cadets.config is importable."
            )

    def explain_event(
        self,
        context: EventContext,
        wrapper: TemporalLinkWrapper,
        device: torch.device,
    ) -> GraphMaskResult:
        """
        Learn an edge mask for a single temporal event using PyG's GraphMaskExplainer.
        """
        edge_messages = context.edge_messages.to(device)
        edge_index = context.edge_index.to(device)
        edge_times = context.edge_times.to(device)

        num_edges = edge_messages.size(0)
        if num_edges == 0:
            return GraphMaskResult(
                event_index=context.event_index,
                edge_importance=torch.tensor([]),
                edges=[],
                loss_history=[],
            )

        # Initialize PyG Explainer with GraphMask algorithm
        # Note: PyG's GraphMaskExplainer parameters might differ slightly in naming/scale
        # We map our parameters to PyG's where possible.
        # penalty_scaling corresponds roughly to sparsity/entropy weights in the original paper,
        # but PyG implementation might have specific kwargs.
        # For now, we use default penalty_scaling=5 and pass our weights if supported
        # or rely on defaults.
        # USE DEBUG VERSION
        algorithm = DebugGraphMaskExplainer(
            num_layers=1, # TemporalLinkWrapper acts as a single layer GNN
            epochs=self.epochs,
            lr=self.lr,
            log=False
        )

        explainer = Explainer(
            model=wrapper,
            algorithm=algorithm,
            explanation_type="phenomenon",
            edge_mask_type="object",
            node_mask_type=None,
            model_config=ModelConfig(
                mode="multiclass_classification",
                task_level="edge",
                return_type="raw",
            ),
        )

        target = torch.tensor([max(context.label, 0)], device=device)

        # Run explanation
        # PyG Explainer call signature: x, edge_index, **kwargs
        explanation = explainer(
            x=context.memory_inputs.to(device),
            edge_index=edge_index,
            target=target,
            edge_attr=edge_messages,
            edge_t=edge_times
        )

        importance = explanation.edge_mask.detach().cpu()

        # PyG GraphMask doesn't expose loss history easily in the standard return
        # We return an empty list for now or could monkeypatch if strictly needed.
        losses: List[float] = []

        edge_meta: List[Tuple[int, int, str, int]] = []
        for idx in range(num_edges):
            src = int(context.edge_index[0, idx])
            dst = int(context.edge_index[1, idx])
            timestamp = int(context.edge_times[idx].item())
            msg_slice = context.edge_messages[idx]
            relation_idx = torch.argmax(
                msg_slice[self.node_embedding_dim:-self.node_embedding_dim]).item()
            relation = self.include_edge_type[relation_idx]
            edge_meta.append((src, dst, relation, timestamp))

        return GraphMaskResult(
            event_index=context.event_index,
            edge_importance=importance,
            edges=edge_meta,
            loss_history=losses,
        )

    def explain_window(
        self,
        contexts: Iterable[Tuple[int, EventContext]],
        wrapper_factory,
        device: torch.device,
        top_k_events: int | None = None,
        fallback_wrapper_factory=None,
        fallback_device: torch.device | None = None,
    ) -> List[GraphMaskResult]:
        """
        Run GraphMask on a selection of events and aggregate the results.
        """
        ordered_contexts = list(contexts)
        if top_k_events is not None:
            ordered_contexts = ordered_contexts[:top_k_events]

        results: List[GraphMaskResult] = []

        for idx, item in enumerate(
            tqdm(ordered_contexts, desc="GraphMask events", leave=False), start=1
        ):
            if not ensure_gpu_space():
                if fallback_wrapper_factory is not None and fallback_device is not None:
                    print(
                        f"[warn] Insufficient GPU memory for GraphMask event "
                        f"{item[1].event_index}; falling back to CPU."
                    )
                    wrapper = fallback_wrapper_factory(
                        item[1], fallback_device)
                    results.append(self.explain_event(
                        item[1], wrapper, fallback_device))
                    continue
                print(
                    f"[warn] Insufficient GPU memory for GraphMask event; skipping "
                    f"context {item[1].event_index}."
                )
                continue
            _, context = item
            log_cuda_memory(f"GraphMask event {context.event_index}", step=idx)
            wrapper = wrapper_factory(context, device)
            results.append(self.explain_event(context, wrapper, device))
        return results

    @staticmethod
    def aggregate(
        results: Iterable[GraphMaskResult],
    ) -> Dict[Tuple[int, int, str], Dict[str, List[float] | float | int | List[int]]]:
        """
        Aggregate per-event masks into a single mapping keyed by edge identity.
        """
        scores: Dict[Tuple[int, int, str], List[float]] = {}
        timestamps: Dict[Tuple[int, int, str], List[int]] = {}
        for res in results:
            for (src, dst, relation, ts), weight in zip(res.edges, res.edge_importance.tolist()):
                key = (src, dst, relation)
                scores.setdefault(key, []).append(weight)
                timestamps.setdefault(key, []).append(ts)

        aggregated: Dict[Tuple[int, int, str],
                         Dict[str, List[float] | float | int | List[int]]] = {}
        for key, weights in scores.items():
            aggregated[key] = {
                "weight": float(sum(weights) / len(weights)),
                "count": len(weights),
                "timestamps": sorted(timestamps.get(key, [])),
            }
        return aggregated
