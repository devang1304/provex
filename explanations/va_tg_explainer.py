"""
Variational temporal explainer (VA-TGExplainer style) for PROVEX TGN.

The implementation learns a variational distribution over temporal edge masks
using the reparameterisation trick.  It produces per-edge importance scores
that respect temporal ordering and account for uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from .utils import EventContext, TemporalLinkWrapper
try:  # pragma: no cover
    from ..config import include_edge_type as _DEFAULT_INCLUDE_EDGE_TYPE, node_embedding_dim as _DEFAULT_NODE_EMBEDDING_DIM
except ImportError:  # pragma: no cover
    try:
        from config import include_edge_type as _DEFAULT_INCLUDE_EDGE_TYPE, node_embedding_dim as _DEFAULT_NODE_EMBEDDING_DIM  # type: ignore
    except ImportError:  # pragma: no cover
        _DEFAULT_INCLUDE_EDGE_TYPE = None
        _DEFAULT_NODE_EMBEDDING_DIM = None


@dataclass
class VATGResult:
    event_index: int
    edge_importance: Tensor
    edges: List[Tuple[int, int, str, int]]
    kl_history: List[float]
    loss_history: List[float]


class VATGExplainer:
    """
    Variational temporal explainer approximating VA-TGExplainer.

    Parameters
    ----------
    epochs: int
        Number of optimisation iterations (default 200).
    lr: float
        Learning rate for Adam optimiser (default 0.01).
    kl_weight: float
        Coefficient for KL divergence regularisation (default 1e-3).
    sparsity_weight: float
        Coefficient encouraging sparse masks (default 1e-3).
    """

    def __init__(
        self,
        epochs: int = 200,
        lr: float = 0.01,
        kl_weight: float = 1e-3,
        sparsity_weight: float = 1e-3,
        max_kl: float = 10.0,
        num_samples: int = 10,
        sparsity_topk: int = 10,
        include_edge_type=None,
        node_embedding_dim=None,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.kl_weight = kl_weight
        self.sparsity_weight = sparsity_weight
        self.max_kl = max_kl
        self.num_samples = num_samples
        self.sparsity_topk = sparsity_topk
        self.include_edge_type = include_edge_type or _DEFAULT_INCLUDE_EDGE_TYPE
        self.node_embedding_dim = node_embedding_dim or _DEFAULT_NODE_EMBEDDING_DIM
        if self.include_edge_type is None or self.node_embedding_dim is None:
            raise ValueError(
                "VATGExplainer requires 'include_edge_type' and 'node_embedding_dim'. "
                "Pass them to the constructor or ensure cadets.config is importable."
            )

    def explain_event(
        self,
        context: EventContext,
        wrapper: TemporalLinkWrapper,
        device: torch.device,
        num_samples: int = 5,
    ) -> VATGResult:
        edge_messages = context.edge_messages.to(device)
        edge_index = context.edge_index.to(device)
        edge_times = context.edge_times.to(device)

        num_edges = edge_messages.size(0)
        if num_edges == 0:
            return VATGResult(
                event_index=context.event_index,
                edge_importance=torch.tensor([]),
                edges=[],
                kl_history=[],
                loss_history=[],
            )
        mu = torch.zeros(num_edges, device=device, requires_grad=True)
        log_sigma = torch.full((num_edges,), math.log(
            0.1), device=device, requires_grad=True)

        optimizer = Adam([mu, log_sigma], lr=self.lr)
        target = torch.tensor([max(context.label, 0)], device=device)

        losses: List[float] = []
        kls: List[float] = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            ce_total = 0.0
            for _ in range(self.num_samples):
                eps = torch.randn_like(mu)
                mask = torch.sigmoid(mu + eps * torch.exp(0.5 * log_sigma))
                masked_messages = edge_messages * mask.unsqueeze(-1)
                logits = wrapper(
                    context.memory_inputs.to(device),
                    edge_index,
                    edge_attr=masked_messages,
                    edge_t=edge_times,
                )
                ce_total = ce_total + F.cross_entropy(logits, target)

            ce_loss = ce_total / self.num_samples
            kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
            mask_mean = torch.sigmoid(mu)
            if self.sparsity_topk > 0 and mask_mean.numel() > 0:
                topk = min(self.sparsity_topk, mask_mean.numel())
                sparsity = mask_mean.topk(topk).values.mean()
            else:
                sparsity = mask_mean.mean()

            loss = ce_loss + self.kl_weight * kl + self.sparsity_weight * sparsity
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            kls.append(kl.item())
            normalized_kl = (kl / max(1, num_edges)).item()
            if normalized_kl > self.max_kl:
                break

        importance = torch.sigmoid(mu).detach().cpu()
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

        return VATGResult(
            event_index=context.event_index,
            edge_importance=importance,
            edges=edge_meta,
            kl_history=kls,
            loss_history=losses,
        )

    @staticmethod
    def aggregate(
        results: Iterable[VATGResult],
    ) -> Dict[Tuple[int, int, str], Dict[str, float | int | List[int]]]:
        scores: Dict[Tuple[int, int, str], List[float]] = {}
        timestamps: Dict[Tuple[int, int, str], List[int]] = {}
        for res in results:
            for (src, dst, relation, ts), weight in zip(res.edges, res.edge_importance.tolist()):
                key = (src, dst, relation)
                scores.setdefault(key, []).append(weight)
                timestamps.setdefault(key, []).append(ts)

        aggregated: Dict[Tuple[int, int, str],
                         Dict[str, float | int | List[int]]] = {}
        for key, weights in scores.items():
            aggregated[key] = {
                "weight": float(sum(weights) / len(weights)),
                "count": len(weights),
                "timestamps": sorted(timestamps.get(key, [])),
            }
        return aggregated
