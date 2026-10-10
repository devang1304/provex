"""Baseline rankings over the same items (absolute event ids) as the explainers.

All return importance scores aligned with ``em.items`` (higher = more important to the anomaly).
"""
from __future__ import annotations

import time
from typing import Dict

import torch
from torch import Tensor

from .event_model import EventModel
from .joint_mask import Explanation


def _wrap(name: str, em: EventModel, scores: Tensor, tau: float, t0: float, **extra) -> Explanation:
    with torch.no_grad():
        L_full = float(em.loss(torch.ones(em.num_items, dtype=em.dtype, device=em.device), hard=True))
    return Explanation(method=name, mode="ranking", items=em.items.tolist(), scores=scores.tolist(),
                       selected=[], valid=True, repaired=0, L_full=L_full, L_selected=float("nan"),
                       tau=tau, runtime_s=time.perf_counter() - t0, extra=extra)


def random_rank(em: EventModel, tau: float, seed: int = 0) -> Explanation:
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(seed)
    return _wrap("random", em, torch.rand(em.num_items, generator=g, dtype=torch.float64), tau, t0)


def recency(em: EventModel, tau: float, data_t: Tensor) -> Explanation:
    t0 = time.perf_counter()
    return _wrap("recency", em, data_t[em.items].to(torch.float64), tau, t0)


def gradient(em: EventModel, tau: float) -> Explanation:
    """dL/dg at g = 1. Removing item k changes L by about -dL/dg_k, so a positive gradient
    means the item raises the anomaly score."""
    t0 = time.perf_counter()
    g = torch.ones(em.num_items, dtype=em.dtype, device=em.device, requires_grad=True)
    em.loss(g).backward()
    return _wrap("gradient", em, g.grad.detach().to(torch.float64), tau, t0)


def integrated_gradients(em: EventModel, tau: float, steps: int = 20) -> Explanation:
    """IG from the all-removed baseline g = 0 to g = 1: attribution of L(1) - L(0)."""
    t0 = time.perf_counter()
    total = torch.zeros(em.num_items, dtype=em.dtype, device=em.device)
    for k in range(1, steps + 1):
        g = torch.full((em.num_items,), k / steps, dtype=em.dtype, device=em.device, requires_grad=True)
        em.loss(g).backward()
        total += g.grad.detach()
    return _wrap("integrated_gradients", em, (total / steps).to(torch.float64), tau, t0)


@torch.no_grad()
def occlusion(em: EventModel, tau: float, max_items: int = 4096) -> Explanation:
    """Leave-one-out: L(all) - L(all but item k), by exact hard deletion."""
    t0 = time.perf_counter()
    ones = torch.ones(em.num_items, dtype=em.dtype, device=em.device)
    L_full = float(em.loss(ones, hard=True))
    scores = torch.full((em.num_items,), float("nan"), dtype=torch.float64)
    for k in range(min(em.num_items, max_items)):
        g = ones.clone(); g[k] = 0
        scores[k] = L_full - float(em.loss(g, hard=True))
    return _wrap("occlusion", em, scores, tau, t0, truncated=em.num_items > max_items)


@torch.no_grad()
def attention(em: EventModel, tau: float) -> Explanation:
    """Attention rollout for neighbour edges: conv2 attention into u/v, times conv1 attention
    into the edge's source. History-only items get 0."""
    t0 = time.perf_counter()
    gnn = em.gnn
    ei = em.ei
    x = em.x
    edge_attr = em._edge_attr(em.lu_off, ei, em.edge_t_off, em.edge_msg)
    _, (ei1, a1) = gnn.conv(x, ei, edge_attr, return_attention_weights=True)
    z = torch.relu(gnn.conv(x, ei, edge_attr))
    _, (ei2, a2) = gnn.conv2(z, ei, edge_attr, return_attention_weights=True)
    a1m, a2m = a1.mean(-1), a2.mean(-1)
    centre = torch.tensor([em.cap.src_local, em.cap.dst_local], device=em.device)
    score_e = torch.zeros(ei.size(1), dtype=torch.float64, device=em.device)
    into_centre = torch.isin(ei[1], centre)
    score_e[into_centre] += a2m[into_centre].double()
    # second hop: edge e' -> w (w in N1) weighted by attention of w -> centre
    w_to_c: Dict[int, float] = {}
    for e in torch.nonzero(into_centre).flatten().tolist():
        w = int(ei[0, e]); w_to_c[w] = w_to_c.get(w, 0.0) + float(a2m[e])
    for e in range(ei.size(1)):
        tgt = int(ei[1, e])
        if tgt in w_to_c:
            score_e[e] += float(a1m[e]) * w_to_c[tgt]
    scores = torch.zeros(em.num_items, dtype=torch.float64, device=em.device)
    scores.index_add_(0, em.edge_item, score_e)
    return _wrap("attention", em, scores, tau, t0)


BASELINES = {
    "random": random_rank, "recency": recency, "gradient": gradient,
    "integrated_gradients": integrated_gradients, "occlusion": occlusion, "attention": attention,
}