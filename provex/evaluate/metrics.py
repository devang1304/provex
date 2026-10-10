"""Metrics in anomaly terms (replace v1's comprehensiveness/sufficiency on p(true label)).

L = CE(event type) is the anomaly score, and tau is the alert threshold of the event's window.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch import Tensor

from ..explain.event_model import EventModel

KS = (1, 3, 5, 10, 20)


def _order(scores: Sequence[float]) -> List[int]:
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    return list(np.argsort(-s, kind="stable"))


@torch.no_grad()
def deletion_curve(em: EventModel, scores, ks=KS) -> Dict[int, float]:
    """L after removing the top-k items (exact hard deletion)."""
    order = _order(scores)
    out = {}
    for k in ks:
        g = torch.ones(em.num_items, dtype=em.dtype, device=em.device)
        g[order[:k]] = 0
        out[k] = float(em.loss(g, hard=True))
    return out


@torch.no_grad()
def insertion_curve(em: EventModel, scores, ks=KS) -> Dict[int, float]:
    """L keeping only the top-k items (all other neighbour edges and history events removed)."""
    order = _order(scores)
    out = {}
    for k in ks:
        g = torch.zeros(em.num_items, dtype=em.dtype, device=em.device)
        g[order[:k]] = 1
        out[k] = float(em.loss(g, hard=True))
    return out


@torch.no_grad()
def channel_decomposition(em: EventModel) -> Dict[str, float]:
    """How much of the anomaly each channel can account for (exact hard deletions)."""
    ones = torch.ones(em.num_items, dtype=em.dtype, device=em.device)
    zeros = torch.zeros_like(ones)
    out = {"full": float(em.loss(ones, hard=True)),
           "no_neighbour_edges": float(em.loss(ones, hard=True, edge_g=zeros)),
           "no_history": float(em.loss(zeros, hard=True, edge_g=ones)),
           "nothing": float(em.loss(zeros, hard=True))}
    x0 = em.x
    try:
        em.x = em.x.clone()
        for r in (em.cap.src_local, em.cap.dst_local):
            em.x[r] = 0
        out["memory_zero_uv"] = float(em.loss(None))
        em.x = torch.zeros_like(x0)
        out["prior_only"] = float(em.loss(zeros, hard=True, history=False))   # no memory, no edges
    finally:
        em.x = x0
    return out


def alert_class(ch: Dict[str, float], tau: float) -> str:
    """prior: anomalous with no memory and no edges (event-type prior alone);
    window: some removal inside the history window drops it below tau (counterfactual exists);
    long_memory: neither, so the anomaly comes from memory accumulated before the window."""
    if ch["prior_only"] >= tau:
        return "prior"
    if ch["nothing"] < tau:
        return "window"
    return "long_memory"

def normalised_dloss(L_full: float, L_k: float, L_floor: float) -> float:
    den = L_full - L_floor
    return float("nan") if abs(den) < 1e-9 else (L_full - L_k) / den


def alert_flip(L_after_removal: float, tau: float) -> bool:
    return L_after_removal < tau


def alert_keep(L_kept_only: float, tau: float) -> bool:
    return L_kept_only >= tau


@torch.no_grad()
def min_cf_size(em: EventModel, scores, tau: float, k_max: int = 256) -> Optional[int]:
    """Smallest k such that removing the top-k items drops L below tau (None if > k_max)."""
    order = _order(scores)
    g = torch.ones(em.num_items, dtype=em.dtype, device=em.device)
    for k in range(1, min(k_max, em.num_items) + 1):
        g[order[k - 1]] = 0
        if float(em.loss(g, hard=True)) < tau:
            return k
    return None


def topk_set(scores, k: int) -> set:
    return set(_order(scores)[:k])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def stability(score_runs: List[Sequence[float]], k: int = 5) -> Dict[str, float]:
    """Mean pairwise top-k Jaccard and Kendall tau_b across seeds."""
    jac, tau = [], []
    for a, b in itertools.combinations(score_runs, 2):
        jac.append(jaccard(topk_set(a, k), topk_set(b, k)))
        tau.append(kendalltau(np.nan_to_num(a), np.nan_to_num(b)).statistic)
    return {"topk_jaccard": float(np.mean(jac)) if jac else float("nan"),
            "kendall_tau_b": float(np.nanmean(tau)) if tau else float("nan")}


def agreement(a: Sequence[float], b: Sequence[float], k: int = 5) -> Dict[str, float]:
    return {"spearman": float(spearmanr(np.nan_to_num(a), np.nan_to_num(b)).statistic),
            "topk_jaccard": jaccard(topk_set(a, k), topk_set(b, k))}


def entity_scores(item_events: Sequence[int], scores: Sequence[float], data, exclude: set) -> Dict[int, float]:
    """Entity (global node id) score = max score over items incident to it."""
    out: Dict[int, float] = {}
    for ev, s in zip(item_events, scores):
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        for n in (int(data.src[ev]), int(data.dst[ev])):
            if n in exclude:
                continue
            out[n] = max(out.get(n, -math.inf), float(s))
    return out


def precision_at_k(ent: Dict[int, float], gt: set, k: int) -> float:
    top = sorted(ent, key=lambda n: -ent[n])[:k]
    return sum(n in gt for n in top) / max(1, len(top))


def entity_auroc(ent: Dict[int, float], gt: set) -> float:
    from sklearn.metrics import roc_auc_score
    ys = [n in gt for n in ent]
    if all(ys) or not any(ys):
        return float("nan")
    return float(roc_auc_score(ys, list(ent.values())))