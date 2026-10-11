"""T-GNNExplainer-style Monte-Carlo tree search over items (temporal baseline; plan E9 fallback).

T-GNNExplainer (Xia et al., ICLR 2023) explains a temporal GNN prediction by searching subsets of
the target event's candidate events with MCTS (the explorer), guided by a learned importance prior
(the navigator), and scores a subset by the model output with only that subset kept. Here:

* candidates are the explainer items (neighbour edges and history events), capped at the
  ``n_candidates`` with the largest prior; the other items are always removed;
* a tree node keeps a set of candidates and a child drops one more. Branching is limited to the
  ``branch`` kept candidates with the smallest prior (drop the least useful first);
* a leaf keeps at most ``k_max`` candidates and is scored by the anomaly loss with only those kept,
  the same preserve objective as PROVEX (the leaf keeps the alert if L >= tau);
* the navigator is the gradient prior dL/dg at g = 1, without training, instead of a learned network.

Scores rank the best leaf's items first, then the other candidates, then the remaining items, each
group by prior. ``valid`` means the best leaf alone keeps the alert.
"""
from __future__ import annotations

import math
import time

import torch

from .event_model import EventModel
from .joint_mask import Explanation


class _Node:
    __slots__ = ("keep", "children", "N", "W", "P")

    def __init__(self, keep: frozenset, P: float):
        self.keep, self.children, self.N, self.W, self.P = keep, None, 0, 0.0, P


def mcts_explain(em: EventModel, tau: float, *, n_candidates: int = 30, k_max: int = 10, n_sim: int = 64,
                 branch: int = 5, c_puct: float = 1.0) -> Explanation:
    t0 = time.perf_counter()
    n = em.num_items
    ones = torch.ones(n, dtype=em.dtype, device=em.device)
    g = ones.clone().requires_grad_(True)
    em.loss(g).backward()
    prior = g.grad.detach().to(torch.float64).cpu()
    with torch.no_grad():
        L_full = float(em.loss(ones, hard=True))
    scale = max(abs(L_full), 1e-6)
    cand = torch.argsort(prior, descending=True)[:min(n_candidates, n)].tolist()
    pr = {i: float(prior[i]) for i in cand}
    leaf_L = {}

    @torch.no_grad()
    def loss_keep(keep: frozenset) -> float:
        if keep not in leaf_L:
            m = torch.zeros(n, dtype=em.dtype, device=em.device)
            if keep:
                m[sorted(keep)] = 1
            leaf_L[keep] = float(em.loss(m, hard=True))
        return leaf_L[keep]

    def expand(node: _Node):
        drop = sorted(node.keep, key=lambda i: (pr[i], i))[:branch]           # least useful first
        p = torch.softmax(-torch.tensor([pr[i] for i in drop], dtype=torch.float64), 0).tolist()
        node.children = [_Node(node.keep - {i}, pi) for i, pi in zip(drop, p)]

    target = min(k_max, len(cand))
    root = _Node(frozenset(cand), 1.0)
    for _ in range(n_sim if n else 0):
        path, node = [root], root
        while len(node.keep) > target:
            if node.children is None:
                expand(node)
            sq = math.sqrt(node.N + 1)
            node = max(node.children, key=lambda c: (c.W / c.N if c.N else 0.0) + c_puct * c.P * sq / (1 + c.N))
            path.append(node)
        r = loss_keep(node.keep) / scale
        for v in path:
            v.N += 1; v.W += r
    leaves = [k for k in leaf_L if len(k) <= target]
    best = max(leaves, key=lambda k: (leaf_L[k], -len(k), sorted(k))) if leaves else frozenset()
    rank = torch.empty(n, dtype=torch.float64)
    rank[torch.argsort(prior)] = torch.arange(n, dtype=torch.float64) / max(n, 1)   # in [0, 1)
    scores = rank.clone()
    for i in cand:
        scores[i] += 2.0 if i in best else 1.0
    L_best = leaf_L.get(best, float("nan"))
    return Explanation(method="mcts", mode="preserve", items=em.items.tolist(), scores=scores.tolist(),
                       selected=em.items[sorted(best)].tolist() if best else [], valid=bool(L_best >= tau),
                       repaired=0, L_full=L_full, L_selected=L_best, tau=tau,
                       runtime_s=time.perf_counter() - t0,
                       extra={"n_candidates": len(cand), "k_max": target, "n_leaves": len(leaves),
                              "n_sim": n_sim})