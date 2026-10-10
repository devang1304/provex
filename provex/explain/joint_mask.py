"""Joint explainer over neighbour edges and history events, with hard-concrete gates.

One gate per item (absolute event index). Gates follow the hard-concrete relaxation of
Louizos et al. (beta = 2/3, gamma = -0.1, zeta = 1.1), so exact 0s and 1s occur. The
optimisation minimises the expected number of kept (preserve) or removed (counterfactual)
items, subject to the anomaly constraint in ``objectives``. A persistent Lagrange multiplier
is updated by dual ascent.

The returned explanation is always checked with exact hard deletion. If it is invalid, a
greedy repair adds back (preserve) or removes more (counterfactual) items in score order,
and the repair is reported.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor

from . import objectives
from .event_model import EventModel

BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1


def hard_concrete_sample(log_alpha: Tensor) -> Tensor:
    u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / BETA)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


def hard_concrete_mean(log_alpha: Tensor) -> Tensor:
    return (torch.sigmoid(log_alpha) * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


def expected_l0(log_alpha: Tensor) -> Tensor:
    return torch.sigmoid(log_alpha - BETA * math.log(-GAMMA / ZETA))


@dataclass
class Explanation:
    method: str
    mode: str
    items: List[int]                 # absolute event ids, aligned with scores
    scores: List[float]              # higher = more important for the chosen mode
    selected: List[int]              # absolute event ids in the explanation set
    valid: bool
    repaired: int
    L_full: float
    L_selected: float
    tau: float
    runtime_s: float
    extra: Dict = field(default_factory=dict)


class JointMaskExplainer:
    def __init__(self, mode: str = "preserve", *, epochs: int = 300, lr: float = 0.05,
                 margin: float = 0.5, T: float = 1.0, dual_lr: float = 0.5, size_coef: float = 1.0,
                 history: bool = True, seed: int = 0, max_repair: int = 512):
        assert mode in objectives.MODES
        self.mode, self.epochs, self.lr, self.margin, self.T = mode, epochs, lr, margin, T
        self.dual_lr, self.size_coef, self.history, self.seed = dual_lr, size_coef, history, seed
        self.max_repair = max_repair

    def _gates_for(self, z: Tensor, em: EventModel) -> Tensor:
        if self.history:
            return z
        g = torch.ones_like(z)
        return torch.where(em.is_edge_item, z, g)

    def explain(self, em: EventModel, tau: float) -> Explanation:
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        n = em.num_items
        ones = torch.ones(n, dtype=em.dtype, device=em.device)
        with torch.no_grad():
            L_full = float(em.loss(ones, hard=True))
            L_floor = float(em.loss(torch.zeros_like(ones), hard=True))
        intrinsic = L_floor >= tau        # alert holds even with every item removed
        if intrinsic:
            # preserve: the minimal sustaining set is empty; counterfactual: no item set can flip
            # the alert. Report it instead of optimising (context does not explain this anomaly).
            nan = [float("nan")] * n
            return Explanation(
                method="joint" if self.history else "joint-edges", mode=self.mode, items=em.items.tolist(),
                scores=nan, selected=[], valid=(self.mode == "preserve"), repaired=0, L_full=L_full,
                L_selected=L_floor, tau=tau, runtime_s=time.perf_counter() - t0,
                extra={"intrinsic": True, "L_floor": L_floor, "cf_feasible": False, "n_items": n})
        log_alpha = torch.full((n,), 3.0, dtype=em.dtype, device=em.device, requires_grad=True)
        opt = torch.optim.Adam([log_alpha], lr=self.lr)
        lam = 1.0
        active = torch.ones(n, dtype=torch.bool, device=em.device) if self.history else em.is_edge_item.clone()
        for _ in range(self.epochs):
            opt.zero_grad()
            z = self._gates_for(hard_concrete_sample(log_alpha), em)
            L = em.loss(z)
            l0 = expected_l0(log_alpha)[active]
            size = l0.sum() if self.mode == "preserve" else (1 - l0).sum()
            pen = objectives.penalty(self.mode, L, L_full, tau, self.margin, self.T)
            (self.size_coef * size / max(1, int(active.sum())) + lam * pen).backward()
            opt.step()
            with torch.no_grad():
                v = float(objectives.violation(self.mode, L, L_full, tau, self.margin))
                lam = max(0.0, lam + self.dual_lr * v)
        with torch.no_grad():
            gate = self._gates_for(hard_concrete_mean(log_alpha), em)
            imp = gate if self.mode == "preserve" else 1 - gate
            imp = torch.where(active, imp, torch.full_like(imp, -1.0))
            keep = gate > 0.5
            valid, L_sel, repaired, keep = self._check_and_repair(em, keep, imp, tau, active)
        sel_mask = keep if self.mode == "preserve" else ~keep
        sel_mask = sel_mask & active
        return Explanation(
            method="joint" if self.history else "joint-edges", mode=self.mode,
            items=em.items.tolist(), scores=imp.tolist(),
            selected=em.items[sel_mask.cpu()].tolist(), valid=valid, repaired=repaired,
            L_full=L_full, L_selected=L_sel, tau=tau, runtime_s=time.perf_counter() - t0,
            extra={"lambda": lam, "n_items": n, "n_active": int(active.sum()), "intrinsic": False,
                   "L_floor": L_floor, "cf_feasible": True})

    @torch.no_grad()
    def _check_and_repair(self, em: EventModel, keep: Tensor, imp: Tensor, tau: float, active: Tensor):
        def L_of(k):
            return float(em.loss(k.to(em.dtype), hard=True))
        L_sel = L_of(keep)
        repaired = 0
        order = torch.argsort(imp, descending=True).tolist()
        if self.mode == "preserve":
            for idx in order:
                if objectives.is_valid("preserve", L_sel, tau) or repaired >= self.max_repair:
                    break
                if active[idx] and not keep[idx]:
                    keep[idx] = True; repaired += 1; L_sel = L_of(keep)
        else:
            for idx in order:
                if objectives.is_valid("counterfactual", L_sel, tau) or repaired >= self.max_repair:
                    break
                if active[idx] and keep[idx]:
                    keep[idx] = False; repaired += 1; L_sel = L_of(keep)
        return objectives.is_valid(self.mode, L_sel, tau), L_sel, repaired, keep