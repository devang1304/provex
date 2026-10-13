"""Calibration of the variational explainer's uncertainty (fixes v1 defect C4).

For a small subset S of items (n <= 14), all 2^n binary keep-masks are enumerated with the
remaining items fixed at their posterior mode. The exact posterior is

    p(b | alert)  proportional to  prod_k Bern(b_k; pi0_k) * sigmoid((L(b) - tau) / T),

where pi0_k = P(m_k > 0.5) under the temporal prior N(mu0_k, 1), i.e. Phi(mu0_k). This is
compared with the variational inclusion probabilities pi_k = Phi(mu_k / sigma_k).
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..explain.event_model import EventModel
from ..explain.joint_mask import Explanation


def _phi(x: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr
    return ndtr(np.asarray(x, dtype=float))


@torch.no_grad()
def posterior_calibration(em: EventModel, ex: Explanation, tau: float, T: float = 1.0, n: int = 10) -> Dict:
    pi = np.asarray(ex.scores, dtype=float)
    mu0 = np.asarray(ex.extra["prior_mu"], dtype=float)
    n = min(n, len(pi))
    if n == 0:
        return {"items": [], "pi_var": [], "pi_exact": [], "abs_err": [], "n": 0}
    S = np.argsort(-(pi * (1 - pi)), kind="stable")[:n]          # most uncertain items
    base = torch.tensor(pi > 0.5, dtype=em.dtype, device=em.device)
    prior_keep = np.clip(_phi(mu0[S]), 1e-9, 1 - 1e-9)
    masks = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    logw = np.empty(len(masks))
    for i, b in enumerate(masks):
        g = base.clone()
        g[torch.as_tensor(S, device=em.device)] = torch.as_tensor(b, dtype=em.dtype, device=em.device)
        L = em.loss(g, hard=True)
        ll = float(F.logsigmoid((L - tau) / T))
        lp = float(np.sum(b * np.log(prior_keep) + (1 - b) * np.log(1 - prior_keep)))
        logw[i] = ll + lp
    w = np.exp(logw - logw.max()); w /= w.sum()
    exact = (w[:, None] * masks).sum(0)
    return {"items": [int(em.items[k]) for k in S], "pi_var": pi[S].tolist(), "pi_exact": exact.tolist(),
            "abs_err": np.abs(pi[S] - exact).tolist(), "n": int(n)}


def ece(pairs: Sequence[tuple], bins: int = 10) -> float:
    """Expected calibration error between predicted (variational) and exact marginals."""
    p = np.array([a for a, _ in pairs]); q = np.array([b for _, b in pairs])
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            tot += m.mean() * abs(p[m].mean() - q[m].mean())
    return float(tot)