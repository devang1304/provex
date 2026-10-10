"""Objectives that explain the anomaly (fixes v1 defect C2).

v1 minimised CE(f(masked), y_true): it searched for evidence that the event looks *normal*.
Here the quantity explained is the anomaly score L = CE(logits, y_true), and the alert holds
while L >= tau (the window threshold).

    preserve        keep a small item set that still sustains the alert:
                    violation = min(L_full, tau + margin) - L(kept)        (want <= 0)
    counterfactual  remove a small item set that drops the event below the threshold:
                    violation = L(remaining) - (tau - margin)              (want <= 0)

The penalties are softplus(violation / T) * T. Softplus keeps gradients alive (a ReLU hinge
zeroes them at the start), and working in loss space avoids -log(1 - p), whose gradient
vanishes for p ~ 1e-11.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

MODES = ("preserve", "counterfactual")


def violation(mode: str, L: Tensor, L_full: float, tau: float, margin: float) -> Tensor:
    if mode == "preserve":
        return min(L_full, tau + margin) - L
    if mode == "counterfactual":
        return L - (tau - margin)
    raise ValueError(mode)


def penalty(mode: str, L: Tensor, L_full: float, tau: float, margin: float, T: float = 1.0) -> Tensor:
    return F.softplus(violation(mode, L, L_full, tau, margin) / T) * T


def is_valid(mode: str, L: float, tau: float) -> bool:
    return L >= tau if mode == "preserve" else L < tau