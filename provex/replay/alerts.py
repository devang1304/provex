"""Alert-driven targets (fixes v1 defects H3 and H4: no hard-coded window, consistent node ranking).

KAIROS decides which windows are alarmed: windows in a queue whose score exceeds beta,
emitted by the vendored KAIROS evaluation as ``alerts.json``. PROVEX explains the anomalous
events inside those windows, i.e. events with loss >= tau_w, where tau_w = mean + 1.5 * std
of the window's per-event losses (upstream ``cal_anomaly_loss``, population std).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .kairos_replay import Window


@dataclass
class Alert:
    day: str
    window_file: str
    start_ns: int
    end_ns: int
    queue_id: int
    queue_score: float
    is_gt_attack_window: bool


def load_alerts(path: str | Path, day: str | None = None) -> List[Alert]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out = [Alert(r["day"], r["window_file"], int(r["start_ns"]), int(r["end_ns"]), int(r["queue_id"]),
                 float(r["queue_score"]), bool(r.get("is_gt_attack_window", False))) for r in rows]
    return [a for a in out if day is None or a.day == day]


def window_threshold(losses: Tensor, w: Window) -> float:
    x = losses[w.start_idx:w.end_idx].double().numpy()
    return float(x.mean() + 1.5 * x.std())


def match_window(alert: Alert, windows: Sequence[Window], tol_ns: int = 1_000) -> Window:
    """Windows are identified by their file-name start/end times (ns); file names carry full ns."""
    for w in windows:
        if abs(w.start_t - alert.start_ns) <= tol_ns and abs(w.end_t - alert.end_ns) <= tol_ns:
            return w
    raise KeyError(f"alerted window {alert.window_file} not found in PROVEX pass-1 windows "
                   "(replay semantics differ from KAIROS - this must not happen)")


def anomalous_events(losses: Tensor, w: Window, tau: float) -> List[int]:
    idx = torch.arange(w.start_idx, w.end_idx)
    sel = idx[losses[w.start_idx:w.end_idx] >= tau]
    order = torch.argsort(losses[sel], descending=True, stable=True)
    return sel[order].tolist()


def choose_targets(events: List[int], top_n: int, sample_n: int, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Top-N by loss plus a uniform random sample of the remaining anomalous events
    (so that reported statistics are not only about the most extreme events)."""
    top = events[:top_n]
    rest = events[top_n:]
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(rest, size=min(sample_n, len(rest)), replace=False).tolist()) if rest else []
    return top, sample


def rank_nodes(events: Sequence[int], losses: Tensor, tau: float, data) -> List[Tuple[int, int, float, float]]:
    """(gid, #anomalous events, sum of (L - tau)+, max L), deterministic order."""
    acc: Dict[int, List[float]] = {}
    for ev in events:
        L = float(losses[ev])
        for n in {int(data.src[ev]), int(data.dst[ev])}:
            acc.setdefault(n, []).append(L)
    rows = [(n, len(v), float(sum(max(0.0, x - tau) for x in v)), float(max(v))) for n, v in acc.items()]
    return sorted(rows, key=lambda r: (-r[1], -r[2], -r[3], r[0]))