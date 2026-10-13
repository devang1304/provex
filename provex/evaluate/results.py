"""Explanation records, schema v2: global ids and labels throughout (fixes v1 defect C1)."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, Optional

from ..data import EDGE_TYPES
from ..explain.joint_mask import Explanation
from ..replay.capture import EventCapture

SCHEMA_VERSION = 2


def _label(names: Dict[int, str], gid: int) -> str:
    return names.get(int(gid), f"node {int(gid)}")


def event_record(cap: EventCapture, data, names: Dict[int, str], day: str) -> dict:
    return {"day": day, "event": cap.event, "src": cap.src, "dst": cap.dst,
            "src_label": _label(names, cap.src), "dst_label": _label(names, cap.dst),
            "type": EDGE_TYPES[cap.y], "t": cap.t, "loss": cap.loss_batch, "tau": cap.tau,
            "predicted_type": EDGE_TYPES[int(cap.logits_batch.argmax())],
            "n_neighbour_edges": int(cap.edge_ev.numel()),
            "n_history_batches": len(cap.hist.steps) if cap.hist is not None else 0}


def explanation_record(cap: EventCapture, ex: Explanation, data, names: Dict[int, str], *,
                       is_edge, is_hist, top_k: int = 25, metrics: Optional[dict] = None,
                       seed: int = 0) -> dict:
    order = sorted(range(len(ex.items)), key=lambda k: -(ex.scores[k] if not math.isnan(ex.scores[k]) else -1e30))
    per_item_extra = {k: v for k, v in ex.extra.items() if isinstance(v, list) and len(v) == len(ex.items)}
    items = []
    for k in order[:top_k]:
        ev = int(ex.items[k])
        row = {"ev": ev, "src": int(data.src[ev]), "dst": int(data.dst[ev]),
               "src_label": _label(names, int(data.src[ev])), "dst_label": _label(names, int(data.dst[ev])),
               "type": EDGE_TYPES[int(data.msg[ev, 16:23].argmax())] if data.msg.size(1) == 39 else None,
               "t": int(data.t[ev]), "score": float(ex.scores[k]),
               "channels": [c for c, f in (("edge", bool(is_edge[k])), ("history", bool(is_hist[k]))) if f]}
        for key, vals in per_item_extra.items():
            row[key] = vals[k]
        items.append(row)
    scalar_extra = {k: v for k, v in ex.extra.items() if k not in per_item_extra}
    return {"schema_version": SCHEMA_VERSION, "method": ex.method, "mode": ex.mode, "seed": seed,
            "valid": ex.valid, "repaired": ex.repaired, "L_full": ex.L_full, "L_selected": ex.L_selected,
            "tau": ex.tau, "runtime_s": ex.runtime_s, "n_items": len(ex.items),
            "selected": [int(e) for e in ex.selected], "top_items": items, "extra": scalar_extra,
            "metrics": metrics or {}}