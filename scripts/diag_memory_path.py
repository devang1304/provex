"""Why does zeroing the endpoints' own memory not move KAIROS's score? (read-only interpretability check)

    python scripts/diag_memory_path.py --ckpt artifact/kairos_upstream/seed_0/models/models_state.pt \
        --day-file artifact/graph_embeddings/graph_4_12.TemporalData.simple --day 4_12 \
        --events @results/compare/events/A_4_12.txt --out results/analysis/memory_path_4_12.json

TransformerConv adds a root term lin_skip(x_i) to the attention-weighted neighbour messages. Stage 1 found
that removing every neighbour edge gives exactly the type-prior score (all memory zero, no edges), and
zeroing the endpoints' memory changes nothing. This script measures, per alert event, the norm of the
endpoints' memory, of the root term and of the full first-layer output, plus the weight norms, so the
paper can state the mechanism rather than only the effect.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provex.data import load_day  # noqa: E402
from provex.explain.event_model import EventModel  # noqa: E402
from provex.model import load_checkpoint  # noqa: E402
from provex.replay.kairos_replay import KairosReplayer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ("--ckpt", "--day-file", "--day", "--events", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--max-events", type=int, default=60)
    a = ap.parse_args()
    memory, gnn, lp, hp = load_checkpoint(a.ckpt, "cpu")
    data = load_day(a.day_file)
    spec = Path(a.events[1:]).read_text().split() if a.events.startswith("@") else a.events.split(",")
    targets = sorted({int(x) for x in spec if x.strip()})[:a.max_events]
    caps = KairosReplayer(memory, gnn, lp, hp).pass2(data, targets, history_batches=0)
    c1, c2 = gnn.conv, gnn.conv2
    rows = []
    with torch.no_grad():
        for ev in targets:
            cap = caps[ev]
            em = EventModel(memory, gnn, lp, cap)
            x, s, d = em.x, cap.src_local, cap.dst_local
            ea = em._edge_attr(em.lu_off, em.ei, em.edge_t_off, em.edge_msg)
            out1 = c1(x, em.ei, ea)
            skip = c1.lin_skip(x) if getattr(c1, "lin_skip", None) is not None else torch.zeros_like(out1)
            skip0 = c1.lin_skip(torch.zeros_like(x[:1])) if getattr(c1, "lin_skip", None) is not None else None
            row = {"event": ev, "mem_norm_nodes_median": float(x.norm(dim=1).median()),
                   "n_nodes": int(x.size(0)), "n_edges": int(em.ei.size(1))}
            for side, i in (("u", s), ("v", d)):
                row[f"mem_norm_{side}"] = float(x[i].norm())
                row[f"skip_norm_{side}"] = float(skip[i].norm())
                row[f"skip_minus_bias_norm_{side}"] = float((skip[i] - skip0[0]).norm()) if skip0 is not None else 0.0
                row[f"conv1_norm_{side}"] = float(out1[i].norm())
            rows.append(row)
    keys = [k for k in rows[0] if k != "event"] if rows else []
    weights = {}
    for name, conv in (("conv1", c1), ("conv2", c2)):
        for lin in ("lin_skip", "lin_value", "lin_key", "lin_query", "lin_edge"):
            m = getattr(conv, lin, None)
            if m is not None and getattr(m, "weight", None) is not None:
                weights[f"{name}.{lin}.weight_fro"] = float(m.weight.norm())
    out = {"day": a.day, "n": len(rows), "weights": weights,
           "median": {k: float(st.median(r[k] for r in rows)) for k in keys},
           "max": {k: float(max(r[k] for r in rows)) for k in keys}, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("day", "n", "weights", "median", "max")}, indent=1))


if __name__ == "__main__":
    main()