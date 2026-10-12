"""Pick the alert events for the technique comparison (step 5) from the stage-1 records.

    python scripts/select_compare_events.py --records "results/stage1/expl/explanations_*.jsonl" \
        --out results/compare/events

Setting A (neighbour edges only, all methods):
  every window-class event, plus a stratified sample of prior-class events (up to --prior-per-stratum
  per (day, event type)).
Setting B (with history): up to --b-per-stratum window-class events per (day, event type).
Sampling is seeded (rng 0) and the rule is written to selection.json, so the paper can state it.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prior-per-stratum", type=int, default=8); ap.add_argument("--b-per-stratum", type=int, default=6)
    a = ap.parse_args()
    recs = {}
    for f in sorted(glob.glob(a.records)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                recs[(r["event"]["day"], r["event"]["event"])] = r
    rng = np.random.default_rng(0)
    strata = defaultdict(list)
    for (day, ev), r in sorted(recs.items()):
        strata[(r["alert_class"], day, r["event"]["type"])].append(ev)

    def pick(evs, k):
        return sorted(evs) if len(evs) <= k else sorted(rng.choice(evs, size=k, replace=False).tolist())

    A, B, rows = defaultdict(list), defaultdict(list), []
    for (cls, day, typ), evs in sorted(strata.items()):
        if cls == "window":
            a_sel, b_sel = sorted(evs), pick(evs, a.b_per_stratum)
        elif cls == "prior":
            a_sel, b_sel = pick(evs, a.prior_per_stratum), []
        else:
            a_sel, b_sel = sorted(evs), []
        A[day] += a_sel; B[day] += b_sel
        rows.append({"class": cls, "day": day, "type": typ, "n": len(evs), "A": len(a_sel), "B": len(b_sel)})
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    for name, sel in (("A", A), ("B", B)):
        for day, evs in sel.items():
            if evs:
                (O / f"{name}_{day}.txt").write_text("\n".join(str(e) for e in sorted(evs)) + "\n")
    summary = {"rule": {"A": f"all window-class events + up to {a.prior_per_stratum} prior-class events per (day, type)",
                        "B": f"up to {a.b_per_stratum} window-class events per (day, type)", "rng": 0},
               "strata": rows, "n_A": sum(map(len, A.values())), "n_B": sum(map(len, B.values()))}
    (O / "selection.json").write_text(json.dumps(summary, indent=1))
    for r in rows:
        print(f"{r['class']:12s} {r['day']:5s} {r['type']:18s} n={r['n']:4d}  A={r['A']:3d}  B={r['B']:3d}")
    print(f"total A={summary['n_A']}  B={summary['n_B']}")


if __name__ == "__main__":
    main()
