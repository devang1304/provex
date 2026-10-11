"""Aggregate E6 (stability across explainer seeds, calibration) from explanation records.

    python scripts/aggregate_e6.py --records 'results/e6/*.jsonl' --out results
"""
import argparse
import glob
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from provex.evaluate.calibration import ece
from provex.evaluate.metrics import agreement, stability


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", required=True); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    runs = defaultdict(lambda: defaultdict(dict))        # method -> (day, event) -> seed -> scores
    pairs = []
    latest = {}                                          # a re-run appends: keep the last record
    for p in glob.glob(a.records):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            for ex in r["explanations"]:            # event ids are per day, so key by (day, event)
                latest[(r["event"]["day"], r["event"]["event"], ex["method"], ex["seed"])] = ex
    for (day, ev, m, seed), ex in latest.items():
        if "scores_all" in ex:
            runs[m][(day, ev)][seed] = ex["scores_all"]
        cal = ex.get("metrics", {}).get("calibration")
        if cal:
            pairs += list(zip(cal["pi_var"], cal["pi_exact"]))
    stab = {}
    for m, evs in runs.items():
        jac, tau = [], []
        for ev, seeds in evs.items():
            if len(seeds) < 2:
                continue
            s = stability([seeds[k] for k in sorted(seeds)], k=5)
            jac.append(s["topk_jaccard"]); tau.append(s["kendall_tau_b"])
        if jac:
            stab[m] = {"topk_jaccard": float(np.nanmean(jac)), "kendall_tau_b": float(np.nanmean(tau)), "n_events": len(jac)}
    # cross-method agreement on seed 0
    agree = {}
    methods = sorted(runs)
    for m1, m2 in itertools.combinations(methods, 2):
        vals = [agreement(runs[m1][ev][0], runs[m2][ev][0], k=5) for ev in runs[m1]
                if ev in runs[m2] and 0 in runs[m1][ev] and 0 in runs[m2][ev]]
        if vals:
            agree[f"{m1}|{m2}"] = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "stability.json").write_text(json.dumps(stab, indent=1))
    (out / "agreement.json").write_text(json.dumps(agree, indent=1))
    cal = {"n_pairs": len(pairs), "ece": ece(pairs) if pairs else None,
           "max_abs_err": float(max(abs(x - y) for x, y in pairs)) if pairs else None,
           "mean_abs_err": float(np.mean([abs(x - y) for x, y in pairs])) if pairs else None}
    (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps({"stability": stab, "calibration": cal}, indent=1))


if __name__ == "__main__":
    main()