"""Summarise explanation runs: time per method, alert classes, item counts and peak memory.

    python scripts/summarize_explain.py 'results/expl/seed_0/explanations_4_6*.jsonl'

Peak RSS is read from the shard logs next to the .jsonl files ("peak RSS <n> MB" lines).
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/expl/seed_0/explanations_*.jsonl"))
    recs = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
    if not recs:
        print("no records"); return
    rt = defaultdict(list)
    per_event = []
    for r in recs:
        tot = 0.0
        for ex in r["explanations"]:
            rt[ex["method"]].append(ex["runtime_s"]); tot += ex["runtime_s"]
        per_event.append(tot)
    print(f"files {len(files)}  events {len(recs)}  classes {dict(Counter(r['alert_class'] for r in recs))}")
    n_items = [r["n_items"] for r in recs]
    print(f"items per event: median {st.median(n_items)}  max {max(n_items)}  "
          f"(edges median {st.median(r['n_edge_items'] for r in recs)}, history median "
          f"{st.median(r['n_history_items'] for r in recs)})")
    print(f"explainer time per event (s): median {st.median(per_event):.1f}  mean {st.mean(per_event):.1f}  "
          f"max {max(per_event):.1f}  (metrics time not included)")
    for m, v in sorted(rt.items(), key=lambda kv: -st.mean(kv[1])):
        print(f"  {m:24s} n={len(v):4d}  median {st.median(v):7.2f}  mean {st.mean(v):7.2f}  max {max(v):7.2f}")
    rss, wall, capture = [], [], []
    for f in {str(Path(f).parent) for f in files}:
        for lg in glob.glob(f + "/*.log"):
            text = open(lg, encoding="utf-8").read()
            rss += [float(x) for x in re.findall(r"peak RSS (\d+) MB", text)]
            capture += [float(x) for x in re.findall(r"captures done in ([\d.]+)s", text)]
            # wall time per event = gap between consecutive progress lines (first: after the captures)
            stamps = [h * 3600 + m * 60 + s for h, m, s, line in
                      ((int(a), int(b), int(c), d) for a, b, c, d in
                       re.findall(r"^\[(\d\d):(\d\d):(\d\d)\] (.*)$", text, flags=re.M))
                      if line.startswith("captures done") or re.match(r"\[\d+/\d+\] ev", line)]
            wall += [(b - a) % 86400 for a, b in zip(stamps, stamps[1:])]
    if capture:
        print(f"capture (day replay) per worker: max {max(capture):.0f} s")
    if wall:
        print(f"wall time per event incl. metrics (s): median {st.median(wall):.0f}  mean {st.mean(wall):.0f}  "
              f"max {max(wall):.0f}  (n={len(wall)})")
    if rss:
        print(f"peak RSS per worker: max {max(rss):.0f} MB")


if __name__ == "__main__":
    main()