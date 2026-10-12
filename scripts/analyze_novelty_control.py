"""H1 control: does the story's triage flag do the work alone, or does KAIROS's alert matter?

    python scripts/analyze_novelty_control.py --graphs artifact/graph_embeddings \
        --names artifact/explanations/node_mapping.json --kairos artifact/kairos_upstream/seed_0 \
        --pass1 results/pass1 --out results/analysis

The story's window flag (make_alert_stories.py, evaluate_stories.py) is a novelty rule: a window is
worth an analyst's time if one of its flagged actions touches an external IP address first seen
today, or runs a program first seen today. This script applies the rule to EVERY 15-minute window
of the validation and test days, not only to the windows KAIROS alerted on, and compares:

  kairos            KAIROS's alert (queue score > beta, from alerts.json)
  kairos+story      KAIROS's alert, then the story flag (the H1 simulated analyst)
  novelty_all       the flag computed on all events of the window (no detector at all)
  novelty_flagged   the flag computed on the window's flagged events (score >= window threshold),
                    without KAIROS's queue aggregation
  queued+story      windows in any KAIROS queue (any queue score), then the story flag

Window truth is KAIROS's own ground-truth windows (detection_metrics.json), as in the detection table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provex.data import EDGE_TYPES, load_day, load_node_names  # noqa: E402
from provex.replay.alerts import window_threshold  # noqa: E402
from provex.replay.kairos_replay import Window  # noqa: E402

EXECUTE = EDGE_TYPES.index("EVENT_EXECUTE")
NEVER = 99


def day_order(day: str) -> int:
    return int(day.split("_")[1])


def ip_of(label):
    if not label or not label.startswith("netflow: "):
        return None
    return label[len("netflow: "):].rsplit(":", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    for k in ("--graphs", "--names", "--kairos", "--pass1", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--internal", default="128.55.12.,127.,0.0.0.0,::1,localhost")
    a = ap.parse_args()
    internal = tuple(x for x in a.internal.split(",") if x)
    names = load_node_names(a.names)
    K = Path(a.kairos)
    det = json.loads((K / "detection_metrics.json").read_text())
    wins_rows = json.loads((K / "windows.json").read_text())
    alerts = json.loads((K / "alerts.json").read_text())
    val_day = det["config"]["validation_graph"][len("graph_"):]
    test_days = [g[len("graph_"):] for g in det["config"]["queue_test_graphs"]]
    eval_days = [val_day] + test_days
    gt = {}
    for att, g in det["ground_truth"].items():
        for w in g.get("windows", []):
            gt[(g["graph"][len("graph_"):], w)] = att
    alerted = {(x["day"], x["window_file"]) for x in alerts}
    queued = set()
    for d in eval_days:
        p = K / f"graph_{d}_history_list"
        if p.exists():
            for q in torch.load(p, weights_only=False):
                for w in q:
                    queued.add((d, Path(str(w.get("name", ""))).name))

    # first day each node and each IP address appears (from 4_2, the first day of data)
    days = [f"4_{i}" for i in range(2, max(day_order(d) for d in test_days) + 1)]
    N = max(names) + 1 if names else 1
    first_day = np.full(N, NEVER, dtype=np.int16)
    ip_first, graphs = {}, {}
    for d in days:
        p = Path(a.graphs) / f"graph_{d}.TemporalData.simple"
        if not p.exists():
            continue
        g = load_day(p)
        src, dst = g.src.numpy(), g.dst.numpy()
        nodes = np.unique(np.concatenate([src, dst]))
        if nodes.max() >= len(first_day):
            first_day = np.concatenate([first_day, np.full(nodes.max() + 1 - len(first_day), NEVER, dtype=np.int16)])
        new = nodes[first_day[nodes] == NEVER]
        first_day[new] = day_order(d)
        for n in new.tolist():
            ip = ip_of(names.get(n, ""))
            if ip is not None:
                ip_first.setdefault(ip, day_order(d))
        if d in eval_days:
            graphs[d] = (src, dst, g.msg[:, 16:23].argmax(1).numpy())
    N = len(first_day)
    is_subject = np.zeros(N, dtype=bool)
    ext_ip_day = np.full(N, NEVER, dtype=np.int16)
    for n, lab in names.items():
        if n >= N:
            continue
        kind, _, name = lab.partition(": ")
        is_subject[n] = kind == "subject"
        if kind == "netflow" and not name.startswith(internal):
            ip = ip_of(lab)
            if ip is not None:
                ext_ip_day[n] = ip_first.get(ip, NEVER)

    rows = []
    for d in eval_days:
        if d not in graphs:
            continue
        src, dst, et = graphs[d]
        do = day_order(d)
        new = first_day == do
        # same definitions as make_alert_stories.py: external IP new today, or a program new today runs
        flag_ev = (ext_ip_day[src] == do) | (ext_ip_day[dst] == do) \
            | ((et == EXECUTE) & (new[src] | new[dst])) | (is_subject[src] & new[src]) | (is_subject[dst] & new[dst])
        P = torch.load(Path(a.pass1) / f"pass1_{d}.pt", weights_only=False)
        L = P["losses"].double().numpy()
        starts = [(r["start_ns"], r["window_file"]) for r in wins_rows if r["day"] == d]
        for w in (Window(**x) for x in P["windows"]):
            wf = next((f for s, f in starts if abs(s - w.start_t) <= 1000), None)
            if wf is None:
                continue
            tau = window_threshold(P["losses"], w)
            hot = L[w.start_idx:w.end_idx] >= tau
            fl = flag_ev[w.start_idx:w.end_idx]
            rows.append({"day": d, "window": wf, "split": "validation" if d == val_day else "test",
                         "gt_attack": gt.get((d, wf)), "kairos": (d, wf) in alerted, "queued": (d, wf) in queued,
                         "novel_all": bool(fl.any()), "novel_flagged": bool((fl & hot).any()),
                         "n_events": int(w.end_idx - w.start_idx), "n_flagged": int(hot.sum()),
                         "n_novel_flagged": int((fl & hot).sum())})

    rules = {
        "kairos": lambda r: r["kairos"],
        "kairos+story": lambda r: r["kairos"] and r["novel_flagged"],
        "novelty_all": lambda r: r["novel_all"],
        "novelty_flagged": lambda r: r["novel_flagged"],
        "queued+story": lambda r: r["queued"] and r["novel_flagged"],
    }
    attacks = sorted(set(gt.values()))
    test = [r for r in rows if r["split"] == "test"]
    n_gt = sum(1 for r in test if r["gt_attack"])
    summary = {}
    for name, rule in rules.items():
        flagged = [r for r in test if rule(r)]
        tp = sum(1 for r in flagged if r["gt_attack"])
        summary[name] = {"test_windows_flagged": len(flagged), "tp": tp, "fp": len(flagged) - tp, "fn": n_gt - tp,
                         "precision": tp / len(flagged) if flagged else None, "recall": tp / n_gt if n_gt else None,
                         "attacks_detected": sorted({r["gt_attack"] for r in flagged if r["gt_attack"]}),
                         "validation_windows_flagged": sum(1 for r in rows if r["split"] == "validation" and rule(r)),
                         "fp_by_day": {d: sum(1 for r in flagged if r["day"] == d and not r["gt_attack"]) for d in test_days}}
    out = {"n_test_windows": len(test), "n_validation_windows": len(rows) - len(test), "n_gt_windows": n_gt,
           "attacks": attacks, "rules": summary, "windows": rows}
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    (O / "novelty_control.json").write_text(json.dumps(out, indent=1))
    L = ["# H1 control: the story flag with and without KAIROS (generated by scripts/analyze_novelty_control.py)", "",
         f"{len(test)} test windows ({', '.join(test_days)}) and {out['n_validation_windows']} validation windows ({val_day}). "
         f"Truth: KAIROS ground-truth windows ({n_gt} windows, {len(attacks)} attacks).", "",
         "| rule | windows flagged (test) | TP | FP | FN | precision | recall | attacks caught | validation windows flagged |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        pr = "n/a" if s["precision"] is None else f"{s['precision']:.2f}"
        rc = "n/a" if s["recall"] is None else f"{s['recall']:.2f}"
        L.append(f"| {name} | {s['test_windows_flagged']} | {s['tp']} | {s['fp']} | {s['fn']} | {pr} | {rc} | "
                 f"{len(s['attacks_detected'])}/{len(attacks)} ({', '.join(s['attacks_detected']) or 'none'}) | "
                 f"{s['validation_windows_flagged']} |")
    L += ["", "False-positive windows per test day:", "", "| rule | " + " | ".join(test_days) + " |",
          "|---|" + "---|" * len(test_days)]
    for name, s in summary.items():
        L.append(f"| {name} | " + " | ".join(str(s["fp_by_day"][d]) for d in test_days) + " |")
    (O / "novelty_control.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()