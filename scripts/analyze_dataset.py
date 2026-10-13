"""Phase A: what CADETS E3 looks like to KAIROS, and how KAIROS scores it (no model re-runs).

    python scripts/analyze_dataset.py --graphs artifact/graph_embeddings \
        --names artifact/explanations/node_mapping.json --gt-dir artifact/kairos_upstream/ground_truth \
        --kairos artifact/kairos_upstream/seed_0 --pass1 results/pass1 --out results/analysis

Data, for every day 4_2 .. 4_13:
- events, active and new nodes by type, event-type mix, gaps, hubs.

Attacks:
- events touching ground-truth (GT) nodes, inside and outside the attack interval;
- their types, and the chain in time order;
- whether GT entities are new that day or were seen in training.
KAIROS keys nodes by name, so a GT node such as nginx also carries benign activity.

Model, using seed-0 scores from PROVEX pass 1 (identical to KAIROS):
- score distributions by day and event type, and where attack events rank;
- how often anomalous events involve new entities;
- the alert mechanics of the GT windows: window statistics and queue scores against beta.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provex.data import EDGE_TYPES, load_day, load_node_names, split_label  # noqa: E402
from provex.evaluate.ground_truth import load_gt_nodes  # noqa: E402
from provex.replay.alerts import window_threshold  # noqa: E402
from provex.replay.kairos_replay import Window  # noqa: E402

DAYS = [f"4_{d}" for d in range(2, 14)]
TRAIN = {"4_2", "4_3", "4_4"}
KINDS = ["subject", "file", "netflow"]
TZ = pytz.timezone("US/Eastern")


def ts(ns) -> str:
    return datetime.fromtimestamp(int(ns) / 1e9, TZ).strftime("%m-%d %H:%M:%S")


def eastern_ns(s: str) -> int:
    return int(TZ.localize(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")).timestamp() * 1e9)


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def frac(mask_num, mask_den=None):
    if mask_den is None:
        return float(mask_num.mean()) if len(mask_num) else None
    return float(mask_num[mask_den].mean()) if mask_den.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True); ap.add_argument("--names", required=True)
    ap.add_argument("--gt-dir", required=True); ap.add_argument("--kairos", required=True)
    ap.add_argument("--pass1", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--chain", type=int, default=80, help="chain steps kept per attack")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = load_node_names(a.names)
    lab = lambda i: names.get(int(i), f"node {int(i)}")  # noqa: E731
    gt_by_day = load_gt_nodes(a.gt_dir)
    K = Path(a.kairos)
    det = json.loads((K / "detection_metrics.json").read_text())
    win_rows = json.loads((K / "windows.json").read_text())
    alerts = json.loads((K / "alerts.json").read_text())
    attack_of_day = {v["graph"][len("graph_"):]: (k, v) for k, v in det.get("ground_truth", {}).items()}

    data = {}
    for day in DAYS:
        p = Path(a.graphs) / f"graph_{day}.TemporalData.simple"
        if p.exists():
            d = load_day(p)
            data[day] = (d.src.numpy().astype(np.int64), d.dst.numpy().astype(np.int64),
                         d.t.numpy().astype(np.int64), d.msg[:, 16:23].argmax(1).numpy().astype(np.int64))
    N = max([max(names) + 1] + [int(max(s.max(), d.max())) + 1 for s, d, _, _ in data.values()])
    kind = np.full(N, -1, np.int8)
    for i, label in names.items():
        k = split_label(label)[0]
        kind[i] = KINDS.index(k) if k in KINDS else -1
    first_day = np.full(N, -1, np.int16)
    in_train = np.zeros(N, bool)
    R = {"n_labelled_nodes": len(names),
         "labelled_by_kind": {k: int((kind == j).sum()) for j, k in enumerate(KINDS)},
         "days": {}, "attacks": {}, "model": {}}

    for di, day in enumerate(DAYS):
        if day not in data:
            continue
        src, dst, t, et = data[day]
        nodes = np.unique(np.concatenate([src, dst]))
        new = nodes[first_day[nodes] < 0]
        first_day[new] = di
        if day in TRAIN:
            in_train[nodes] = True
        deg = np.bincount(np.concatenate([src, dst]), minlength=N)
        top = np.argsort(-deg)[:10]
        gaps = np.diff(t) / 6e10
        R["days"][day] = {
            "events": int(len(src)), "first": ts(t[0]), "last": ts(t[-1]),
            "active_nodes": int(len(nodes)),
            "active_by_kind": {k: int((kind[nodes] == j).sum()) for j, k in enumerate(KINDS)},
            "new_nodes": int(len(new)), "new_by_kind": {k: int((kind[new] == j).sum()) for j, k in enumerate(KINDS)},
            "event_types": {EDGE_TYPES[j]: int(c) for j, c in enumerate(np.bincount(et, minlength=len(EDGE_TYPES)))},
            "max_gap_min": float(gaps.max()) if len(gaps) else 0.0, "gaps_over_15min": int((gaps > 15).sum()),
            "hubs": [[lab(i), int(deg[i])] for i in top]}

        touch = inside = None
        if day in attack_of_day and day in gt_by_day:
            att, info = attack_of_day[day]
            gt = np.array(sorted(g for g in gt_by_day[day] if g < N), dtype=np.int64)
            isgt = np.zeros(N, bool); isgt[gt] = True
            lo, hi = (eastern_ns(s) for s in info["interval_us_eastern"])
            in_iv = (t >= lo) & (t <= hi)
            touch = isgt[src] | isgt[dst]
            inside = touch & in_iv
            idx_in = np.nonzero(inside)[0]
            chain = []
            for e in idx_in:
                key = (int(src[e]), int(et[e]), int(dst[e]))
                if chain and chain[-1][0] == key:
                    chain[-1][1] += 1; chain[-1][3] = int(t[e])
                else:
                    chain.append([key, 1, int(t[e]), int(t[e])])
            rows = []
            for g in gt:
                m = (src == g) | (dst == g)
                rows.append({"node": int(g), "label": lab(g), "kind": KINDS[kind[g]] if kind[g] >= 0 else "?",
                             "events_today": int(m.sum()), "events_in_interval": int((m & in_iv).sum()),
                             "first_seen": DAYS[first_day[g]] if first_day[g] >= 0 else None,
                             "seen_in_training": bool(in_train[g])})
            R["attacks"][att] = {
                "day": day, "interval": info["interval_us_eastern"], "gt_nodes": int(len(gt)),
                "gt_nodes_active_today": sum(r["events_today"] > 0 for r in rows),
                "gt_nodes_new_today": sum(r["first_seen"] == day for r in rows),
                "gt_nodes_seen_in_training": sum(r["seen_in_training"] for r in rows),
                "events_in_interval": int(in_iv.sum()), "gt_events_today": int(touch.sum()),
                "gt_events_in_interval": int(len(idx_in)),
                "gt_types_in_interval": {EDGE_TYPES[j]: int(c) for j, c in
                                         enumerate(np.bincount(et[idx_in], minlength=len(EDGE_TYPES))) if c},
                "nodes": sorted(rows, key=lambda r: (-r["events_in_interval"], -r["events_today"])),
                "chain_steps": len(chain),
                "chain": [{"src": lab(k[0]), "type": EDGE_TYPES[k[1]], "dst": lab(k[2]), "n": n,
                           "first": ts(f), "last": ts(l)} for k, n, f, l in chain[:a.chain]]}

        pf = Path(a.pass1) / f"pass1_{day}.pt"
        if not pf.exists():
            continue
        P = torch.load(pf, weights_only=False)
        L = P["losses"].double().numpy()
        if len(L) != len(src):
            raise SystemExit(f"{day}: pass1 has {len(L)} losses for {len(src)} events")
        anom = np.zeros(len(L), bool); tau_e = np.full(len(L), np.nan)
        for w in (Window(**w) for w in P["windows"]):
            tau = window_threshold(P["losses"], w)
            anom[w.start_idx:w.end_idx] = L[w.start_idx:w.end_idx] >= tau
            tau_e[w.start_idx:w.end_idx] = tau
        new_today = (first_day[src] == di) | (first_day[dst] == di)
        unseen = ~(in_train[src] & in_train[dst])
        M = {"loss_quantiles": {str(q): pct(L, q) for q in (50, 90, 99, 99.9)}, "loss_max": float(L.max()),
             "anomalous_events": int(anom.sum()), "anomalous_frac": frac(anom),
             "by_type": {EDGE_TYPES[j]: {"n": int((et == j).sum()), "mean_loss": float(L[et == j].mean()),
                                         "anomalous_frac": frac(anom[et == j])}
                         for j in range(len(EDGE_TYPES)) if (et == j).any()},
             "novelty": {"all_new_endpoint": frac(new_today), "anomalous_new_endpoint": frac(new_today, anom),
                         "all_endpoint_unseen_in_training": frac(unseen),
                         "anomalous_endpoint_unseen_in_training": frac(unseen, anom)}}
        if touch is not None:
            srt = np.sort(L)
            rk = np.searchsorted(srt, L[inside], side="right") / len(L)
            top = np.nonzero(inside)[0][np.argsort(-L[inside])[:15]]
            M["attack"] = {"gt_events_in_interval": int(inside.sum()),
                           "anomalous_frac_gt_in_interval": frac(anom, inside),
                           "anomalous_frac_other_in_interval": frac(anom, in_iv & ~touch),
                           "median_pct_rank_gt_in_interval": float(np.median(rk)) if len(rk) else None,
                           "top_gt_events": [{"src": lab(src[e]), "type": EDGE_TYPES[et[e]], "dst": lab(dst[e]),
                                              "t": ts(t[e]), "loss": float(L[e]), "tau": float(tau_e[e]),
                                              "new_endpoint": bool(new_today[e])} for e in top]}
            gt_w = set(info.get("windows", []))
            al_w = {x["window_file"] for x in alerts if x["day"] == day}
            day_rows = [w for w in win_rows if w["day"] == day]
            avgs = np.array([w["anomalous_loss_avg"] if w["anomalous_loss_avg"] is not None else np.nan
                             for w in day_rows], dtype=float)
            hp = K / f"graph_{day}_history_list"
            queues = torch.load(hp, weights_only=False) if hp.exists() else []
            qscore = []
            for q in queues:
                s = 1.0
                for x in q:
                    s *= float(x["loss"]) + 1.0
                qscore.append(s)
            rows = []
            for w in day_rows:
                f = w["window_file"]
                if f not in gt_w and f not in al_w:
                    continue
                qs = [(len(q), qscore[qi]) for qi, q in enumerate(queues) if any(str(x.get("name", "")).endswith(f) for x in q)]
                rows.append({"window": f, "gt": f in gt_w, "alerted": f in al_w, "n_edges": w["n_edges"],
                             "n_anomalous_edges": w["n_anomalous_edges"], "threshold": w["threshold"],
                             "anomalous_loss_avg": w["anomalous_loss_avg"],
                             "loss_avg_pct_rank": float(np.nanmean(avgs <= (w["anomalous_loss_avg"] or 0.0))),
                             "queues": [{"windows": n, "score": s} for n, s in qs]})
            M["windows"] = rows
            M["n_queues"] = len(queues)
            M["max_queue_score"] = max(qscore) if qscore else None
        R["model"][day] = M

    (out / "dataset.json").write_text(json.dumps(R, indent=1))
    (out / "dataset.md").write_text(render(R))
    print(f"wrote {out / 'dataset.json'} and {out / 'dataset.md'}")


def render(R) -> str:
    L = ["# CADETS E3 as KAIROS sees it (generated by scripts/analyze_dataset.py)", "",
         f"Labelled nodes: {R['n_labelled_nodes']} ({R['labelled_by_kind']})", "",
         "## Days", "", "| day | events | active S/F/N | new S/F/N | top event types | max gap (min) |",
         "|---|---|---|---|---|---|"]
    for d, x in R["days"].items():
        a, n = x["active_by_kind"], x["new_by_kind"]
        tt = sorted(x["event_types"].items(), key=lambda kv: -kv[1])[:3]
        L.append(f"| {d} | {x['events']:,} | {a['subject']}/{a['file']}/{a['netflow']} | "
                 f"{n['subject']}/{n['file']}/{n['netflow']} | "
                 + ", ".join(f"{k.replace('EVENT_', '')} {v / x['events']:.0%}" for k, v in tt)
                 + f" | {x['max_gap_min']:.0f} |")
    for att, x in R["attacks"].items():
        L += ["", f"## {att} ({x['day']}, {x['interval'][0]} to {x['interval'][1]} US/Eastern)", "",
              f"GT nodes {x['gt_nodes']}: active today {x['gt_nodes_active_today']}, new today "
              f"{x['gt_nodes_new_today']}, seen in training {x['gt_nodes_seen_in_training']}. "
              f"Events in the interval {x['events_in_interval']:,}, of which touching GT {x['gt_events_in_interval']:,} "
              f"(GT events all day {x['gt_events_today']:,}). Types: {x['gt_types_in_interval']}.", "",
              "| GT node | kind | events in interval | events today | first seen | in training |", "|---|---|---|---|---|---|"]
        for r in x["nodes"]:
            L.append(f"| `{r['label']}` | {r['kind']} | {r['events_in_interval']} | {r['events_today']} | "
                     f"{r['first_seen']} | {r['seen_in_training']} |")
        L += ["", f"Chain ({x['chain_steps']} steps, consecutive repeats merged; first {len(x['chain'])}):", "", "```"]
        L += [f"{c['first']}  {c['src']}  -{c['type'].replace('EVENT_', '')}->  {c['dst']}  x{c['n']}" for c in x["chain"]]
        L.append("```")
    L += ["", "## Model (seed 0)", "", "| day | p50 | p99 | p99.9 | max | anomalous | anomalous with new endpoint "
          "(all events) | anomalous with endpoint unseen in training (all) |", "|---|---|---|---|---|---|---|---|"]
    for d, m in R["model"].items():
        q, nv = m["loss_quantiles"], m["novelty"]
        f2 = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
        L.append(f"| {d} | {q['50']:.2f} | {q['99']:.2f} | {q['99.9']:.2f} | {m['loss_max']:.1f} | "
                 f"{m['anomalous_frac']:.2%} | {f2(nv['anomalous_new_endpoint'])} ({f2(nv['all_new_endpoint'])}) | "
                 f"{f2(nv['anomalous_endpoint_unseen_in_training'])} ({f2(nv['all_endpoint_unseen_in_training'])}) |")
    for d, m in R["model"].items():
        if "attack" not in m:
            continue
        x = m["attack"]
        L += ["", f"### Attack day {d}", "",
              f"GT events in the interval: {x['gt_events_in_interval']}; anomalous: "
              f"{x['anomalous_frac_gt_in_interval']}; other events in the interval anomalous: "
              f"{x['anomalous_frac_other_in_interval']}; median percentile rank of GT events: "
              f"{x['median_pct_rank_gt_in_interval']}. Queues: {m.get('n_queues')}, max queue score "
              f"{m.get('max_queue_score')} (beta 100).", "",
              "| window | GT | alerted | edges | anomalous edges | threshold | anomalous loss avg (pct rank) | queues (windows: score) |",
              "|---|---|---|---|---|---|---|---|"]
        for w in m["windows"]:
            qs = "; ".join(f"{q['windows']}: {q['score']:.1f}" for q in w["queues"]) or "none"
            n2 = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
            L.append(f"| {w['window'][:30]} | {w['gt']} | {w['alerted']} | {w['n_edges']} | {w['n_anomalous_edges']} | "
                     f"{n2(w['threshold'])} | {n2(w['anomalous_loss_avg'])} ({w['loss_avg_pct_rank']:.0%}) | {qs} |")
        L += ["", "Top GT events by score:", "", "| t | event | loss | tau | new endpoint |", "|---|---|---|---|---|"]
        for e in x["top_gt_events"]:
            L.append(f"| {e['t']} | `{e['src']}` -{e['type'].replace('EVENT_', '')}-> `{e['dst']}` | {e['loss']:.2f} | "
                     f"{e['tau']:.2f} | {e['new_endpoint']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()