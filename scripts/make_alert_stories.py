"""Alert-level explanation stories for SOC triage (Tier 1 / 2 / 3), built from KAIROS's own scores.

    python scripts/make_alert_stories.py --graphs artifact/graph_embeddings \
        --names artifact/explanations/node_mapping.json --kairos artifact/kairos_upstream/seed_0 \
        --pass1 results/pass1 --records "results/stage1/expl/explanations_*.jsonl" \
        [--explanations "results/expl/seed_0/explanations_*.jsonl"] --out results/stories

An alert is one KAIROS queue: a chain of 15-minute windows. For each alert, the story has three tiers.

Tier 1 (triage):
- The flagged events (score at or above the window threshold) are grouped into actions
  (subject, event type, object) and ranked by score.
- Each action carries facts an analyst can act on:
  - is the action anomalous for this model from its type alone (a rare action)?
  - is an entity new today, or never seen during training?
  - is a network peer outside the monitored network?
- At most --k actions per window, plus --k-alert actions for the whole alert.
- Two orders are kept:
  - by score: KAIROS's own ranking;
  - by triage: a fixed, analyst-style order (new external IP addresses first, then newly
    created programs executing or new processes, then score). The triage order is fixed, not
    learned. It was chosen after reading the 04-06 score-order story, where benign DNS and php-fpm
    actions ranked above the attacker's C2, and was not changed for 04-12 (the cleaner test).
- Network novelty is judged per IP address. KAIROS keys sockets by IP:port, so every
  ephemeral port would otherwise look new.

Tier 2 (narrative, on demand): for context-dependent events, the past events that make them
anomalous, in time order. Taken from explanation records when they are given.

Tier 3 (internals): the share of rare-action events, how much of each window the top actions
cover, and knowledge limits.

Nothing here uses ground truth. Evaluation against DARPA's indicators is a separate script.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provex.data import EDGE_TYPES, load_day, load_node_names  # noqa: E402
from provex.replay.alerts import window_threshold  # noqa: E402
from provex.replay.kairos_replay import Window  # noqa: E402

TZ = pytz.timezone("US/Eastern")
TRAIN = ("4_2", "4_3", "4_4")


def hhmmss(ns) -> str:
    return datetime.fromtimestamp(int(ns) / 1e9, TZ).strftime("%H:%M:%S")


def day_order(day: str) -> int:
    return int(day.split("_")[1])


def short_type(t: str) -> str:
    return t.replace("EVENT_", "")


def load_records(pattern):
    recs = {}
    for f in sorted(glob.glob(pattern or "")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                recs[(r["event"]["day"], r["event"]["event"])] = r
    return recs


def best_explanation(rec):
    """The explanation an analyst would see for a context-dependent event: a valid counterfactual if one
    exists, else a valid preserve set, else the variational ranking."""
    ex = {(x["method"], x.get("seed", 0)): x for x in rec.get("explanations", [])}
    for key in (("joint-cf", 0), ("joint", 0), ("variational", 0)):
        x = ex.get(key)
        if x is not None and (x["valid"] or key[0] == "variational") and x["top_items"]:
            return x
    return None


def main():
    ap = argparse.ArgumentParser()
    for k in ("--graphs", "--names", "--kairos", "--pass1", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--records", required=True, help="stage-1 records (channels per alert event)")
    ap.add_argument("--explanations", default="", help="explanation records for Tier 2 (optional)")
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--k-alert", type=int, default=10)
    ap.add_argument("--internal", default="128.55.12.,127.,0.0.0.0,::1,localhost",
                    help="address prefixes of the monitored network (comma separated)")
    a = ap.parse_args()
    internal = tuple(x for x in a.internal.split(",") if x)
    names = load_node_names(a.names)
    K = Path(a.kairos)
    alerts = json.loads((K / "alerts.json").read_text())
    alert_days = sorted({x["day"] for x in alerts}, key=day_order)
    days = [f"4_{d}" for d in range(2, day_order(alert_days[-1]) + 1)]

    # novelty: first day each node (and each IP address) appears, and whether it appeared during training
    first_day, seen_train, data = {}, set(), {}
    ip_first_day = {}
    for d in days:
        p = Path(a.graphs) / f"graph_{d}.TemporalData.simple"
        if not p.exists():
            continue
        g = load_day(p)
        src, dst = g.src.numpy(), g.dst.numpy()
        nodes = np.unique(np.concatenate([src, dst]))
        for n in nodes.tolist():
            if n not in first_day:
                first_day[n] = d
                ip = ip_of(names.get(n, ""))
                if ip is not None:
                    ip_first_day.setdefault(ip, d)
        if d in TRAIN:
            seen_train.update(nodes.tolist())
        if d in alert_days:
            data[d] = (src, dst, g.t.numpy(), g.msg[:, 16:23].argmax(1).numpy())

    # type-only score per event type (constant per type), measured in stage 1
    stage1 = load_records(a.records)
    prior_by_type = defaultdict(list)
    for r in stage1.values():
        prior_by_type[r["event"]["type"]].append(r["channels"]["prior_only"])
    prior = {t: st.median(v) for t, v in prior_by_type.items()}
    expl = load_records(a.explanations) if a.explanations else {}

    def label(n):
        return names.get(int(n), f"node {int(n)}")

    def kind(n):
        return label(n).partition(": ")[0]

    def external(n):
        kind, _, name = label(n).partition(": ")
        if kind != "netflow":
            return False
        return not name.startswith(internal)

    stories = []
    queues = defaultdict(list)
    for al in alerts:
        queues[(al["day"], al["queue_id"])].append(al)
    for (day, qid), als in sorted(queues.items(), key=lambda kv: (day_order(kv[0][0]), kv[0][1])):
        src, dst, t, et = data[day]
        P = torch.load(Path(a.pass1) / f"pass1_{day}.pt", weights_only=False)
        L = P["losses"].double().numpy()
        wins = [Window(**w) for w in P["windows"]]
        windows, alert_actions = [], {}
        for al in sorted(als, key=lambda x: x["start_ns"]):
            w = next(w for w in wins if abs(w.start_t - al["start_ns"]) <= 1000 and abs(w.end_t - al["end_ns"]) <= 1000)
            tau = window_threshold(P["losses"], w)
            idx = np.nonzero(L[w.start_idx:w.end_idx] >= tau)[0] + w.start_idx
            acts = {}
            n_rare = 0
            for e in idx.tolist():
                ty = EDGE_TYPES[int(et[e])]
                rare = prior.get(ty) is not None and prior[ty] >= tau
                n_rare += rare
                key = (int(src[e]), ty, int(dst[e]))
                x = acts.get(key)
                if x is None:
                    acts[key] = x = {"src": key[0], "type": ty, "dst": key[2], "count": 0, "max_score": -1.0,
                                     "first_ns": int(t[e]), "last_ns": int(t[e]), "example_event": e, "rare": rare}
                x["count"] += 1
                x["first_ns"] = min(x["first_ns"], int(t[e])); x["last_ns"] = max(x["last_ns"], int(t[e]))
                if L[e] > x["max_score"]:
                    x["max_score"], x["example_event"] = float(L[e]), e
            ranked = sorted(acts.values(), key=lambda x: (-x["max_score"], -x["count"]))
            for x in ranked:
                for side in ("src", "dst"):
                    n = x[side]
                    x[f"{side}_label"] = label(n)
                    x[f"{side}_new_today"] = first_day.get(n) == day
                    x[f"{side}_unseen_in_training"] = n not in seen_train
                x["external_peer"] = next((x[f"{s}_label"] for s in ("src", "dst") if external(x[s])), None)
                peer_ip = ip_of(x["external_peer"]) if x["external_peer"] else None
                x["external_ip_new_today"] = peer_ip is not None and ip_first_day.get(peer_ip) == day
                x["new_program_runs"] = (x["type"] == "EVENT_EXECUTE" and (x["src_new_today"] or x["dst_new_today"])) \
                    or (kind(x["src"]) == "subject" and x["src_new_today"]) or (kind(x["dst"]) == "subject" and x["dst_new_today"])
                x["tau"] = tau
            # KAIROS keeps one socket node per local and remote endpoint but labels it by the remote
            # address only, so several nodes share a label. Merge actions an analyst cannot tell apart.
            merged = {}
            for x in ranked:
                k3 = (x["src_label"], x["type"], x["dst_label"])
                y = merged.get(k3)
                if y is None:
                    merged[k3] = dict(x, n_nodes_merged=1)
                    continue
                y["n_nodes_merged"] += 1
                y["count"] += x["count"]
                y["first_ns"] = min(y["first_ns"], x["first_ns"]); y["last_ns"] = max(y["last_ns"], x["last_ns"])
                if x["max_score"] > y["max_score"]:
                    y["max_score"], y["example_event"] = x["max_score"], x["example_event"]
                for flag in ("rare", "src_new_today", "dst_new_today", "src_unseen_in_training",
                             "dst_unseen_in_training", "external_ip_new_today", "new_program_runs"):
                    y[flag] = y[flag] or x[flag]
            ranked = sorted(merged.values(), key=lambda x: (-x["max_score"], -x["count"]))
            for x in ranked:
                x["first"], x["last"] = hhmmss(x["first_ns"]), hhmmss(x["last_ns"])
            ranked_triage = sorted(ranked, key=triage_key)
            top = ranked_triage[:a.k]
            top_by_score = ranked[:a.k]
            covered = sum(x["count"] for x in top)
            for x in ranked:
                k2 = (x["src_label"], x["type"], x["dst_label"])
                if k2 not in alert_actions or x["max_score"] > alert_actions[k2]["max_score"]:
                    alert_actions[k2] = dict(x, window=al["window_file"])
            tier2 = []
            for e in idx.tolist():
                r = expl.get((day, e))
                if r is None or r["alert_class"] != "window":
                    continue
                bx = best_explanation(r)
                if bx is None:
                    continue
                items = sorted(bx["top_items"][:5], key=lambda it: it["t"])
                tier2.append({"event": e, "method": bx["method"], "valid": bx["valid"], "mode": bx["mode"],
                              "narrative": [{"t": hhmmss(it["t"]), "src": it["src_label"], "type": it["type"],
                                             "dst": it["dst_label"], "score": it["score"]} for it in items]})
            windows.append({"window": al["window_file"], "start": hhmmss(w.start_t), "end": hhmmss(w.end_t),
                            "tau": tau, "n_flagged": int(len(idx)), "n_actions": len(ranked),
                            "n_rare_events": int(n_rare), "n_context_events": int(len(idx) - n_rare),
                            "tier1": top, "tier1_by_score": top_by_score, "tier2": tier2,
                            "tier3": {"rare_share": n_rare / len(idx) if len(idx) else None,
                                      "top_k_event_coverage": covered / len(idx) if len(idx) else None}})
        n_flagged = sum(w["n_flagged"] for w in windows)
        alert_top = sorted(alert_actions.values(), key=triage_key)[:a.k_alert]
        alert_top_by_score = sorted(alert_actions.values(), key=lambda x: (-x["max_score"], -x["count"]))[:a.k_alert]
        n_items = len(alert_top) + sum(len(w["tier1"]) for w in windows)
        stories.append({"day": day, "queue_id": qid, "queue_score": als[0]["queue_score"], "n_windows": len(windows),
                        "n_flagged_events": n_flagged, "n_story_items": n_items,
                        "evidence_reduction": n_flagged / n_items if n_items else None,
                        "tier1_alert": alert_top, "tier1_alert_by_score": alert_top_by_score, "windows": windows,
                        "limits": ["Rare actions are anomalous for this model from the event type alone; their "
                                   "context only raises or lowers the score.",
                                   "Context-dependent events are explained from recent history (Tier 2) on demand.",
                                   "KAIROS alerts only when several linked windows are anomalous; short attacks "
                                   "can pass without an alert."]})
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    (O / "stories.json").write_text(json.dumps({"type_only_score": prior, "alerts": stories}, indent=1))
    (O / "stories.md").write_text(render(stories))
    print(render(stories)[:6000])


def ip_of(label):
    """IP address of a netflow label ('netflow: 1.2.3.4:80' -> '1.2.3.4'), else None."""
    if not label or not label.startswith("netflow: "):
        return None
    return label[len("netflow: "):].rsplit(":", 1)[0]


def triage_key(x):
    """Analyst-style order fixed in advance: new external IP addresses, then new programs running, then score."""
    return (not x.get("external_ip_new_today", False), not x.get("new_program_runs", False), -x["max_score"], -x["count"])


def flags(x):
    f = []
    if x["rare"]:
        f.append("rare action")
    for side in ("src", "dst"):
        if x[f"{side}_new_today"]:
            f.append(f"{x[f'{side}_label']} is new today")
        elif x[f"{side}_unseen_in_training"]:
            f.append(f"{x[f'{side}_label']} was never seen in training")
    if x["external_peer"]:
        f.append("external IP new today" if x.get("external_ip_new_today") else "external peer")
    if x.get("new_program_runs"):
        f.append("new program running")
    return "; ".join(f)


def line(x):
    return (f"{x['first']}–{x['last']}  `{x['src_label']}` {short_type(x['type'])} `{x['dst_label']}` "
            f"x{x['count']}  (score {x['max_score']:.2f} vs threshold {x['tau']:.2f}; {flags(x) or 'no flags'})")


def render(stories) -> str:
    L = ["# KAIROS alerts explained (generated by scripts/make_alert_stories.py)", ""]
    for s in stories:
        L += [f"## Alert {s['day']} queue {s['queue_id']}: {s['n_windows']} windows, {s['n_flagged_events']:,} flagged events "
              f"-> {s['n_story_items']} story items (x{s['evidence_reduction']:.0f} reduction)", "",
              f"### Tier 1: top actions across the alert", ""]
        L += [f"- {line(x)}" for x in s["tier1_alert"]]
        for w in s["windows"]:
            r = w["tier3"]["rare_share"]; cov = w["tier3"]["top_k_event_coverage"]
            L += ["", f"### Window {w['start']}–{w['end']}: {w['n_flagged']:,} flagged events in {w['n_actions']} actions; "
                  f"{(r or 0):.0%} rare actions; the top {len(w['tier1'])} cover {(cov or 0):.0%} of flagged events", ""]
            L += [f"- {line(x)}" for x in w["tier1"]]
            if w["n_context_events"]:
                L.append(f"- {w['n_context_events']} context-dependent events: Tier 2 narrative "
                         + ("below" if w["tier2"] else "on demand"))
            for t2 in w["tier2"][:3]:
                L.append(f"  - event {t2['event']} ({t2['method']}, {'valid' if t2['valid'] else 'not valid'}):")
                L += [f"    - {n['t']} `{n['src']}` {short_type(n['type'] or '?')} `{n['dst']}`" for n in t2["narrative"]]
        L += ["", "Limits: " + " ".join(s["limits"]), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()