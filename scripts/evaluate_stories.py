"""H1 (decision burden) and H3 (story versus reality): PROVEX alert stories against KAIROS's own outputs.

    python scripts/evaluate_stories.py --stories results/stories/stories.json \
        --summary results/investigation/kairos_summary_graphs.json \
        --steps ground_truth/cadets_e3/darpa/steps.json --iocs ground_truth/cadets_e3/darpa/iocs.json \
        --windows results/analysis/alert_windows.json --out results/analysis

For each KAIROS alert, four views are compared:
- raw alert: every flagged event in every window of the queue;
- KAIROS summary graphs: its Louvain communities over the flagged events;
- PROVEX story in score order;
- PROVEX story in triage order (fixed in advance; see make_alert_stories.py).

Measures (DARPA's report is the ground truth; it is used only here, never to build a story):
- burden: items an analyst reads;
- the rank of the first item that touches attacker infrastructure;
- DARPA attack steps recovered (steps.json, matched within the attack interval plus slack);
- window triage by a simulated analyst who flags a window only when its story shows a new external
  IP or a new program running. Truth: the window holds flagged events that touch DARPA's
  indicators (from alert_windows.json).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pytz

TZ = pytz.timezone("US/Eastern")
DICT_LABEL = re.compile(r"^\{'(\w+)': '(.*)'\}$")


def norm(label: str) -> str:
    """KAIROS window files label nodes "{'file': '/x'}"; PROVEX uses "file: /x"."""
    m = DICT_LABEL.match(label or "")
    return f"{m.group(1)}: {m.group(2)}" if m else (label or "")


def matches(label: str, pat: dict) -> bool:
    kind, _, name = norm(label).partition(": ")
    if pat["kind"] != kind:
        return False
    if kind == "netflow":
        ip, _, port = name.rpartition(":")
        return ip == pat["ip"] and ("port" not in pat or port == pat["port"])
    if kind == "file":
        return name == pat["path"]
    return name == pat.get("name")


def ioc_patterns(iocs):
    pats = {}
    for att, a in iocs["attacks"].items():
        ps = [{"kind": "netflow", "ip": ip} for ip in a.get("ips", {})]
        ps += [{"kind": "file", "path": f} for f in a.get("files", []) if f.startswith("/")]
        ps += [{"kind": "subject", "name": s} for s in a.get("processes", []) if s not in ("nginx", "sshd", "sendmail")]
        pats[a["day"]] = (att, ps)
    return pats


def eastern_ns(s):
    return int(TZ.localize(datetime.strptime(s, "%Y-%m-%d %H:%M")).timestamp() * 1e9)


def item_hits(item, pats):
    return any(matches(item.get(f"{s}_label", ""), p) for s in ("src", "dst") for p in pats)


def first_hit_rank(items, pats):
    return next((i + 1 for i, x in enumerate(items) if item_hits(x, pats)), None)


def steps_covered(items, steps, lo, hi, use_time=True):
    got = []
    for st in steps:
        for x in items:
            in_time = (not use_time) or ("first_ns" not in x) or (x["last_ns"] >= lo and x["first_ns"] <= hi)
            if in_time and item_hits(x, st["patterns"]):
                got.append(st["step"]); break
    return got


def main():
    ap = argparse.ArgumentParser()
    for k in ("--stories", "--summary", "--steps", "--iocs", "--windows", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    stories = json.loads(Path(a.stories).read_text())["alerts"]
    summary = {(q["day"], q["queue_id"]): q for q in json.loads(Path(a.summary).read_text())}
    steps_all = json.loads(Path(a.steps).read_text())
    iocs = json.loads(Path(a.iocs).read_text())
    pats_by_day = ioc_patterns(iocs)
    win_truth = {(w["day"], w["window"]): (w["frac_specific_ioc"] or 0) > 0 for w in json.loads(Path(a.windows).read_text())}
    slack = steps_all.get("slack_minutes", 15) * 60 * 10**9
    rows = []
    for s in stories:
        day = s["day"]
        att, pats = pats_by_day.get(day, (None, []))
        steps = steps_all["attacks"].get(att, []) if att else []
        lo = eastern_ns(iocs["attacks"][att]["start"]) - slack if att else 0
        hi = eastern_ns(iocs["attacks"][att]["end"]) + slack if att else 0
        q = summary.get((day, s["queue_id"]), {})
        # KAIROS summary graphs: communities touching attacker infrastructure
        comm = []
        for c in q.get("communities", []):
            hits = [n for n in c["nodes"] if any(matches(n, p) for p in pats)]
            comm.append({"n_nodes": c["n_nodes"], "n_edges": c["n_edges"], "ioc_nodes": len(hits)})
        comm_steps = []
        all_nodes = [{"src_label": n} for c in q.get("communities", []) for n in c["nodes"]]
        if steps:
            comm_steps = steps_covered(all_nodes, steps, lo, hi, use_time=False)
        ioc_comms = [c for c in comm if c["ioc_nodes"]]
        by_size = sorted(comm, key=lambda c: -c["n_nodes"])
        inspected, read_nodes = 0, 0
        for c in by_size:
            read_nodes += c["n_nodes"]; inspected += 1
            if c["ioc_nodes"]:
                break
        story_items_triage = list(s["tier1_alert"]) + [x for w in s["windows"] for x in w["tier1"]]
        story_items_score = list(s["tier1_alert_by_score"]) + [x for w in s["windows"] for x in w["tier1_by_score"]]
        # simulated analyst window triage from the Tier-1 story only
        tri = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for w in s["windows"]:
            flag = any(x.get("external_ip_new_today") or x.get("new_program_runs") for x in w["tier1"])
            truth = win_truth.get((day, w["window"]), False)
            tri[("t" if flag == truth else "f") + ("p" if flag else "n")] += 1
        raw_truth = sum(win_truth.get((day, w["window"]), False) for w in s["windows"])
        rows.append({
            "day": day, "queue_id": s["queue_id"], "attack": att, "windows": s["n_windows"],
            "windows_with_attacker_events": raw_truth,
            "raw": {"flagged_events": s["n_flagged_events"], "actions": sum(w["n_actions"] for w in s["windows"])},
            "kairos_summary": {"nodes": q.get("graph_nodes"), "edges": q.get("graph_edges"), "communities": q.get("n_communities"),
                               "communities_with_iocs": len(ioc_comms),
                               "smallest_ioc_community_nodes": min((c["n_nodes"] for c in ioc_comms), default=None),
                               "communities_read_by_size_until_first_ioc": inspected if ioc_comms else None,
                               "nodes_read_until_first_ioc": read_nodes if ioc_comms else None,
                               "steps_covered": comm_steps},
            "story_triage": {"items": len(story_items_triage),
                             "first_ioc_rank_alert_top": first_hit_rank(s["tier1_alert"], pats),
                             "ioc_items_in_alert_top": sum(item_hits(x, pats) for x in s["tier1_alert"]),
                             "steps_covered": steps_covered(story_items_triage, steps, lo, hi) if steps else []},
            "story_score": {"items": len(story_items_score),
                            "first_ioc_rank_alert_top": first_hit_rank(s["tier1_alert_by_score"], pats),
                            "ioc_items_in_alert_top": sum(item_hits(x, pats) for x in s["tier1_alert_by_score"]),
                            "steps_covered": steps_covered(story_items_score, steps, lo, hi) if steps else []},
            "steps_total": [st["step"] for st in steps],
            "window_triage": tri})
    out = {"alerts": rows,
           "missed_attacks": [att for att, x in iocs["attacks"].items() if x["day"] not in {r["day"] for r in rows}]}
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    (O / "h1_h3.json").write_text(json.dumps(out, indent=1))
    L = ["# H1 decision burden and H3 story vs reality (generated by scripts/evaluate_stories.py)", ""]
    for r in rows:
        k, st, ss, t = r["kairos_summary"], r["story_triage"], r["story_score"], r["window_triage"]
        n = len(r["steps_total"])
        L += [f"## {r['day']} alert ({r['attack']}): {r['windows']} windows, {r['windows_with_attacker_events']} with attacker events", "",
              "| view | items to read | first attacker item (rank) | DARPA steps recovered |", "|---|---|---|---|",
              f"| raw KAIROS alert | {r['raw']['flagged_events']:,} flagged events ({r['raw']['actions']:,} distinct actions) | n/a | n/a |",
              f"| KAIROS summary graphs | {k['nodes']} nodes / {k['edges']} edges in {k['communities']} communities | "
              f"community {k['communities_read_by_size_until_first_ioc']} by size ({k['nodes_read_until_first_ioc']} nodes read); "
              f"{k['communities_with_iocs']} communities hold attacker nodes, smallest {k['smallest_ioc_community_nodes']} nodes | "
              f"{len(k['steps_covered'])}/{n} ({', '.join(k['steps_covered']) or 'none'}) |",
              f"| PROVEX story, score order | {ss['items']} | {ss['first_ioc_rank_alert_top']} in the alert top 10 ({ss['ioc_items_in_alert_top']} attacker items) | "
              f"{len(ss['steps_covered'])}/{n} ({', '.join(ss['steps_covered']) or 'none'}) |",
              f"| PROVEX story, triage order | {st['items']} | {st['first_ioc_rank_alert_top']} in the alert top 10 ({st['ioc_items_in_alert_top']} attacker items) | "
              f"{len(st['steps_covered'])}/{n} ({', '.join(st['steps_covered']) or 'none'}) |", "",
              f"Simulated analyst window triage from the story: TP {t['tp']}, FP {t['fp']}, FN {t['fn']}, TN {t['tn']} "
              f"(KAIROS flags all {r['windows']} windows).", ""]
    L.append(f"Attacks without a KAIROS alert (nothing to explain): {', '.join(out['missed_attacks']) or 'none'}.")
    (O / "h1_h3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()