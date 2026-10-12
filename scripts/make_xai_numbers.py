"""Numbers and tables for the XAI-in-SOC paper from the analysis outputs (no hand-typed values).

    python scripts/make_xai_numbers.py --analysis results/analysis --out paper/generated

Reads (missing inputs give "n/a"): dataset.json, stage1.json, alert_rule.json, alert_windows.json,
h1_h3.json, novelty_control.json, techniques.json. Writes numbers_xai.tex (macros) and
tab_h1.tex, tab_novelty.tex, tab_alert_rule.tex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DAY = {"4_6": "04-06", "4_11": "04-11", "4_12": "04-12", "4_13": "04-13"}
WORD = {"06": "Six", "11": "Eleven", "12": "Twelve", "13": "Thirteen"}


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def num(x, nd=0):
    if x is None:
        return "n/a"
    if isinstance(x, float) and abs(x) >= 1e6:
        m, e = f"{x:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"   # own math mode, so ranges and text use it as is
    return f"{x:,.{nd}f}"


def pct(x, nd=0):
    return "n/a" if x is None else f"{100 * x:.{nd}f}\\%"


def rng(vals, fmt):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}--{fmt(hi)}"


def table(caption, label, header, rows, align):
    return ("\\begin{table}[t]\n\\centering\\small\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\fittable{\\begin{tabular}{" + align + "}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n"
            + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis"); ap.add_argument("--out", default="paper/generated")
    a = ap.parse_args()
    A, O = Path(a.analysis), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    n = {}

    ds = load(A / "dataset.json") or {}
    kinds = ds.get("labelled_by_kind", {})
    n["NumNodes"] = num(ds.get("n_labelled_nodes"))
    n["NumProcesses"] = num(kinds.get("subject")); n["NumFiles"] = num(kinds.get("file"))
    n["NumNetflows"] = num(kinds.get("netflow"))
    days = ds.get("days", {})
    test = ["4_6", "4_7", "4_8", "4_9", "4_10", "4_11", "4_12"]
    n["ActiveFilesThirteen"] = num(days.get("4_13", {}).get("active_by_kind", {}).get("file"))
    n["ActiveFilesOtherTest"] = rng([days.get(d, {}).get("active_by_kind", {}).get("file") for d in test], lambda v: num(v))
    n["EventsPerTestDay"] = rng([days.get(d, {}).get("events") for d in test + ["4_13"]], lambda v: num(v / 1e6, 1))
    n["GapSixMin"] = num(days.get("4_6", {}).get("max_gap_min")); n["GapElevenMin"] = num(days.get("4_11", {}).get("max_gap_min"))

    s1 = load(A / "stage1.json") or {}
    cl = s1.get("classes", {})
    tot = sum(cl.values()) if cl else None
    n["NumAlertEvents"] = num(tot)
    n["NumPriorEvents"] = num(cl.get("prior")); n["NumWindowEvents"] = num(cl.get("window"))
    n["NumLongMemoryEvents"] = num(cl.get("long_memory", 0) if cl else None)
    n["FracPriorEvents"] = pct(cl.get("prior", 0) / tot) if tot else "n/a"
    bg = s1.get("by_group", {})
    n["NumTopRanked"] = num(sum(bg.get("rank top", {}).values()) if bg else None)
    n["NumTopRankedPrior"] = num(bg.get("rank top", {}).get("prior"))
    for grp, tag in (("endpoint is a labelled attack node", "GTEndpoint"), ("touches a DARPA indicator", "Darpa"),
                     ("labelled attack window", "GTWindow")):
        g = bg.get(grp, {})
        n[f"NumWindow{tag}"] = num(g.get("window", 0) if bg else None)
        n[f"NumPrior{tag}"] = num(g.get("prior", 0) if bg else None)
    bt = s1.get("by_type", {})
    for t, w in (("EVENT_RECVFROM", "Recv"), ("EVENT_WRITE", "Write"), ("EVENT_EXECUTE", "Execute"),
                 ("EVENT_SENDTO", "Send"), ("EVENT_CLOSE", "Close"), ("EVENT_OPEN", "Open")):
        n[f"NumAlert{w}"] = num(sum(bt.get(t, {}).values()) if bt else None)
    ce = s1.get("context_effect", {})
    n["FracContextRaises"] = pct(s1.get("frac_context_raises_score"))
    n["ContextGainMedianPrior"] = num(ce.get("prior", {}).get("vs_prior", {}).get("50"), 2)
    n["ContextGainMedianWindow"] = num(ce.get("window", {}).get("vs_prior", {}).get("50"), 2)
    n["HistoryGainMedianWindow"] = num(ce.get("window", {}).get("vs_no_history", {}).get("50"), 2)
    mz = [abs(v) for c in ce.values() for v in c.get("vs_memory_zero", {}).values()]
    n["MemZeroMaxEffect"] = num(max(mz), 4) if mz else "n/a"
    n["SpearmanFullPrior"] = num(s1.get("spearman_full_vs_prior"), 2)

    ar = load(A / "alert_rule.json") or {}
    seeds = ar.get("seeds", {})
    n["NumRuleSeeds"] = str(len(seeds)) if seeds else "n/a"
    n["ValMaxQueue"] = rng([s.get("validation_max_queue_score") for s in seeds.values()], lambda v: num(v, 1))
    atts = sorted({k for s in seeds.values() for k in s.get("attacks", {})})
    rows = []
    for att in atts:
        per = [s["attacks"][att] for s in seeds.values() if att in s.get("attacks", {})]
        w = WORD.get(att[-2:], att[-2:])
        n[f"BetaNeeded{w}"] = rng([p["best_queue_score"] for p in per], lambda v: num(v, 1 if v < 100 else 0))
        n[f"BestWindowScore{w}"] = rng([p["best_window_score"] for p in per], lambda v: num(v, 2))
        rows.append([DAY.get(per[0]["day"], per[0]["day"]), str(per[0]["gt_windows"]), n[f"BestWindowScore{w}"],
                     n[f"BetaNeeded{w}"]])
    sw = {}
    for s in seeds.values():
        for r in s.get("beta_sweep", []):
            sw.setdefault(r["beta"], []).append(r)
    rows2 = []
    for b in (100, 30, 6, 5):
        rs = sw.get(b) or sw.get(float(b))
        if not rs:
            continue
        caught = sorted({x for r in rs for x in r.get("attacks_detected", [])}, key=lambda d: int(d.split("_")[1]))
        tag = {100: "Hundred", 30: "Thirty", 6: "Six", 5: "Five"}[b]   # TeX macro names take letters only
        n[f"FPWindowsBeta{tag}"] = rng([r["fp_windows"] for r in rs], lambda v: num(v))
        n[f"ValFPWindowsBeta{tag}"] = rng([r["validation_fp_windows"] for r in rs], lambda v: num(v))
        rows2.append([str(b), ", ".join(DAY.get(c, c) for c in caught) or "none",   # caught: day ids
                      n[f"FPWindowsBeta{tag}"], n[f"ValFPWindowsBeta{tag}"]])
    vmw = [s.get("single_window_rule", {}).get("threshold_validation_max_window_score") for s in seeds.values()]
    n["ValMaxWindowScore"] = rng(vmw, lambda v: num(v, 2))
    if rows:
        (O / "tab_alert_rule.tex").write_text(
            table("Why KAIROS misses two attacks. Each attack's best window scores no higher than the busiest "
                  "validation window (" + n["ValMaxWindowScore"] + "), so detection rests on chaining windows "
                  "into queues; $\\beta$ is the queue score each attack needs. Ranges over " + n["NumRuleSeeds"] +
                  " seeds.", "tab:alert-rule", ["Attack", "GT windows", "Best window score", "$\\beta$ needed"],
                  rows, "lrrr")
            + table("The price of a lower alert threshold $\\beta$: attacks caught and false-positive windows on "
                    "the eight test days and on the validation day.", "tab:beta",
                    ["$\\beta$", "Attacks caught", "FP windows (test)", "FP windows (validation)"], rows2, "rlrr"))

    aw = load(A / "alert_windows.json") or []
    riders = [w for w in aw if not w.get("gt_window")]
    gtw = [w for w in aw if w.get("gt_window")]
    n["NumRiderWindows"] = num(len(riders)) if aw else "n/a"
    n["NumRiderWindowsNoIOC"] = num(sum(1 for w in riders if not w.get("frac_specific_ioc"))) if aw else "n/a"
    n["GTWindowIOCShare"] = rng([w.get("frac_specific_ioc") for w in gtw], lambda v: pct(v, 1)) if gtw else "n/a"

    h = load(A / "h1_h3.json") or {}
    rows = []
    tp = fp = fn = tn = 0
    for r in h.get("alerts", []):
        d = DAY.get(r["day"], r["day"]); w = WORD.get(d[-2:], d[-2:])
        k, st_, ss = r["kairos_summary"], r["story_triage"], r["story_score"]
        nt = len(r["steps_total"])
        n[f"FlaggedEvents{w}"] = num(r["raw"]["flagged_events"]); n[f"Actions{w}"] = num(r["raw"]["actions"])
        n[f"AlertWindows{w}"] = num(r["windows"]); n[f"StoryItems{w}"] = num(st_["items"])
        n[f"Reduction{w}"] = num(r["raw"]["flagged_events"] / st_["items"]) if st_["items"] else "n/a"
        n[f"SummaryNodes{w}"] = num(k["nodes"]); n[f"SummaryCommunities{w}"] = num(k["communities"])
        n[f"SummaryNodesRead{w}"] = num(k["nodes_read_until_first_ioc"])
        n[f"SummarySteps{w}"] = f"{len(k['steps_covered'])}/{nt}"
        n[f"StorySteps{w}"] = f"{len(st_['steps_covered'])}/{nt}"
        n[f"StoryScoreSteps{w}"] = f"{len(ss['steps_covered'])}/{nt}"
        t = r["window_triage"]; tp += t["tp"]; fp += t["fp"]; fn += t["fn"]; tn += t["tn"]
        rk = lambda x: "none" if x is None else str(x)
        rows += [[f"{d} raw alert", f"{r['raw']['flagged_events']:,} events ({r['raw']['actions']:,} actions)", "--", "--"],
                 [f"{d} KAIROS summary", f"{k['nodes']} nodes, {k['communities']} communities",
                  f"{k['nodes_read_until_first_ioc']} nodes read", f"{len(k['steps_covered'])}/{nt}"],
                 [f"{d} story, score order", f"{ss['items']} actions", rk(ss["first_ioc_rank_alert_top"]),
                  f"{len(ss['steps_covered'])}/{nt}"],
                 [f"{d} story, triage order", f"{st_['items']} actions", rk(st_["first_ioc_rank_alert_top"]),
                  f"{len(st_['steps_covered'])}/{nt}"]]
    if h:
        n["TriageTP"], n["TriageFP"], n["TriageFN"], n["TriageTN"] = map(str, (tp, fp, fn, tn))
        n["TriageWindowsBefore"] = str(tp + fp + fn + tn); n["TriageWindowsAfter"] = str(tp + fp)
        n["TriagePrecisionBefore"] = pct((tp + fn) / (tp + fp + fn + tn)) if (tp + fp + fn + tn) else "n/a"
        n["TriagePrecisionAfter"] = pct(tp / (tp + fp)) if (tp + fp) else "n/a"
        n["MissedAttacks"] = ", ".join(x for x in h.get("missed_attacks", [])).replace("_", "\\_") or "none"
        (O / "tab_h1.tex").write_text(table(
            "Decision burden and story versus reality for the two KAIROS alerts. Items to read; rank of the first "
            "attacker action among the alert's top 10 (for KAIROS summary graphs: nodes in larger communities read "
            "before the first community holding an attacker entity); DARPA attack steps recovered.",
            "tab:h1", ["View", "Items to read", "First attacker item", "DARPA steps"], rows, "llll"))

    nc = load(A / "novelty_control.json") or {}
    rows = []
    label = {"kairos": "KAIROS alert ($\\beta{=}100$)", "kairos+story": "KAIROS alert + story flag",
             "novelty_all": "Novelty, all events (no detector)", "novelty_flagged": "Novelty, flagged events, every window",
             "queued+story": "Any KAIROS queue + story flag"}
    for k, v in nc.get("rules", {}).items():
        tag = {"kairos": "Kairos", "kairos+story": "KairosStory", "novelty_all": "NoveltyAll",
               "novelty_flagged": "NoveltyFlagged", "queued+story": "QueuedStory"}[k]
        n[f"Ctl{tag}Flagged"] = num(v["test_windows_flagged"]); n[f"Ctl{tag}FP"] = num(v["fp"])
        n[f"Ctl{tag}TP"] = num(v["tp"]); n[f"Ctl{tag}Attacks"] = str(len(v["attacks_detected"]))
        n[f"Ctl{tag}Val"] = num(v["validation_windows_flagged"])
        n[f"Ctl{tag}Precision"] = "n/a" if v["precision"] is None else f"{v['precision']:.2f}"
        rows.append([label[k], num(v["test_windows_flagged"]), num(v["tp"]), num(v["fp"]),
                     "n/a" if v["precision"] is None else f"{v['precision']:.2f}",
                     f"{len(v['attacks_detected'])}/{len(nc.get('attacks', []))}", num(v["validation_windows_flagged"])])
    if nc:
        n["CtlTestWindows"] = num(nc.get("n_test_windows")); n["CtlGTWindows"] = num(nc.get("n_gt_windows"))
        (O / "tab_novelty.tex").write_text(table(
            "Is the story's novelty flag enough on its own? The flag (an external IP address or a program first "
            "seen today) applied with and without KAIROS, over all " + n["CtlTestWindows"] + " test windows.",
            "tab:novelty", ["Rule", "Windows", "TP", "FP", "Precision", "Attacks", "Val. windows"], rows, "lrrrrrr"))

    # H2-H5: technique comparison, as \H<setting><class><method><metric> (letters only), plus E6/E7
    tq = load(A / "techniques.json") or {}
    short = {"joint": "Joint", "joint-cf": "JointCF", "joint-edges": "JointEdges", "variational": "Variational",
             "graphmask": "GraphMask", "gnnexplainer-anomaly": "GnnxAnomaly", "gnnexplainer-v1": "GnnxVone",
             "mcts": "Mcts", "occlusion": "Occlusion", "integrated_gradients": "IntGrad", "gradient": "Gradient",
             "attention": "Attention", "recency": "Recency", "random": "Random"}
    ci_metrics = {"dl5": ("DLFive", lambda v: num(v, 2)), "ndl5": ("NDLFive", lambda v: num(v, 2)),
                  "flip1": ("FlipOne", pct), "flip5": ("FlipFive", pct), "flip10": ("FlipTen", pct),
                  "keep5": ("KeepFive", pct), "raises5": ("RaisesFive", pct), "win5_vs_random": ("WinsFive", pct),
                  "own_valid": ("OwnValid", pct), "mincf_found": ("MinCFFound", pct), "hist5": ("HistFive", pct),
                  "gt_p@5": ("GtPFive", lambda v: num(v, 2))}
    scalars = {"own_size_median": ("OwnSize", 0), "mincf_median": ("MinCF", 0), "runtime_median_s": ("Time", 2)}
    for setting in ("A", "B"):
        sm = tq.get(setting) or {}
        n[f"H{setting}WallMedian"] = num(sm.get("wall_per_event_s_median"))
        n[f"H{setting}Dropped"] = num(sm.get("n_dropped_not_exact"))
        for cls, ctag in (("window", "Win"), ("prior", "Prior")):
            c = sm.get("classes", {}).get(cls, {})
            p = f"H{setting}{ctag}"
            n[p + "N"] = num(c.get("n_events")); n[p + "Margin"] = num(c.get("margin_median"), 2)
            mp_ = c.get("margin_p10_p90") or [None, None]
            n[p + "MarginLo"] = num(mp_[0], 2); n[p + "MarginHi"] = num(mp_[1], 2)
            n[p + "FloorAlerts"] = num(c.get("floor_still_alerts")); n[p + "Items"] = num(c.get("items_median"))
            n[p + "HistItems"] = num(c.get("history_items_median"))
            for m, sh in short.items():
                r = c.get("methods", {}).get(m, {})
                for key, (tag, fmt) in ci_metrics.items():
                    v = r.get(key)
                    n[p + sh + tag] = fmt(v[0]) if v and v[0] is not None else "n/a"
                for key, (tag, nd) in scalars.items():
                    n[p + sh + tag] = num(r.get(key), nd)
    stab = tq.get("stability") or {}
    for m, sh in short.items():
        v = stab.get(m, {})
        n[f"Stab{sh}Jaccard"] = num(v.get("topk_jaccard"), 2); n[f"Stab{sh}Kendall"] = num(v.get("kendall_tau_b"), 2)
    cal = tq.get("calibration") or {}
    n["CalECE"] = num(cal.get("ece"), 3); n["CalMaxErr"] = num(cal.get("max_abs_err"), 3)
    n["CalMeanErr"] = num(cal.get("mean_abs_err"), 3); n["CalPairs"] = num(cal.get("n_pairs"))
    e7 = tq.get("e7") or {}
    n["EsevenN"] = num(e7.get("n")); n["EsevenMedErr"] = num(e7.get("median_rel_err"), 3)
    n["EsevenPNinety"] = num(e7.get("p90_rel_err"), 3); n["EsevenSpearman"] = num(e7.get("spearman"), 2)
    n["EsevenFlipAgree"] = pct(e7.get("flip_agreement")); n["EsevenExact"] = pct(e7.get("exact_scope_share"))
    mpath = load(A / "memory_path_4_12.json") or {}
    md = mpath.get("median", {})
    n["MemPathMemNormU"] = num(md.get("mem_norm_u"), 3); n["MemPathSkipNormU"] = num(md.get("skip_minus_bias_norm_u"), 3)
    n["MemPathConvNormU"] = num(md.get("conv1_norm_u"), 3)

    # every macro and table the paper uses exists even when an input is missing (CI compiles on empty results)
    for w in WORD.values():
        for key in ("BetaNeeded", "BestWindowScore", "FlaggedEvents", "Actions", "AlertWindows", "StoryItems",
                    "Reduction", "SummaryNodes", "SummaryCommunities", "SummaryNodesRead", "SummarySteps",
                    "StorySteps", "StoryScoreSteps"):
            n.setdefault(f"{key}{w}", "n/a")
    for tag in ("Hundred", "Thirty", "Six", "Five"):
        n.setdefault(f"FPWindowsBeta{tag}", "n/a"); n.setdefault(f"ValFPWindowsBeta{tag}", "n/a")
    for tag in ("Kairos", "KairosStory", "NoveltyAll", "NoveltyFlagged", "QueuedStory"):
        for key in ("Flagged", "FP", "TP", "Attacks", "Val", "Precision"):
            n.setdefault(f"Ctl{tag}{key}", "n/a")
    for key in ("TriageTP", "TriageFP", "TriageFN", "TriageTN", "TriageWindowsBefore", "TriageWindowsAfter",
                "TriagePrecisionBefore", "TriagePrecisionAfter", "MissedAttacks", "CtlTestWindows", "CtlGTWindows",
                "ValMaxWindowScore", "NumRuleSeeds", "ValMaxQueue"):
        n.setdefault(key, "n/a")
    for name in ("tab_h1.tex", "tab_novelty.tex", "tab_alert_rule.tex"):
        if not (O / name).exists():
            (O / name).write_text("% no input for this table\n")
    with open(O / "numbers_xai.tex", "w") as fh:
        for k, v in sorted(n.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps(n, indent=1))


if __name__ == "__main__":
    main()