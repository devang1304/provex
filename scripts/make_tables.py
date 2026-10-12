"""Generate every number and table of paper/provex_v2.tex from results/ (no hand-typed values).

    python scripts/make_tables.py --results results --out paper/generated

Inputs (missing ones degrade to "n/a"):
  results/kairos/seed_<n>/detection_metrics.json        KAIROS detection (vendored evaluation)
  results/expl/seed_<n>/explanations_<day>.jsonl        provex.pipeline explain records
  results/parity/parity_<day>.json                      provex.pipeline parity
  results/calibration/*.json, results/local_vs_full/*.json, results/meta.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

METHOD_ORDER = ["joint", "joint-cf", "joint-edges", "variational", "graphmask", "gnnexplainer-anomaly",
                "gnnexplainer-v1", "mcts", "occlusion", "integrated_gradients", "gradient", "attention", "recency",
                "random"]
METHOD_LABEL = {"joint": "PROVEX joint (preserve)", "joint-cf": "PROVEX joint (counterfactual)",
                "joint-edges": "PROVEX joint, edges only", "variational": "PROVEX variational",
                "graphmask": "GraphMask-TC", "gnnexplainer-anomaly": "GNNExplainer + our objective",
                "gnnexplainer-v1": "GNNExplainer (v1 objective)", "mcts": "MCTS (T-GNNExplainer-style)",
                "occlusion": "Occlusion (LOO)",
                "integrated_gradients": "Integrated gradients", "gradient": "Gradient", "attention": "Attention",
                "recency": "Recency", "random": "Random"}
DAY_LABEL = {"4_6": "04-06", "4_11": "04-11", "4_12": "04-12", "4_13": "04-13"}


def fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{100 * x:.{max(0, nd - 1)}f}\\%" if pct else f"{x:.{nd}f}"


def mean_sd(xs, nd=2, pct=False):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return fmt(xs[0], nd, pct)
    m, s = st.mean(xs), st.pstdev(xs)
    return f"{fmt(m, nd, pct)}$\\pm${fmt(s, nd, pct).rstrip('%').rstrip(chr(92))}" + ("\\%" if pct else "")


def tex_escape(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"),
                 ("$", "\\$"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def table(caption, label, header, rows, align):
    out = ["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\fittable{{\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/generated")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    meta = json.loads((R / "meta.json").read_text()) if (R / "meta.json").exists() else {}
    nums = {}

    # ---------------------------------------------------------------- detection (E1)
    det = {}
    for p in sorted(glob.glob(str(R / "kairos" / "seed_*" / "detection_metrics.json"))):
        det[Path(p).parent.name] = json.loads(Path(p).read_text())
    nums["NumSeeds"] = str(len(det)) if det else "n/a"
    hit = sorted(a for a in {k for d in det.values() for k in d.get("per_attack", {})}
                 if all(d["per_attack"].get(a, {}).get("detected") for d in det.values()))
    nums["NumAttacksDetected"] = str(len(hit)) if det else "n/a"
    nums["AttacksDetected"] = (" and ".join("04-" + a[-2:] for a in hit) or "none") if det else "n/a"
    rows = []
    attacks = sorted({k for d in det.values() for k in d.get("per_attack", {})})
    for att in attacks:
        rs = [d["per_attack"][att] for d in det.values() if att in d.get("per_attack", {})]
        if not rs or not rs[0].get("evaluated", False):
            rows.append([tex_escape(att), "n/a", "n/a", "n/a"]); continue
        rows.append([tex_escape(att), f"{sum(r['detected'] for r in rs)}/{len(rs)}",
                     mean_sd([r["n_detected_windows"] / max(1, r["n_gt_windows"]) for r in rs], 2, pct=True),
                     str(rs[0]["n_gt_windows"])])
    ov = [d.get("overall", {}) for d in det.values()]
    up = [d.get("upstream_protocol") or {} for d in det.values()]
    rows.append(["All test windows (P / R / FPR)",
                 mean_sd([o.get("precision") for o in ov], 2) + " / " + mean_sd([o.get("recall") for o in ov], 2),
                 mean_sd([o.get("fpr") for o in ov], 3), str(ov[0].get("n_windows", "n/a")) if ov else "n/a"])
    rows.append(["Upstream protocol (P / R)", mean_sd([u.get("precision") for u in up], 2) + " / " +
                 mean_sd([u.get("recall") for u in up], 2), "", ""])
    (O / "tab_detection.tex").write_text(table(
        f"KAIROS detection on CADETS E3 (window level), mean$\\pm$sd over {nums['NumSeeds']} seeds.",
        "tab:detection", ["Attack / summary", "Detected (seeds)", "GT windows flagged", "\\#GT windows"],
        rows, "lccc"))

    # ---------------------------------------------------------------- explanations (E3-E6, E8)
    recs = []
    for p in sorted(glob.glob(str(R / "expl" / "seed_*" / "explanations_*.jsonl"))):
        for r in load_jsonl(p):
            r["_model_seed"] = Path(p).parent.name
            recs.append(r)
    # explain appends to its .jsonl, so a re-run shard repeats events: keep the last record per event
    uniq = {}
    for r in recs:
        uniq[(r["_model_seed"], r["event"]["day"], r["event"]["event"])] = r
    recs = list(uniq.values())
    # taxonomy
    tax = defaultdict(lambda: defaultdict(int))
    for r in recs:
        if r["explanations"] and r["explanations"][0]["seed"] != 0:
            continue
        d = r["event"]["day"]; grp = "mal" if r.get("gt_endpoint") else "other"
        tax[(d, grp)][r["alert_class"]] += 1
        tax[("all", "all")][r["alert_class"]] += 1
    tot = sum(tax[("all", "all")].values())
    nums["NumExplainedAlerts"] = str(tot)
    nums["FracPriorAlerts"] = fmt(tax[("all", "all")]["prior"] / tot, 2, pct=True) if tot else "n/a"
    nums["FracWindowAlerts"] = fmt(tax[("all", "all")]["window"] / tot, 2, pct=True) if tot else "n/a"
    nums["FracLongMemoryAlerts"] = fmt(tax[("all", "all")]["long_memory"] / tot, 2, pct=True) if tot else "n/a"
    # removing the whole window does not flip prior / long-memory alerts; did a smaller removal?
    cf = [ex for r in recs if r["alert_class"] != "window"
          for ex in r["explanations"] if ex["method"] == "joint-cf" and ex["seed"] == 0]
    nums["CFFoundNonWindow"] = f"{sum(bool(ex['valid']) for ex in cf)}/{len(cf)} such alerts" if cf else "n/a"
    rows = []
    for (d, grp), c in sorted(tax.items()):
        if d == "all":
            continue
        n = sum(c.values())
        rows.append([DAY_LABEL.get(d, d.replace("_", "-")), "GT endpoint" if grp == "mal" else "other", str(n),
                     fmt(c["prior"] / n, 2, True), fmt(c["long_memory"] / n, 2, True), fmt(c["window"] / n, 2, True)])
    (O / "tab_taxonomy.tex").write_text(table(
        "Alert taxonomy: share of explained alert events whose anomaly is driven by the event-type prior, by "
        "memory accumulated before the history window, or by events inside the window (a counterfactual exists).",
        "tab:taxonomy", ["Day", "Events", "$n$", "Prior", "Long memory", "Window"], rows, "llrrrr"))

    # faithfulness on window-explainable alerts
    by_m = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["alert_class"] != "window":
            continue
        tau = r["event"]["tau"]; ch = r["channels"]
        floor = ch["nothing"]; full = ch["full"]
        for ex in r["explanations"]:
            m = ex["method"]; mt = ex["metrics"]
            dele = {int(k): v for k, v in mt.get("deletion", {}).items()}
            ins = {int(k): v for k, v in mt.get("insertion", {}).items()}
            if 5 in dele and abs(full - floor) > 1e-6:
                by_m[m]["ndl5"].append((full - dele[5]) / (full - floor))
            for k in (1, 5, 10):
                if k in dele:
                    by_m[m][f"flip{k}"].append(float(dele[k] < tau))
                if k in ins:
                    by_m[m][f"keep{k}"].append(float(ins[k] >= tau))
            if mt.get("min_cf_size") is not None:
                by_m[m]["mincf"].append(mt["min_cf_size"])
            by_m[m]["runtime"].append(ex["runtime_s"])
            if ex["mode"] in ("preserve", "counterfactual") and ex["method"].startswith("joint"):
                by_m[m]["valid"].append(float(ex["valid"]))
    rows = []
    for m in METHOD_ORDER:
        if m not in by_m:
            continue
        s = by_m[m]
        rows.append([METHOD_LABEL.get(m, m), fmt(st.mean(s["ndl5"]), 2) if s["ndl5"] else "n/a",
                     fmt(st.mean(s["flip1"]), 2, True) if s["flip1"] else "n/a",
                     fmt(st.mean(s["flip5"]), 2, True) if s["flip5"] else "n/a",
                     fmt(st.mean(s["keep5"]), 2, True) if s["keep5"] else "n/a",
                     f"{st.median(s['mincf']):.0f}" if s["mincf"] else "n/a"])
    (O / "tab_faithfulness.tex").write_text(table(
        "Faithfulness on window-explainable alerts. NDL@5: normalised drop of the anomaly score after deleting "
        "the top-5 items; Flip@k: share of alerts that fall below the threshold after deleting the top-$k$; "
        "Keep@5: share still alerting with only the top-5 items kept; Min-CF: median smallest $k$ that flips.",
        "tab:faith", ["Method", "NDL@5", "Flip@1", "Flip@5", "Keep@5", "Min-CF"], rows, "lrrrrr"))

    # ground truth entities
    gt = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if not r.get("gt_endpoint"):
            continue
        for ex in r["explanations"]:
            g = ex["metrics"].get("gt")
            if g:
                for k, v in g.items():
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        gt[ex["method"]][k].append(v)
    rows = [[METHOD_LABEL.get(m, m)] + [fmt(st.mean(gt[m][k]), 2) if gt[m][k] else "n/a"
                                        for k in ("p@1", "p@3", "p@5", "auroc")]
            for m in METHOD_ORDER if m in gt]
    (O / "tab_groundtruth.tex").write_text(table(
        "Agreement with node-level ground truth on alerts whose endpoints are attack entities "
        "(entities other than the explained event's own endpoints).",
        "tab:gt", ["Method", "P@1", "P@3", "P@5", "AUROC"], rows, "lrrrr"))

    # stability + calibration
    stab = json.loads((R / "stability.json").read_text()) if (R / "stability.json").exists() else {}
    cal = json.loads((R / "calibration.json").read_text()) if (R / "calibration.json").exists() else {}
    rows = [[METHOD_LABEL.get(m, m), fmt(v.get("topk_jaccard"), 2), fmt(v.get("kendall_tau_b"), 2)]
            for m, v in stab.items()]
    rows.append(["Calibration (variational, exact enumeration)", "ECE " + fmt(cal.get("ece"), 3),
                 "max err " + fmt(cal.get("max_abs_err"), 3)])
    (O / "tab_stability_calibration.tex").write_text(table(
        "Stability across 5 explainer seeds (top-5 Jaccard, Kendall $\\tau_b$) and calibration of the "
        "variational inclusion probabilities against exact posteriors. GraphMask is deterministic at inference "
        "and the baselines have no seed, so they are omitted.",
        "tab:stab", ["Method", "Top-5 Jaccard", "Kendall $\\tau_b$"], rows, "lrr"))

    # runtime
    rows = [[METHOD_LABEL.get(m, m), fmt(st.median(by_m[m]["runtime"]), 1), fmt(max(by_m[m]["runtime"]), 1)]
            for m in METHOD_ORDER if by_m.get(m, {}).get("runtime")]
    (O / "tab_runtime.tex").write_text(table(
        f"Explanation time per event (seconds) on {meta.get('hardware_cpu', 'the CPU VM')}.",
        "tab:runtime", ["Method", "Median", "Max"], rows, "lrr"))

    # local vs full replay (E7) + parity (E2)
    lvf = json.loads((R / "local_vs_full.json").read_text()) if (R / "local_vs_full.json").exists() else {}
    par = [json.loads(Path(p).read_text()) for p in glob.glob(str(R / "parity" / "parity_*.json"))]
    nums["ParityDaysOK"] = f"{sum(p.get('ok', False) for p in par)}/{len(par)}" if par else "n/a"
    nums["LocalVsFullMedianErr"] = fmt(lvf.get("median_rel_err"), 3)
    nums["LocalVsFullSpearman"] = fmt(lvf.get("spearman"), 3)
    nums["NumExactDeletionsTest"] = str(meta.get("exact_deletions_test", 100))
    gmf = R / "graphmask" / "graphmask_seed0.pt.json"
    gm_final = json.loads(gmf.read_text()).get("final", {}) if gmf.exists() else {}
    nums["GMTrainEvents"] = str(gm_final.get("n_events_trained", "n/a"))
    nums["GMOpenFrac"] = fmt(gm_final.get("open_frac_hard"), 2, pct=True)
    nums["GMValidFrac"] = fmt(gm_final.get("frac_valid_hard"), 2, pct=True)
    nums["GMAllClosedDev"] = fmt(gm_final.get("dev_all_closed_mean"), 2, pct=True)
    nums["GMAllClosedValid"] = fmt(gm_final.get("frac_valid_all_closed"), 2, pct=True)
    nums["TopN"] = str(meta.get("top_n", 25)); nums["SampleN"] = str(meta.get("sample_n", 25))
    nums["ReleaseTag"] = meta.get("release_tag", "v2.0.0")
    nums["ReleaseDOI"] = tex_escape(meta.get("release_doi", "the GitHub release"))
    nums["HardwareCPU"] = tex_escape(meta.get("hardware_cpu", "n/a"))
    nums["HardwareGPU"] = tex_escape(meta.get("hardware_gpu", "n/a"))
    (O / "results_text.tex").write_text(
        f"% generated\nReplay parity with the detector held on {nums['ParityDaysOK']} test days. "
        f"For history gating, the median relative error of the functional replay (influence-closure scope) "
        f"against full re-execution "
        f"was {nums['LocalVsFullMedianErr']} (Spearman {nums['LocalVsFullSpearman']}).\n")
    if not (O / "case_studies.tex").exists():
        (O / "case_studies.tex").write_text("% generated by scripts/make_case_studies.py\n")
    with open(O / "numbers.tex", "w") as f:
        for k, v in sorted(nums.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()