"""H2 (faithfulness), H4 (knowledge limits) and H5 (cost): the technique comparison of step 5.

    python scripts/analyze_techniques.py --a "results/compare/A/explanations_*.jsonl" \
        --b "results/compare/B/explanations_*.jsonl" --e7 "results/compare/local_vs_full_*.json" \
        --results results --out results/analysis --tex paper/generated

Settings: A = neighbour edges only (all methods, every window-class alert event and a stratified
sample of prior-class ones); B = neighbour edges plus the last 16 batches of history (a subset).
Per setting, alert class and method:
  NDL@5     normalised drop of the score after deleting the top-5 items, (L_full - L_del5) / (L_full - L_floor),
            where L_floor removes every item (negative: the top items push the score UP, i.e. they explain normality)
  Flip@k    share of events whose score falls below the window threshold after deleting the top-k
  Keep@k    share still at or above the threshold with ONLY the top-k kept
  Min-CF    smallest k on the budget grid whose deletion flips the alert (window class only), and how often one exists
  history@5 share of the top-5 items that are history events (setting B)
95% intervals are percentile bootstraps over events (2,000 resamples, rng 0).
Stability and calibration come from results/stability.json and results/calibration.json (aggregate_e6.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

METHOD_ORDER = ["joint", "joint-cf", "joint-edges", "variational", "graphmask", "gnnexplainer-anomaly",
                "gnnexplainer-v1", "mcts", "occlusion", "integrated_gradients", "gradient", "attention", "recency",
                "random"]
LABEL = {"joint": "PROVEX joint (preserve)", "joint-cf": "PROVEX joint (counterfactual)",
         "joint-edges": "PROVEX joint, edges only", "variational": "PROVEX variational",
         "graphmask": "GraphMask-TC", "gnnexplainer-anomaly": "GNNExplainer, anomaly objective",
         "gnnexplainer-v1": "GNNExplainer, standard objective", "mcts": "MCTS (T-GNNExplainer-style)",
         "occlusion": "Occlusion", "integrated_gradients": "Integrated gradients", "gradient": "Gradient",
         "attention": "Attention", "recency": "Recency", "random": "Random"}
RNG = np.random.default_rng(0)


def load(pattern):
    recs = {}
    for p in sorted(glob.glob(pattern)):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                recs[(r["event"]["day"], r["event"]["event"])] = r   # a re-run shard repeats events: keep the last
    return list(recs.values())


def ci(xs, stat=np.mean):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    if len(xs) == 0:
        return None, None, None
    if len(xs) == 1:
        return float(stat(xs)), None, None
    boot = [stat(xs[RNG.integers(0, len(xs), len(xs))]) for _ in range(2000)]
    return float(stat(xs)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def per_method(recs):
    out = defaultdict(lambda: defaultdict(list))
    for r in recs:
        tau, ch = r["event"]["tau"], r["channels"]
        full, floor = ch["full"], ch["nothing"]
        rnd = next((x for x in r["explanations"] if x["method"] == "random" and x.get("seed", 0) == 0), None)
        rnd_d5 = rnd["metrics"].get("deletion", {}).get("5") if rnd else None
        for ex in r["explanations"]:
            if ex.get("seed", 0) != 0:
                continue
            m, mt = ex["method"], ex["metrics"]
            dele = {int(k): v for k, v in mt.get("deletion", {}).items()}
            ins = {int(k): v for k, v in mt.get("insertion", {}).items()}
            s = out[m]
            s["events"].append(r["event"]["event"])
            if 5 in dele:
                s["dl5"].append(full - dele[5])                     # nats; > 0: deleting the top 5 lowers the score
                s["raises5"].append(float(dele[5] > full + 1e-6))
                if full - floor > 0.1:                              # normalised only where context raised the score clearly
                    s["ndl5"].append((full - dele[5]) / (full - floor))
                if rnd_d5 is not None and m != "random":
                    s["win5_vs_random"].append(float(dele[5] < rnd_d5))
            for k in (1, 5, 10):
                if k in dele:
                    s[f"flip{k}"].append(float(dele[k] < tau))
                if k in ins:
                    s[f"keep{k}"].append(float(ins[k] >= tau))
            if r["alert_class"] == "window" and "min_cf_size" in mt:
                s["mincf_found"].append(float(mt["min_cf_size"] is not None))
                if mt["min_cf_size"] is not None:
                    s["mincf"].append(mt["min_cf_size"])
            if ex.get("mode") in ("preserve", "counterfactual") and ex.get("valid") is not None:
                s["own_valid"].append(float(bool(ex["valid"])))      # the method's own set, checked by exact deletion
                if ex["valid"]:
                    s["own_size"].append(len(ex.get("selected", [])))
            top5 = ex.get("top_items", [])[:5]
            if top5:
                s["hist5"].append(sum("history" in it.get("channels", []) and "edge" not in it.get("channels", [])
                                      for it in top5) / len(top5))
            s["runtime"].append(ex["runtime_s"])
            s["metrics_s"].append(ex.get("metrics_s", 0.0))
            g = mt.get("gt")
            if r.get("gt_endpoint") and g:
                for k in ("p@1", "p@5", "auroc"):
                    v = g.get(k)
                    if v is not None and v == v:
                        s[f"gt_{k}"].append(v)
    return out


def summarise(recs, setting):
    res = {}
    # every metric assumes the recomputed score is the detector's; drop events where it is not
    bad = [r for r in recs if abs(r.get("full_minus_detector", r["channels"]["full"] - r["event"]["loss"])) > 1e-3]
    recs = [r for r in recs if r not in bad]
    for cls in ("window", "prior"):
        sub = [r for r in recs if r["alert_class"] == cls]
        if not sub:
            continue
        pm = per_method(sub)
        rows = {}
        for m in METHOD_ORDER:
            if m not in pm:
                continue
            s = pm[m]
            row = {"n": len(s["events"])}
            for key in ("ndl5", "flip1", "flip5", "flip10", "keep5", "keep10", "raises5", "hist5", "mincf_found",
                        "own_valid", "win5_vs_random", "gt_p@1", "gt_p@5", "gt_auroc"):
                row[key] = ci(s[key])
            row["dl5"] = ci(s["dl5"], np.median)
            row["mincf_median"] = float(st.median(s["mincf"])) if s["mincf"] else None
            row["own_size_median"] = float(st.median(s["own_size"])) if s["own_size"] else None
            row["runtime_median_s"] = float(st.median(s["runtime"])) if s["runtime"] else None
            row["runtime_max_s"] = float(max(s["runtime"])) if s["runtime"] else None
            rows[m] = row
        floor_above = sum(r["channels"]["nothing"] >= r["event"]["tau"] for r in sub)
        margin = [r["channels"]["full"] - r["event"]["tau"] for r in sub]
        res[cls] = {"n_events": len(sub), "floor_still_alerts": floor_above,
                    "margin_median": float(st.median(margin)),
                    "margin_p10_p90": [float(np.quantile(margin, 0.1)), float(np.quantile(margin, 0.9))],
                    "items_median": float(st.median([r["n_items"] for r in sub])),
                    "history_items_median": float(st.median([r["n_history_items"] for r in sub])),
                    "forward_ms_median": float(st.median([r.get("forward_ms", 0.0) for r in sub])),
                    "methods": rows}
    wall = [sum(ex["runtime_s"] + ex.get("metrics_s", 0.0) for ex in r["explanations"]) for r in recs]
    return {"setting": setting, "n_events": len(recs), "n_dropped_not_exact": len(bad), "classes": res,
            "wall_per_event_s_median": float(st.median(wall)) if wall else None}


def f(x, nd=2, pct=False):
    if x is None:
        return "n/a"
    return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"


def fci(t, nd=2, pct=False):
    v, lo, hi = t if t else (None, None, None)
    if v is None:
        return "n/a"
    if lo is None:
        return f(v, nd, pct)
    return f"{f(v, nd, pct)} [{f(lo, nd, pct)}, {f(hi, nd, pct)}]"


def md_table(summary):
    L = []
    for cls, c in summary["classes"].items():
        L += [f"### Setting {summary['setting']}, {cls}-class events (n = {c['n_events']}; items median "
              f"{c['items_median']:.0f}, history items median {c['history_items_median']:.0f}; alert margin L - tau "
              f"median {c['margin_median']:.2f} (p10 {c['margin_p10_p90'][0]:.2f}, p90 {c['margin_p10_p90'][1]:.2f}); "
              f"removing every item still alerts for {c['floor_still_alerts']}/{c['n_events']})", "",
              "| method | dL@5 (nats, median) | NDL@5 | Flip@1 | Flip@5 | Flip@10 | Keep@5 | top-5 raises score | "
              "beats random (dL@5) | own set valid | own set size | Min-CF found | Min-CF median | history share of top-5 | "
              "GT P@5 | time median (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, r in c["methods"].items():
            L.append(f"| {LABEL.get(m, m)} | {fci(r['dl5'])} | {fci(r['ndl5'])} | {fci(r['flip1'], pct=True)} | "
                     f"{fci(r['flip5'], pct=True)} | {fci(r['flip10'], pct=True)} | {fci(r['keep5'], pct=True)} | "
                     f"{fci(r['raises5'], pct=True)} | {fci(r['win5_vs_random'], pct=True)} | {fci(r['own_valid'], pct=True)} | "
                     f"{f(r['own_size_median'], 0)} | {fci(r['mincf_found'], pct=True)} | {f(r['mincf_median'], 0)} | "
                     f"{fci(r['hist5'], pct=True)} | {fci(r['gt_p@5'])} | {f(r['runtime_median_s'], 1)} |")
        L.append("")
    return L


def tex_table(summary, cls, caption, label):
    c = summary["classes"].get(cls)
    if not c:
        return ""
    rows = []
    for m, r in c["methods"].items():
        rows.append(" & ".join([LABEL.get(m, m), f(r["dl5"][0], 2),
                                f(r["flip1"][0], pct=True), f(r["flip5"][0], pct=True), f(r["keep5"][0], pct=True),
                                f(r["win5_vs_random"][0], pct=True) if m != "random" else "--",
                                f(r["mincf_median"], 0) if cls == "window" else "--",
                                f(r["runtime_median_s"], 1)]).replace("%", "\\%") + " \\\\")
    return ("\\begin{table*}[t]\n\\centering\\small\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\setlength{\\tabcolsep}{5pt}\n"
            "\\fittable{\\begin{tabular}{lrrrrrrr}\n\\toprule\nMethod & $\\Delta L$@5 & Flip@1 & Flip@5 & Keep@5 & $>$Random & Min-CF & Time (s) \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", default="")
    ap.add_argument("--e7", default=""); ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True); ap.add_argument("--tex", default="")
    a = ap.parse_args()
    out = {"A": summarise(load(a.a), "A")}
    recs_b = load(a.b) if a.b else []
    if recs_b:
        out["B"] = summarise(recs_b, "B")
    tau_of = {(r["event"]["day"], r["event"]["event"]): r["event"]["tau"] for r in recs_b}
    R = Path(a.results)
    for name in ("stability", "calibration"):
        p = R / f"{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else None
    if a.e7:
        rows = []
        for p in sorted(glob.glob(a.e7)):                   # local_vs_full_<day>.json; event ids are per day
            day = Path(p).stem.replace("local_vs_full_", "")
            for row in json.loads(Path(p).read_text())["rows"]:
                row["key"] = (day, row["event"])
                rows.append(row)
        if rows:
            from scipy.stats import spearmanr
            d_true = np.array([r["L_full_replay"] - r["L_orig"] for r in rows])
            d_f = np.array([r["L_functional"] - r["L_orig"] for r in rows])
            rel = np.abs(d_f - d_true) / np.maximum(np.abs(d_true), 1e-3)
            out["e7"] = {"n": len(rows), "median_rel_err": float(np.median(rel)), "p90_rel_err": float(np.quantile(rel, 0.9)),
                         "spearman": float(spearmanr(d_f, d_true).statistic),
                         "flip_agreement": float(np.mean([(r["L_functional"] < tau_of[r["key"]]) ==
                                                          (r["L_full_replay"] < tau_of[r["key"]])
                                                          for r in rows if r["key"] in tau_of])) if tau_of else None,
                         "exact_scope_share": float(np.mean([bool(r.get("exact_scope")) for r in rows]))}
            # by deletion kind and size: does the replay's alert decision agree with full re-execution, and
            # how large is the score drop under each (the relative error above is inflated by tiny drops)
            by = {}
            for kind in ("top", "random"):
                for k in sorted({r["k"] for r in rows}):
                    rr = [r for r in rows if r["kind"] == kind and r["k"] == k and r["key"] in tau_of]
                    if not rr:
                        continue
                    ft = [r["L_full_replay"] < tau_of[r["key"]] for r in rr]
                    ff = [r["L_functional"] < tau_of[r["key"]] for r in rr]
                    by[f"{kind}{k}"] = {"n": len(rr), "agree": float(np.mean([a == b for a, b in zip(ft, ff)])),
                                        "flip_full": float(np.mean(ft)), "flip_replay": float(np.mean(ff)),
                                        "drop_full_median": float(np.median([r["L_orig"] - r["L_full_replay"] for r in rr])),
                                        "drop_replay_median": float(np.median([r["L_orig"] - r["L_functional"] for r in rr])),
                                        "abs_diff_median": float(np.median([abs(r["L_functional"] - r["L_full_replay"]) for r in rr]))}
            out["e7"]["by"] = by
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    (O / "techniques.json").write_text(json.dumps(out, indent=1))
    L = ["# H2/H4/H5: technique comparison (generated by scripts/analyze_techniques.py)", "",
         "NDL@5 = share of the context contribution (L_full - L_floor) removed by deleting the top-5 items. "
         "Brackets: 95% bootstrap intervals over events.", ""]
    for key in ("A", "B"):
        if key in out:
            L += [f"## Setting {key} (wall time per event, all methods, median {f(out[key]['wall_per_event_s_median'], 0)} s; "
                  f"{out[key]['n_dropped_not_exact']} events dropped because the recomputed score was not the detector's)", ""]
            L += md_table(out[key])
    if out.get("stability"):
        L += ["## Stability across 5 explainer seeds (E6)", "", "| method | top-5 Jaccard | Kendall tau_b |", "|---|---|---|"]
        for m, v in out["stability"].items():
            L.append(f"| {LABEL.get(m, m)} | {f(v.get('topk_jaccard'))} | {f(v.get('kendall_tau_b'))} |")
        L.append("")
    if out.get("calibration"):
        c = out["calibration"]
        L += [f"Variational calibration against exact enumeration: ECE {f(c.get('ece'), 3)}, max error {f(c.get('max_abs_err'), 3)}.", ""]
    if out.get("e7"):
        e = out["e7"]
        L += [f"## Local vs full replay (E7): n = {e['n']}, median relative error {f(e['median_rel_err'], 3)}, "
              f"p90 {f(e['p90_rel_err'], 3)}, Spearman {f(e['spearman'], 3)}, alert decision agrees "
              f"{f(e['flip_agreement'], pct=True)}, exact scope {f(e['exact_scope_share'], pct=True)}", ""]
    (O / "techniques.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if a.tex:
        T = Path(a.tex); T.mkdir(parents=True, exist_ok=True)
        (T / "tab_faith_window.tex").write_text(tex_table(
            out["A"], "window", "Faithfulness on context-dependent (window-class) alert events, neighbour-edge "
            "setting. $\\Delta L$@5: median drop of the score (nats) after deleting the top-5 items; Flip@$k$: share "
            "of events that drop below the threshold; Keep@5: share still alerting with only the top-5 kept; "
            "$>$Random: share of events where deleting the top 5 lowers the score more than deleting a random 5; "
            "Min-CF: median smallest flipping $k$ on the budget grid.", "tab:faith-window"))
        (T / "tab_faith_prior.tex").write_text(tex_table(
            out["A"], "prior", "The same metrics on rare-action (prior-class) alert events, for which removing "
            "every item still leaves the score above the threshold.",
            "tab:faith-prior"))


if __name__ == "__main__":
    main()