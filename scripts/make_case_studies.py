"""Generate paper/generated/case_studies.tex from the E3 explanation records (no hand-typed values).

    python scripts/make_case_studies.py --results results --gt-dir artifact/kairos_upstream/ground_truth \
        --out paper/generated

One alert event per attack day is chosen by a fixed rule, never by hand:
  1. its window is a ground-truth attack window, one endpoint is a ground-truth entity, and the alert is
     window-explainable (removing the whole history window flips it); highest detector loss first;
  2. otherwise the same without the class condition (the class is then reported);
  3. otherwise any alert in a ground-truth attack window.
Ties go to the earlier event. A day without such an alert says so instead of picking another one.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provex.evaluate.ground_truth import load_gt_nodes  # noqa: E402

DAYS = ["4_6", "4_11", "4_12", "4_13"]
DAY_LABEL = {"4_6": "04-06", "4_11": "04-11", "4_12": "04-12", "4_13": "04-13"}
CLASS_TEXT = {
    "window": "window-explainable: removing every item of the history window drops it below the threshold",
    "prior": "prior-driven: it stays above the threshold with every window item removed, and the event-type "
             "prior alone (no memory, no edges) is anomalous",
    "long_memory": "long-memory-driven: it stays above the threshold with every window item removed, but the "
                   "event-type prior alone is not anomalous, so it rests on memory built up before the window",
}
TIER_TEXT = {1: "the highest-scoring window-explainable alert event in a ground-truth attack window with an "
                "attack entity as an endpoint",
             2: "the highest-scoring alert event in a ground-truth attack window with an attack entity as an "
                "endpoint (no such event was window-explainable)",
             3: "the highest-scoring alert event in a ground-truth attack window (no such event had an attack "
                "entity as an endpoint)"}
_TEX = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%", "#": r"\#", "&": r"\&",
        "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s) -> str:
    """Escape for LaTeX text mode; long paths may break after '/'."""
    return "".join(_TEX.get(c, c) for c in str(s)).replace("/", "/\\allowbreak{}")


def short(label: str, n: int = 44) -> str:
    return label if len(label) <= n else label[: n // 2 - 1] + "..." + label[-(n // 2 - 2):]


def num(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def ago(seconds: float) -> str:
    s = abs(seconds)
    if s < 1:
        return "$<$1\\,s"
    if s < 120:
        return f"{s:.0f}\\,s"
    if s < 7200:
        return f"{s / 60:.0f}\\,min"
    return f"{s / 3600:.1f}\\,h"


def etype(t) -> str:
    return tex(str(t or "?").replace("EVENT_", ""))


def entity(label: str, gid: int, gt: set) -> str:
    s = "\\texttt{" + tex(short(label)) + "}"
    return "\\textbf{" + s + "}" if gid in gt else s


def load_records(results: Path, model_seed: int, day: str) -> list:
    uniq = {}
    for p in sorted((results / "expl" / f"seed_{model_seed}").glob(f"explanations_{day}*.jsonl")):
        if p.stem != f"explanations_{day}" and not p.stem.startswith(f"explanations_{day}_shard"):
            continue                                  # e.g. 4_1 must not pick up 4_11
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                uniq[r["event"]["event"]] = r         # a re-run shard appends: keep the last record
    return list(uniq.values())


def choose(recs: list):
    gtw = [r for r in recs if r["origin"].get("gt_window")]
    tiers = ([r for r in gtw if r.get("gt_endpoint") and r["alert_class"] == "window"],
             [r for r in gtw if r.get("gt_endpoint")],
             gtw)
    for tier, cands in enumerate(tiers, 1):
        if cands:
            return tier, min(cands, key=lambda r: (-r["event"]["loss"], r["event"]["event"]))
    return 0, None


def explanation(r: dict, method: str, seed: int = 0):
    return next((x for x in r["explanations"] if x["method"] == method and x.get("seed", 0) == seed), None)


def deletion(ex: dict, k: int):
    return (ex or {}).get("metrics", {}).get("deletion", {}).get(str(k))


def case(day: str, recs: list, gt: set, k: int) -> str:
    lab = DAY_LABEL.get(day, day)
    out = [f"\\paragraph{{{lab}}}"]
    if not recs:
        return "\n".join(out + [f"No explanation records were produced for {lab}.", ""])
    tier, r = choose(recs)
    if r is None:
        return "\n".join(out + [f"KAIROS raised no alert inside a ground-truth attack window on {lab}, so there "
                                f"is no attack alert to explain ({len(recs)} alert events outside it were "
                                f"explained).", ""])
    e, ch, o = r["event"], r["channels"], r["origin"]
    tau, cls = e["tau"], r["alert_class"]
    out.insert(0, f"% {day}: event {e['event']}, window {o.get('window')}, queue {o.get('queue_id')}, "
                  f"selection tier {tier}")
    out.append(f"We explain {TIER_TEXT[tier]}: {entity(e['src_label'], e['src'], gt)} "
               f"$\\xrightarrow{{\\texttt{{{etype(e['type'])}}}}}$ {entity(e['dst_label'], e['dst'], gt)}, "
               f"with anomaly score $L={num(e['loss'])}$ against the window threshold $\\tau={num(tau)}$ "
               f"(KAIROS expected \\texttt{{{etype(e.get('predicted_type'))}}}). "
               f"The alert is {CLASS_TEXT[cls]}. Removing the neighbour edges gives "
               f"$L={num(ch.get('no_neighbour_edges'))}$, removing the history gives "
               f"$L={num(ch.get('no_history'))}$, removing both gives $L_\\varnothing={num(ch.get('nothing'))}$, "
               f"zeroing the memories of both endpoints gives $L={num(ch.get('memory_zero_uv'))}$, and the type "
               f"prior alone gives $L_{{\\text{{prior}}}}={num(ch.get('prior_only'))}$. "
               f"The explainers see {r['n_edge_items']} neighbour edges and {r['n_history_items']} history "
               f"events.")
    cf, pre, var = explanation(r, "joint-cf"), explanation(r, "joint"), explanation(r, "variational")
    v1 = explanation(r, "gnnexplainer-v1")
    if cf is not None and cf["valid"] and cf["selected"]:
        main, what = cf, "counterfactual"
        sent = (f"The PROVEX counterfactual explanation removes {len(cf['selected'])} of {cf['n_items']} items, "
                f"which lowers $L$ to {num(cf['L_selected'])} ($<\\tau$).")
    elif pre is not None and pre["valid"] and pre["selected"]:
        main, what = pre, "preserve"
        sent = (f"The PROVEX preserve explanation keeps {len(pre['selected'])} of {pre['n_items']} items, "
                f"which on their own hold $L$ at {num(pre['L_selected'])} ($\\ge\\tau$).")
    else:
        main, what = None, None
        sent = ("The counterfactual search found no removal inside the window that flips this alert, and the "
                "empty set already sustains it, so PROVEX reports the event and its channel decomposition "
                "instead of an item ranking.")
    out.append(sent)
    if main is not None:
        d1, d5 = deletion(main, 1), deletion(main, 5)
        mets = main.get("metrics", {})
        s = f"Deleting its top-1 and top-5 items gives $L={num(d1)}$ and $L={num(d5)}$"
        if "min_cf_size" not in mets:                 # only computed for window-explainable alerts
            s += "."
        elif mets["min_cf_size"] is not None:
            s += f"; the smallest flipping prefix has {mets['min_cf_size']} items."
        else:
            s += "; no prefix of its ranking (up to 256 items) flips the alert."
        if v1 is not None and deletion(v1, 5) is not None:
            v5 = deletion(v1, 5)
            s += (f" Deleting the top-5 items of the v1 objective gives $L={num(v5)}$, so the alert "
                  f"{'flips' if v5 < tau else 'stays'}.")
        out.append(s)
        top = main["top_items"][:k]
        var_by_ev = {it["ev"]: it for it in (var or {}).get("top_items", [])}
        sel = set(main["selected"])
        n_gt = sum(1 for it in top if it["src"] in gt or it["dst"] in gt)
        out.append(f"{n_gt} of these top-{len(top)} items touch ground-truth attack entities "
                   f"(Table~\\ref{{tab:case-{day.replace('_', '-')}}}).")
        rows = []
        for i, it in enumerate(top, 1):
            v = var_by_ev.get(it["ev"])
            pi = f"{num(v['score'])}$\\pm${num(v.get('std_m'))}" if v is not None else "--"
            chan = "+".join("hist." if c == "history" else c for c in it.get("channels", [])) or "--"
            rows.append(f"{i} & {entity(it['src_label'], it['src'], gt)} $\\rightarrow$ "
                        f"{entity(it['dst_label'], it['dst'], gt)} & \\texttt{{{etype(it.get('type'))}}} & "
                        f"{chan} & {ago((e['t'] - it['t']) / 1e9)} & {num(it['score'])} & "
                        f"{'yes' if it['ev'] in sel else 'no'} & {pi} \\\\")
        label = "removed by the counterfactual" if what == "counterfactual" else "kept by the preserve set"
        out += ["\\begin{table*}[t]", "\\centering\\footnotesize",
                f"\\caption{{Case study, {lab}: the top-{len(top)} items of the PROVEX {what} explanation for the "
                f"alert event above. Channel: neighbour edge of the current batch or memory update from an "
                f"earlier batch (hist.). $\\Delta t$: time before the alert event. In set: {label}. "
                f"$\\pi\\pm$sd: the variational explainer's inclusion probability and its standard deviation "
                f"(-- if outside its top 25). Bold: ground-truth attack entity.}}",
                f"\\label{{tab:case-{day.replace('_', '-')}}}",
                "\\begin{tabular}{rp{0.46\\textwidth}lllrcc}", "\\toprule",
                "\\# & Event & Type & Channel & $\\Delta t$ & Score & In set & $\\pi\\pm$sd \\\\", "\\midrule",
                *rows, "\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out + [""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--gt-dir", default="")
    ap.add_argument("--out", default="paper/generated")
    ap.add_argument("--days", default=",".join(DAYS))
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    gt_by_day = load_gt_nodes(a.gt_dir) if a.gt_dir and Path(a.gt_dir).is_dir() else {}
    parts = ["% generated by scripts/make_case_studies.py -- do not edit",
             "For each attack day we explain one alert, chosen by a fixed rule rather than by hand: the "
             "highest-scoring alert event inside a ground-truth attack window whose endpoint is an attack entity, "
             "preferring window-explainable alerts. Scores are from KAIROS seed "
             f"{a.model_seed}.", ""]
    for day in [d for d in a.days.split(",") if d]:
        parts.append(case(day, load_records(Path(a.results), a.model_seed, day), gt_by_day.get(day, set()), a.k))
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    (O / "case_studies.tex").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {O / 'case_studies.tex'}")


if __name__ == "__main__":
    main()