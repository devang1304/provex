"""Render each alert's Tier-1 story (triage order) as a LaTeX table for the case studies.

    python scripts/make_story_tex.py --stories results/stories/stories.json --out paper/generated

Writes story_<day>.tex per alert (the alert-level top actions) with the facts the story attaches to each
action. Nothing is typed by hand: entities, times and flags come from stories.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DAY = {"4_6": "04-06", "4_11": "04-11", "4_12": "04-12", "4_13": "04-13"}
SHORT_TYPE = {"EVENT_RECVFROM": "receive", "EVENT_SENDTO": "send", "EVENT_WRITE": "write", "EVENT_READ": "read",
              "EVENT_EXECUTE": "execute", "EVENT_OPEN": "open", "EVENT_CLOSE": "close"}


def tex(s: str, width: int = 34) -> str:
    s = s.partition(": ")[2] or s
    if len(s) > width:
        s = "..." + s[-(width - 3):]
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"),
                 ("$", "\\$"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return "\\texttt{" + s + "}"


def flags(x) -> str:
    f = []
    if x.get("external_ip_new_today"):
        f.append("new ext.\\ IP")
    if x.get("new_program_runs"):
        f.append("new program")
    if x.get("rare"):
        f.append("rare")
    return ", ".join(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    stories = json.loads(Path(a.stories).read_text())["alerts"] if Path(a.stories).exists() else []
    written = set()
    for st in stories:
        day = st["day"]
        rows = []
        for x in st["tier1_alert"][:a.k]:
            t = x["first"][:5] if x["first"][:5] == x["last"][:5] else f"{x['first'][:5]}--{x['last'][:5]}"
            rows.append(f"{t} & {tex(x['src_label'])} & {SHORT_TYPE.get(x['type'], x['type'])} & "
                        f"{tex(x['dst_label'])} & {x['count']} & {flags(x)} \\\\")
        cap = (f"Tier-1 story of the {DAY.get(day, day)} alert ({st['n_windows']} windows, "
               f"{st['n_flagged_events']:,} flagged events), top {len(rows)} actions in triage order. "
               "Times are US/Eastern; $n$ counts events; flags are the facts the story attaches.")
        body = ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\caption{" + cap + "}\n\\label{tab:story-"
                + day.replace("_", "-") + "}\n\\fittable{\\begin{tabular}{lllllp{0.22\\textwidth}}\n\\toprule\n"
                "Time & Subject & Action & Object & $n$ & Flags \\\\\n\\midrule\n" + "\n".join(rows)
                + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
        (O / f"story_{day}.tex").write_text(body)
        written.add(day)
    for day in ("4_6", "4_12"):                       # the paper inputs these; keep CI compiling on empty results
        if day not in written:
            (O / f"story_{day}.tex").write_text("% no story\n")
    print("wrote", sorted(written))


if __name__ == "__main__":
    main()