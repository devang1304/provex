"""Generate Markdown analyst reports from explanation artifacts."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pytz

try:  # Optional GPT dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore # pylint: disable=invalid-name

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = object

EST = pytz.timezone("US/Eastern")


@dataclass
class EdgeSummary:
    """Container for aggregated edge statistics."""
    src: str
    dst: str
    relation: str
    weight: float
    count: int
    rank_score: float
    timestamps: List[str]


def _to_local_time(ns: int) -> str:
    seconds, nanos = divmod(int(ns), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, pytz.UTC).astimezone(EST)
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{str(nanos).zfill(9)}"


def _format_timestamp(raw: Any) -> str:
    if raw is None:
        return "—"
    if isinstance(raw, (int, float)):
        try:
            return _to_local_time(int(raw))
        except (ValueError, TypeError):
            return str(int(raw))
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return "—"
        if stripped.isdigit():
            try:
                return _to_local_time(int(stripped))
            except (ValueError, TypeError):
                return stripped
        try:
            iso = stripped.replace("T", " ")
            if iso.endswith("Z"):
                iso = iso[:-1] + "+00:00"
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            else:
                dt = dt.astimezone(pytz.UTC)
            return dt.astimezone(EST).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return stripped
    return str(raw)


def _resolve_node(node_id: int, node_map: Dict[int, str]) -> str:
    label = node_map.get(node_id)
    return label if label else f"Node {node_id}"


def _summarise_edges(
    aggregate: Optional[List[Dict[str, object]]], node_map: Dict[int, str]
) -> List[EdgeSummary]:
    if not aggregate or not isinstance(aggregate, list):
        return []
    items: List[EdgeSummary] = []
    for entry in aggregate:
        src_id = entry.get("src")
        dst_id = entry.get("dst")
        if src_id is None or dst_id is None:
            continue
        src = _resolve_node(int(src_id), node_map) if isinstance(
            src_id, (int, float)
        ) else str(src_id)
        dst = _resolve_node(int(dst_id), node_map) if isinstance(
            dst_id, (int, float)
        ) else str(dst_id)
        relation = entry.get("relation", "?")
        weight = float(entry.get("weight", 0.0))
        count = int(entry.get("count", 0))
        rank_score = weight * (1.0 + math.log1p(max(count, 0)))
        timestamps_raw = (entry.get("timestamps") or [])[:5]
        timestamps = [_format_timestamp(ts) for ts in timestamps_raw]
        items.append(EdgeSummary(src, dst, relation, weight, count, rank_score, timestamps))
    items.sort(
        key=lambda e: e.rank_score,
        reverse=True,
    )
    return items


def _load_node_map(mapping_path: Optional[Path]) -> Dict[int, str]:
    if mapping_path and mapping_path.exists():
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data.items()}
    return {}


def _prepare_node_entries(
    nodes: List[Dict[str, Any]], node_map: Dict[int, str]
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for node in nodes:
        label = _resolve_node(node["node_id"], node_map)
        gnn_edges: List[Dict[str, Any]] = []
        for gnn_event in node.get("gnn", []):
            for edge in gnn_event.get("top_edges", []):
                ts_raw = edge.get("timestamp")
                when = _format_timestamp(ts_raw)
                src_id = edge.get("src")
                dst_id = edge.get("dst")
                src_label = (_resolve_node(int(src_id), node_map)
                             if isinstance(src_id, (int, float)) else "Unknown")
                dst_label = (_resolve_node(int(dst_id), node_map)
                             if isinstance(dst_id, (int, float)) else "Unknown")
                gnn_edges.append(
                    {
                        "src": src_label,
                        "dst": dst_label,
                        "relation": edge.get("relation", "?"),
                        "weight": edge.get("weight", 0.0),
                        "when": when,
                    }
                )
        gnn_edges.sort(key=lambda e: e["weight"], reverse=True)

        va = node.get("va_tg", {})
        va_edges = []
        for entry in va.get("aggregate", []):
            ts_list = entry.get("timestamps", []) or []
            first_ts = ts_list[0] if ts_list else None
            when = _format_timestamp(first_ts)
            src_id = entry.get("src")
            dst_id = entry.get("dst")
            src_label = (_resolve_node(int(src_id), node_map)
                         if isinstance(src_id, (int, float)) else "Unknown")
            dst_label = (_resolve_node(int(dst_id), node_map)
                         if isinstance(dst_id, (int, float)) else "Unknown")
            va_edges.append(
                {
                    "src": src_label,
                    "dst": dst_label,
                    "relation": entry.get("relation", "?"),
                    "weight": entry.get("weight", 0.0),
                    "when": when,
                }
            )
        va_edges.sort(key=lambda e: e["weight"], reverse=True)

        raw_score = float(node.get("score", 0.0))
        event_count = int(node.get("event_count", 0) or 0)
        max_event_loss = float(node.get("max_event_loss", 0.0) or 0.0)
        if max_event_loss == 0.0:
            event_losses = node.get("event_losses") or []
            if event_losses:
                max_event_loss = max((float(val) for val in event_losses), default=0.0)
        avg_score_raw = node.get("avg_score")
        if avg_score_raw is None:
            avg_score = raw_score / event_count if event_count else raw_score
        else:
            avg_score = float(avg_score_raw)

        prepared.append(
            {
                "label": label,
                "score": avg_score,
                "total_score": raw_score,
                "event_count": event_count,
                "max_event_loss": max_event_loss,
                "gnn_edges": gnn_edges,
                "va_edges": va_edges,
            }
        )
    prepared.sort(key=lambda e: e["score"], reverse=True)
    return prepared


def _call_gpt_section(
    client: OpenAIClient,
    question: str,
    instructions: str,
    summary_payload: Dict[str, Any],
    format_hint: str,
) -> Optional[str]:
    system_prompt = (
        "You are an incident-response analyst reviewing PROVEX temporal graph explanations. "
        "Use concise, plain language so non-experts can understand. "
        "Reason strictly from the supplied JSON payload; do not speculate beyond the data."
    )
    user_prompt = (
        f"Answer the question: {question}\n"
        f"{instructions}\n"
        f"Formatting rules: {format_hint}"
    )
    print(f"[GPT] Preparing section '{question}'")

    input_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": json.dumps(summary_payload, sort_keys=True)},
    ]
    stream_input: Any = input_payload

    try:
        start_time = time.time()
        with client.responses.stream(
            model="gpt-5",
            input=stream_input,
            reasoning={"effort": "low"},
            text={"verbosity": "medium"},
        ) as stream:
            collected: List[str] = []
            for event in stream:
                if event.type == "response.output_text.delta":
                    chunk = event.delta
                    if chunk:
                        collected.append(chunk)
                elif event.type == "response.completed":
                    pass
            stream.close()
        text = "".join(collected).strip()
        elapsed = time.time() - start_time
        print(f"[GPT] Section '{question}' completed in {elapsed:.2f}s")
        return text if text else None
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[GPT] Section '{question}' failed: {exc}")
        return None


def _build_gpt_analysis(summary_payload: Dict[str, Any]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key)
    sections = [
        (
            "What happened?",
            "Summarise the overall activity and whether PROVEX detections align with any "
            "ground-truth attacks noted in the payload. Mention the attack goal if it is obvious.",
            "Return one Markdown paragraph containing 2-3 sentences, each no longer than "
            "20-25 words.",
        ),
        (
            "Who's involved?",
            "List the top suspicious nodes or processes with human-readable names, their "
            "role/evidence, average loss (avg_loss), peak loss for any single event "
            "(max_event_loss), total loss (total_loss), event counts (event_count), whether any "
            "event meets or exceeds the threshold (above_threshold), and whether the model marks "
            "them as malicious.",
            "Return a Markdown table with columns | Node | Role/Evidence | Avg Loss | Peak Loss | "
            "Events | Total Loss | >= Threshold | Model Alignment |. "
            "Use 'Yes' or 'No' in the >= Threshold column. Limit to 5 rows.",
        ),
        (
            "Why flagged?",
            "Highlight key relations or events (include timestamps when available) that triggered "
            "the alert, using readable node names.",
            "Return 3-5 Markdown bullet points starting with '- '.",
        ),
        (
            "What's missing or risky?",
            "Describe gaps, blind spots, or potential false positives implied by the payload.",
            "Return 2-3 Markdown bullet points starting with '- '.",
        ),
        (
            "What next?",
            "Provide concrete follow-up actions for analysts or model owners grounded strictly "
            "in the payload.",
            "Return 2-3 Markdown bullet points starting with '- '.",
        ),
    ]

    sections_md: List[str] = []
    for question, instructions, format_hint in sections:
        content = _call_gpt_section(client, question, instructions, summary_payload, format_hint)
        if not content:
            return None
        sections_md.append(f"## {question}\n{content.strip()}\n")

    return "\n".join(sections_md).strip()


def _render_markdown(context: Dict[str, Any]) -> str:
    lines = [
        "# PROVEX Explanation Report",
        "",
        f"**Window**: {context['window_path']}  ",
        f"**Events**: {context['num_events']}  ",
        f"**Threshold**: {context['threshold']:.4f}",
    ]

    high_loss_candidates = context.get("high_loss_candidates", 0)
    high_loss_used = context.get("high_loss_used", 0)
    graphmask_events = context.get("graphmask_events", 0)
    if high_loss_used > 0:
        lines.append(
            f"**High-loss events analysed**: {high_loss_used} of {high_loss_candidates} "
            f"(GraphMask runs: {graphmask_events})  "
        )
    elif high_loss_candidates > 0:
        lines.append(
            f"**High-loss events analysed**: none (0 of {high_loss_candidates}; "
            f"GraphMask runs: {graphmask_events})  "
        )

    metrics_note = context.get("metrics_note")
    if metrics_note:
        lines.append(f"*Info:* {metrics_note}")

    lines.append("")

    if context.get("gpt_summary"):
        lines.append(str(context["gpt_summary"]))
        lines.append("")

    lines.append("## Node Score Overview")
    lines.append(
        "| Node | Avg Loss | Peak Loss | Events | Total Loss | >= Threshold |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for node in context["nodes"][:8]:
        total_loss = node.get("total_score", 0.0)
        peak_loss = node.get("max_event_loss", 0.0)
        meets_threshold = "Yes" if node.get("above_threshold") else "No"
        lines.append(
            f"| {node['label']} | {node['score']:.3f} | {peak_loss:.3f} | "
            f"{node.get('event_count', 0)} | {total_loss:,.3f} | {meets_threshold} |"
        )
    lines.append("")

    lines.append("## Key Graph Insights (GraphMask)")
    lines.append(
        "| Relation | Source | Destination | Weight | Count | Rank Score | Example Times |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for edge in context["graph_edges"][:10]:
        times = ", ".join(edge.timestamps) if edge.timestamps else "—"
        lines.append(
            f"| {edge.relation} | {edge.src} | {edge.dst} | {edge.weight:.3f} | "
            f"{edge.count} | {edge.rank_score:.3f} | {times} |"
        )
    lines.append("")

    lines.append("## Node-Level Explanations")
    for node in context["nodes"][:8]:
        event_count = node.get("event_count", 0)
        total_loss = node.get("total_score", 0.0)
        peak_loss = node.get("max_event_loss", 0.0)
        lines.append(
            f"### {node['label']} (avg_loss={node['score']:.3f}, peak_loss={peak_loss:.3f}, "
            f"events={event_count}, total_loss={total_loss:,.3f})"
        )
        if node["gnn_edges"]:
            top = ", ".join(
                f"{edge['relation']} {edge['src']} → {edge['dst']} "
                f"({edge['weight']:.3f}, {edge.get('when', '—')})"
                for edge in node["gnn_edges"][:3]
            )
            lines.append(f"- GNN top edges: {top}")
        if node["va_edges"]:
            top = ", ".join(
                f"{edge['relation']} {edge['src']} → {edge['dst']} "
                f"({edge['weight']:.3f}, {edge.get('when', '—')})"
                for edge in node["va_edges"][:3]
            )
            lines.append(f"- VA aggregate edges: {top}")
        lines.append("")
    return "\n".join(lines)


def build_reports(
    report_json: Dict[str, object],
    output_dir: Path,
    node_mapping_path: Optional[Path] = None,
    run_gpt: bool = True,
    existing_summary: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Orchestrate the creation of Markdown reports from explanation data.
    """
    node_map = _load_node_map(node_mapping_path)

    window_path = report_json.get("window_path", "unknown")
    num_events = report_json.get("num_events", 0)

    graphmask_data = report_json.get("graphmask") or {}
    aggregate_edges_raw = graphmask_data.get("aggregate")
    aggregate_edges = aggregate_edges_raw if isinstance(
        aggregate_edges_raw, list
    ) else []
    graph_edges = _summarise_edges(aggregate_edges, node_map)

    node_entries = _prepare_node_entries(report_json.get("nodes", []), node_map)

    threshold_raw = report_json.get("threshold")
    threshold_value = float(
        threshold_raw) if threshold_raw is not None else 0.0
    for node in node_entries:
        node["above_threshold"] = node.get(
            "max_event_loss", 0.0) >= threshold_value

    metrics = report_json.get("metrics") or {}
    per_event_raw = graphmask_data.get("per_event")
    per_event = per_event_raw if isinstance(per_event_raw, list) else []
    inferred_default = len(per_event)
    metrics_provided = bool(metrics)
    high_loss_candidates = (int(metrics.get("high_loss_candidates", inferred_default))
                            if metrics else inferred_default)
    high_loss_used = int(metrics.get("high_loss_used", inferred_default)) if metrics else 0
    graphmask_events = (int(metrics.get("graphmask_events", inferred_default))
                        if metrics else inferred_default)
    metrics_note: Optional[str] = None
    if not metrics_provided and graphmask_events:
        warn_threshold = 512
        metrics_note = (
            f"GraphMask statistics inferred from payload (per_event count={graphmask_events})."
            if graphmask_events <= warn_threshold
            else f"GraphMask statistics inferred from payload; large event count "
            f"({graphmask_events}) may indicate fallback values."
        )

    summary_payload = {
        "window": window_path,
        "threshold": threshold_value,
        "num_events": num_events,
        "attack_windows": report_json.get("attack_windows"),
        "analysis_stats": {
            "high_loss_candidates": high_loss_candidates,
            "high_loss_used": high_loss_used,
            "graphmask_events": graphmask_events,
            "metrics_inferred": not metrics_provided,
            "metrics_note": metrics_note,
        },
        "graphmask": {
            "total_edges": len(graph_edges),
            "top_edges": [
                {
                    **edge.__dict__,
                    "rank_score": edge.weight * (1.0 + math.log1p(max(edge.count, 0))),
                }
                for edge in graph_edges[:10]
            ],
        },
        "node_scores": [
            {
                "label": node["label"],
                "avg_loss": node["score"],
                "total_loss": node.get("total_score", 0.0),
                "event_count": node.get("event_count", 0),
                "above_threshold": node.get("above_threshold", False),
                "max_event_loss": node.get("max_event_loss", 0.0),
            }
            for node in node_entries[:10]
        ],
        "top_nodes": [
            {
                "label": node["label"],
                "avg_loss": node["score"],
                "total_loss": node.get("total_score", 0.0),
                "event_count": node.get("event_count", 0),
                "above_threshold": node.get("above_threshold", False),
                "max_event_loss": node.get("max_event_loss", 0.0),
                "gnn_edges": node["gnn_edges"][:5],
                "va_edges": node["va_edges"][:5],
            }
            for node in node_entries[:10]
        ],
    }
    gpt_summary: Optional[str] = existing_summary
    if run_gpt:
        generated_summary = _build_gpt_analysis(summary_payload)
        if generated_summary:
            gpt_summary = generated_summary
        else:
            print("[warn] GPT analysis unavailable; continuing without AI summary.")

    context = {
        "window_path": window_path,
        "num_events": num_events,
        "threshold": threshold_value,
        "graph_edges": graph_edges,
        "nodes": node_entries,
        "gpt_summary": gpt_summary,
        "high_loss_candidates": high_loss_candidates,
        "high_loss_used": high_loss_used,
        "graphmask_events": graphmask_events,
        "metrics_note": metrics_note,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(window_path)
    base_name = raw_path.stem.replace(":", "_").replace("/", "_")
    if not base_name:
        base_name = "window"
    md_path = output_dir / f"{base_name}_report.md"

    md_path.write_text(_render_markdown(context), encoding="utf-8")

    return md_path, gpt_summary
