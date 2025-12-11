"""
Minimal Streamlit dashboard for PROVEX explanation artifacts.

Run from the /provex directory:
    streamlit run reporting/streamlit_dashboard.py
"""
from __future__ import annotations
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import pytz
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components


CAD_DIR = Path(__file__).resolve().parents[1]
if str(CAD_DIR) not in sys.path:
    sys.path.insert(0, str(CAD_DIR))

# pylint: disable=wildcard-import,unused-wildcard-import
import config
from config import *
# pylint: enable=wildcard-import,unused-wildcard-import

EST = pytz.timezone("US/Eastern")
EXPLANATION_DIR = CAD_DIR / "artifact" / "explanations"
DEFAULT_PATTERN = "*_explanations.json"
MAX_ROWS = 15


@st.cache_data(show_spinner=False)
def load_mapping(path: Optional[Path]) -> Dict[int, str]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


@st.cache_data(show_spinner=False)
def load_payload(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_local_time(value: Optional[int]) -> str:
    if value is None:
        return "—"
    seconds, nanos = divmod(int(value), 1_000_000_000)
    # Use fromtimestamp with timezone to avoid naive datetime issues
    dt = datetime.fromtimestamp(seconds, pytz.UTC).astimezone(EST)
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{str(nanos).zfill(9)}"


def resolve_label(node_id: Optional[int], mapping: Dict[int, str]) -> str:
    """Resolve node ID to full label from mapping (activity: subject format)."""
    if node_id is None:
        return "Unknown"
    return mapping.get(int(node_id), f"Node {node_id}")


def parse_activity_subject(label: str) -> Tuple[str, str]:
    """Parse 'activity: subject' format into (activity, subject) tuple."""
    if ": " in label:
        parts = label.split(": ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return "", label


def get_display_label(label: str, max_len: int = 25) -> str:
    """Get a short display label for graphs - just the subject basename."""
    _, subject = parse_activity_subject(label)

    # For file paths, extract just the basename
    if "/" in subject:
        basename = subject.rsplit("/", 1)[-1]
        display = basename if basename else subject
    else:
        display = subject

    # Truncate if still too long
    if len(display) > max_len:
        display = display[:max_len-3] + "..."

    return display


@st.cache_data(show_spinner=False)
def summarise_graph_edges(payload: Dict[str, object], mapping: Dict[int, str]) -> List[Dict[str, object]]:
    aggregate = payload.get("graphmask", {}).get("aggregate")
    if not isinstance(aggregate, list):
        return []
    rows: List[Dict[str, object]] = []
    for item in aggregate:
        src = resolve_label(item.get("src"), mapping)
        dst = resolve_label(item.get("dst"), mapping)
        relation = item.get("relation", "?")
        weight = float(item.get("weight", 0.0))
        count = int(item.get("count", 0))
        timestamps = item.get("timestamps", [])
        first_seen = to_local_time(timestamps[0]) if timestamps else "—"
        last_seen = to_local_time(timestamps[-1]) if timestamps else "—"
        rows.append(
            {
                "relation": relation,
                "source": src,
                "destination": dst,
                "weight": f"{weight:.3f}",
                "count": str(count),
                "first_seen": first_seen,
                "last_seen": last_seen,
            }
        )
    rows.sort(key=lambda r: float(r["weight"]), reverse=True)
    return rows


@st.cache_data(show_spinner=False)
def summarise_nodes(payload: Dict[str, object], mapping: Dict[int, str], threshold: float) -> List[Dict[str, object]]:
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    rows: List[Dict[str, object]] = []
    for node in nodes:
        label = resolve_label(node.get("node_id"), mapping)
        avg_loss = float(node.get("avg_score", node.get("score", 0.0)))
        total_loss = float(node.get("score", 0.0))
        event_count = int(node.get("event_count", 0))
        peak_loss = float(node.get("max_event_loss", 0.0))
        above = float(node.get("max_event_loss", 0.0)) >= threshold
        rows.append(
            {
                "label": label,
                "avg_loss": f"{avg_loss:.3f}",
                "peak_loss": f"{peak_loss:.3f}",
                "events": str(event_count),
                "total_loss": f"{total_loss:,.3f}",
                "above": above,
            }
        )
    rows.sort(key=lambda r: float(r["avg_loss"]), reverse=True)
    return rows


def summarise_node_detail(node: Dict[str, object], mapping: Dict[int, str]) -> Dict[str, List[Tuple[str, str]]]:
    def flatten_edges(source: List[Dict[str, object]]) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        for entry in source or []:
            relation = entry.get("relation", "?")
            src = resolve_label(entry.get("src"), mapping) if entry.get(
                "src") is not None else "Unknown"
            dst = resolve_label(entry.get("dst"), mapping) if entry.get(
                "dst") is not None else "Unknown"
            weight = float(entry.get("weight", 0.0))
            rows.append((f"{relation} {src} → {dst}", f"{weight:.3f}"))
        rows.sort(key=lambda r: float(r[1]), reverse=True)
        return rows

    gnn_edges: List[Tuple[str, str]] = []
    for event in node.get("gnn", []) or []:
        gnn_edges.extend(flatten_edges(event.get("top_edges", []))[:5])

    va_edges: List[Tuple[str, str]] = []
    for entry in node.get("va_tg", {}).get("events", []) or []:
        va_edges.extend(flatten_edges(entry.get("edges", []))[:5])
    va_aggregate = flatten_edges(node.get("va_tg", {}).get("aggregate", []))

    return {
        "gnn_edges": gnn_edges[:MAX_ROWS],
        "va_edges": va_edges[:MAX_ROWS],
        "va_aggregate": va_aggregate[:MAX_ROWS],
    }


def render_table(title: str, rows: List[Dict[str, object]], columns: List[str], empty_message: str = "No data.") -> None:
    st.markdown(f"#### {title}")
    if not rows:
        st.info(empty_message)
        return
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows[:MAX_ROWS]:
        cells = [str(row.get(column, "—")) for column in columns]
        body.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join([header, divider, *body]))
    if len(rows) > MAX_ROWS:
        st.caption(f"Showing first {MAX_ROWS} of {len(rows)} rows.")


def render_pairs(title: str, pairs: List[Tuple[str, str]]) -> None:
    st.markdown(f"##### {title}")
    if not pairs:
        st.info("No entries.")
        return
    header = "| Item | Weight |"
    divider = "| --- | ---: |"
    body = ["| " + label + " | " + weight +
            " |" for label, weight in pairs[:MAX_ROWS]]
    st.markdown("\n".join([header, divider, *body]))


def _relation_color(rel: str) -> str:
    palette = {
        "EVENT_WRITE": "#8dd3c7",
        "EVENT_READ": "#ffffb3",
        "EVENT_CLOSE": "#bebada",
        "EVENT_OPEN": "#fb8072",
        "EVENT_EXECUTE": "#80b1d3",
        "EVENT_SENDTO": "#fdb462",
        "EVENT_RECVFROM": "#b3de69",
    }
    return palette.get(rel, "#d9d9d9")


def plot_top_edges_bar(graph_rows: List[Dict[str, object]], k: int = 10) -> go.Figure:
    top = graph_rows[:k]
    labels = [f"{r['relation']} {r['source']} → {r['destination']}" for r in top]
    weights = [float(r["weight"]) for r in top]
    colors = [_relation_color(r["relation"]) for r in top]
    hover = [
        f"<b>{r['relation']}</b><br>src: {r['source']}<br>dst: {r['destination']}<br>weight: {r['weight']}<br>count: {r['count']}<br>first: {r['first_seen']}<br>last: {r['last_seen']}"
        for r in top
    ]
    fig = go.Figure(
        go.Bar(
            x=weights,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{customdata}",
            customdata=hover,
            name="Weight",
            showlegend=True,
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Top GraphMask edges", x=0.5, xanchor="center"),
        xaxis_title="Weight",
        yaxis_title="Edge (relation src → dst)",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.05, xanchor="right", x=1),
    )
    fig.update_yaxes(automargin=True)
    return fig


def plot_node_scores_bar(node_rows: List[Dict[str, object]], threshold: float, k: int = 10) -> go.Figure:
    top = node_rows[:k]
    labels = [r["label"] for r in top]
    scores = [float(r["avg_loss"]) for r in top]
    colors = ["#d62728" if r.get("above") else "#2ca02c" for r in top]
    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="avg_loss=%{x:.3f}<extra></extra>",
            name="Avg loss",
            showlegend=True,
        )
    )
    # Add a legendable threshold line as a separate scatter trace
    if labels:
        fig.add_trace(
            go.Scatter(
                x=[threshold, threshold],
                y=[-0.5, len(labels) - 0.5],
                mode="lines",
                line=dict(color="#7f7f7f", width=2, dash="dash"),
                name="Event‑loss threshold",
                hoverinfo="skip",
                showlegend=True,
            )
        )
    # Annotation for threshold semantics
    if labels:
        fig.add_annotation(
            x=threshold,
            y=1.02,
            xref="x",
            yref="paper",
            text="event‑loss threshold",
            showarrow=False,
            font=dict(color="#7f7f7f", size=12),
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Top nodes by average loss", x=0.5, xanchor="center"),
        xaxis_title="Average loss",
        yaxis_title="Node",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.05, xanchor="right", x=1),
    )
    fig.update_yaxes(automargin=True)
    return fig


def plot_event_timeline(payload: Dict[str, object]) -> go.Figure:
    """Plot event timeline scaled to the full attack window with 0s for missing minutes."""
    # Get attack window boundaries
    start_ns = payload.get("start_ns")
    end_ns = payload.get("end_ns")

    # Use all aggregate timestamps across edges; bin by minute.
    aggregate = payload.get("graphmask", {}).get("aggregate")
    if not isinstance(aggregate, list):
        return go.Figure()

    counts: Dict[str, int] = {}
    for entry in aggregate:
        for ts in entry.get("timestamps", []) or []:
            # Accept int, float, and digit-like strings; skip anything else
            if isinstance(ts, str):
                if ts.isdigit():
                    ts = int(ts)
                else:
                    continue
            seconds, _ = divmod(int(ts), 1_000_000_000)
            dt = datetime.fromtimestamp(seconds, pytz.UTC).astimezone(EST)
            key = dt.strftime("%Y-%m-%d %H:%M")
            counts[key] = counts.get(key, 0) + 1

    if not counts and not (start_ns and end_ns):
        return go.Figure()

    # Generate all minutes in the attack window and fill with 0s for missing data
    all_minutes = []
    if start_ns and end_ns:
        try:
            start_sec, _ = divmod(int(start_ns), 1_000_000_000)
            end_sec, _ = divmod(int(end_ns), 1_000_000_000)
            start_dt = datetime.fromtimestamp(start_sec, pytz.UTC).astimezone(EST)
            end_dt = datetime.fromtimestamp(end_sec, pytz.UTC).astimezone(EST)
            # Round down to minute
            current = start_dt.replace(second=0, microsecond=0)
            end_minute = end_dt.replace(second=0, microsecond=0)
            while current <= end_minute:
                all_minutes.append(current.strftime("%Y-%m-%d %H:%M"))
                current = current + pd.Timedelta(minutes=1)
        except Exception:
            pass

    # If we have attack window, use all minutes; otherwise use existing data
    if all_minutes:
        x = all_minutes
        y = [counts.get(m, 0) for m in all_minutes]
    else:
        items = sorted(counts.items())
        x = [k for k, _ in items]
        y = [v for _, v in items]

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="Events/min",
            showlegend=True,
        )
    )

    layout_kwargs = dict(
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time (minute)",
        yaxis_title="Events",
        title=dict(text="Event Timeline (Attack Window)", x=0.5, xanchor="center"),
    )

    fig.update_layout(**layout_kwargs)
    return fig


def plot_sankey(graph_rows: List[Dict[str, object]]) -> go.Figure:
    """Plots a Sankey diagram of the whole subgraph with clean labels."""
    if not graph_rows:
        return go.Figure()

    # Create a set of all unique nodes
    nodes = set()
    relations = set()
    for row in graph_rows:
        nodes.add(row["source"])
        nodes.add(row["destination"])
        relations.add(row["relation"])

    node_list = list(nodes)
    # Create display labels for Sankey nodes (just subject basename)
    node_display_labels = [get_display_label(node) for node in node_list]
    node_map = {node: i for i, node in enumerate(node_list)}

    sources = [node_map[row["source"]] for row in graph_rows]
    targets = [node_map[row["destination"]] for row in graph_rows]
    values = [float(row["weight"]) for row in graph_rows]
    # Simplified labels for clarity
    link_labels = [f"{row['relation']}" for row in graph_rows]
    link_colors = [_relation_color(row["relation"]) for row in graph_rows]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",  # Better node positioning
        node=dict(
            pad=30,  # More space between nodes
            thickness=25,  # Thicker nodes for visibility
            line=dict(color="black", width=0.5),
            label=node_display_labels,
            customdata=node_list,  # Full labels for hover
            hovertemplate='%{customdata}<extra></extra>',
            color="#4a90d9"  # Better contrast color
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>%{label}<br>Weight: %{value:.3f}<extra></extra>'
        )
    )])

    # Add dummy traces for legend
    for rel in sorted(relations):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=_relation_color(rel)),
            legendgroup=rel,
            showlegend=True,
            name=rel
        ))

    fig.update_layout(
        title_text="Event Flow (Sankey)",
        font=dict(size=14, color="#333"),  # Larger, darker font
        height=700,  # Taller chart
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        )
    )
    return fig


def plot_heatmap(payload: Dict[str, object], mapping: Dict[int, str]) -> go.Figure:
    """Plots a heatmap of node activity over time (log scale)."""
    aggregate = payload.get("graphmask", {}).get("aggregate")
    if not isinstance(aggregate, list):
        return go.Figure()

    data = []
    for entry in aggregate:
        src_id = entry.get("src")
        try:
            src_id_int = int(src_id)
            full_label = resolve_label(src_id_int, mapping)
            display_label = get_display_label(full_label)
        except Exception:
            display_label = str(src_id)

        for ts in entry.get("timestamps", []) or []:
            if isinstance(ts, str) and not ts.isdigit():
                continue
            ts_int = int(ts)
            seconds, _ = divmod(ts_int, 1_000_000_000)
            dt = datetime.fromtimestamp(seconds, pytz.UTC).astimezone(EST)
            data.append({
                "Time": dt.strftime("%Y-%m-%d %H:%M"),
                "Node": display_label
            })

    if not data:
        return go.Figure()

    df = pd.DataFrame(data)

    # Pre-aggregate counts for log scale
    counts = df.groupby(["Time", "Node"]).size().reset_index(name="Count")
    # Apply log transform
    import numpy as np
    counts["LogCount"] = np.log1p(counts["Count"])

    # Pivot for heatmap
    z_data = counts.pivot(index="Node", columns="Time", values="LogCount").fillna(0)
    custom_data = counts.pivot(index="Node", columns="Time", values="Count").fillna(0)

    fig = px.imshow(
        z_data,
        labels=dict(x="Time", y="Node", color="Events (Log)"),
        title="Node Activity Heatmap (Log Scale)",
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig.update_traces(
        customdata=custom_data,
        hovertemplate="Time: %{x}<br>Node: %{y}<br>Events: %{customdata}<extra></extra>"
    )
    fig.update_layout(height=500)

    return fig

def plot_static_network(graph_rows: List[Dict[str, object]], node_rows: List[Dict[str, object]]) -> go.Figure:
    """Plots a static network graph using NetworkX and Plotly with node and edge labels."""
    if not graph_rows:
        return go.Figure()

    G = nx.DiGraph()

    for row in graph_rows:
        src = row["source"]
        dst = row["destination"]
        w = float(row["weight"])
        rel = row["relation"]

        G.add_edge(src, dst, weight=w, relation=rel)

        # Ensure nodes are added
        if src not in G.nodes:
            G.add_node(src)
        if dst not in G.nodes:
            G.add_node(dst)

    # Layout with more spacing
    pos = nx.spring_layout(G, k=1.5, seed=42, iterations=50)

    # Create edge traces with colors based on relation type
    edge_traces = []
    edge_annotations = []

    for edge in G.edges(data=True):
        src, dst, data = edge
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        rel = data.get("relation", "?")
        weight = data.get("weight", 0)

        # Edge line
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=max(1, weight * 3), color=_relation_color(rel)),
            hoverinfo='text',
            text=f"{rel}<br>{src} → {dst}<br>weight: {weight:.3f}",
            mode='lines',
            showlegend=False,
        ))

        # Edge label annotation at midpoint
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        # Shorten relation name for display
        short_rel = rel.replace("EVENT_", "")
        edge_annotations.append(dict(
            x=mid_x, y=mid_y,
            text=short_rel,
            showarrow=False,
            font=dict(size=12, color="#666"),
            bgcolor="rgba(255,255,255,0.7)",
        ))

    # Create node traces with labels
    node_x = []
    node_y = []
    node_labels = []
    node_hover = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Use full mapping label (activity: subject format)
        node_labels.append(node)
        node_hover.append(f"<b>{node}</b>")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_hover,
        text=node_labels,
        textposition="top center",
        textfont=dict(size=16, color="#333"),  # increased font size
        marker=dict(
            color='#1f77b4',
            size=30,  # increased node size for better visibility
            line=dict(width=2, color='#fff'),
        ),
        showlegend=False,
    )


    # Add legend traces for relation types
    relations = set()
    for row in graph_rows:
        relations.add(row["relation"])

    legend_traces = []
    for rel in sorted(relations):
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=4, color=_relation_color(rel)),
            name=rel.replace("EVENT_", ""),
            showlegend=True,
        ))

    fig = go.Figure(
        data=edge_traces + [node_trace] + legend_traces,
        layout=go.Layout(
            title='GraphMask Sub-graph',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            annotations=edge_annotations,
            height=600,
        )
    )
    return fig





def parse_gpt_sections(md: Optional[str]) -> Dict[str, str]:
    if not md:
        return {}
    sections: Dict[str, str] = {}
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def _sanitize_base(stem: str) -> str:
    return stem.replace(":", "_").replace("/", "_") or "window"


def compute_report_md_path(payload: Dict[str, object], base_dir: Path) -> Path:
    raw = str(payload.get("window_path") or "")
    stem = _sanitize_base(Path(raw).stem)
    return (base_dir / f"{stem}_report.md").resolve()


def extract_node_md_sections(md_text: str) -> List[Tuple[str, str]]:
    """Extract (title, content) tuples for node-level subsections from report Markdown."""
    lines = md_text.splitlines()
    in_block = False
    current_title: Optional[str] = None
    buf: List[str] = []
    out: List[Tuple[str, str]] = []
    for line in lines:
        if line.startswith("## ") and "Node-Level Explanations" in line:
            in_block = True
            current_title = None
            buf = []
            continue
        if not in_block:
            continue
        if line.startswith("## ") and "Node-Level Explanations" not in line:
            # End of node section
            if current_title is not None:
                out.append((current_title, "\n".join(buf).strip()))
            break
        if line.startswith("### "):
            if current_title is not None:
                out.append((current_title, "\n".join(buf).strip()))
                buf = []
            current_title = line[4:].strip()
        else:
            if current_title is not None:
                buf.append(line)
    if in_block and current_title is not None:
        out.append((current_title, "\n".join(buf).strip()))
    return out


def _sanitize_mermaid_id(raw: object) -> str:
    try:
        value = int(raw)
    except Exception:
        return f"node_{abs(hash(str(raw))) % 10_000}"
    return f"node_{value}"


def _format_mermaid_label(label: str, avg_loss: float, total_loss: float, events: int) -> str:
    base = (label or "Unknown").replace('"', "'")
    base = base.replace("<", "&lt;").replace(">", "&gt;")
    return f"{base}<br/>avg={avg_loss:.2f}<br/>total={total_loss:.1f}<br/>events={events}"


@st.cache_data(show_spinner=False)
def generate_mermaid_graph(payload: Dict[str, object], node_map: Dict[int, str], max_edges: int = 20) -> str:
    graphmask = payload.get("graphmask", {}) or {}
    aggregate = graphmask.get("aggregate")
    if not isinstance(aggregate, list) or not aggregate:
        return ""

    nodes = payload.get("nodes", []) or []
    node_info: Dict[int, Dict[str, object]] = {}
    for node in nodes:
        nid = node.get("node_id")
        if nid is None:
            continue
        try:
            key = int(nid)
        except Exception:
            continue
        label = resolve_label(key, node_map)
        avg_loss = float(node.get("avg_score", node.get("score", 0.0)))
        total_loss = float(node.get("score", 0.0))
        events = int(node.get("event_count", 0))
        node_info[key] = {
            "label": label,
            "avg": avg_loss,
            "total": total_loss,
            "events": events,
        }

    try:
        sorted_edges = sorted(aggregate, key=lambda item: float(
            item.get("weight", 0.0)), reverse=True)
    except Exception:
        sorted_edges = aggregate
    top_edges = sorted_edges[:max_edges]
    if not top_edges:
        return ""

    lines = ["graph LR"]
    defined_nodes: Dict[str, bool] = {}

    def ensure_node(nid: object) -> str:
        mermaid_id = _sanitize_mermaid_id(nid)
        if mermaid_id not in defined_nodes:
            try:
                numeric = int(nid)
            except Exception:
                numeric = None
            details = node_info.get(numeric, {})
            label = details.get("label") or resolve_label(
                numeric, node_map) if numeric is not None else str(nid)
            avg = float(details.get("avg", 0.0))
            total = float(details.get("total", 0.0))
            events = int(details.get("events", 0))
            text = _format_mermaid_label(label, avg, total, events)
            lines.append(f"    {mermaid_id}[\"{text}\"]")
            defined_nodes[mermaid_id] = True
        return mermaid_id

    for edge in top_edges:
        src = ensure_node(edge.get("src"))
        dst = ensure_node(edge.get("dst"))
        weight = float(edge.get("weight", 0.0))
        count = int(edge.get("count", 0))
        relation = str(edge.get("relation", "?"))
        edge_label = f"{relation} (w={weight:.2f}, n={count})".replace(
            "\"", "'")
        lines.append(f"    {src} -->|{edge_label}| {dst}")

    return "\n".join(lines)


def render_mermaid(code: str, height: int = 600) -> None:
    chart_id = f"mermaid-{uuid4().hex}"
    encoded_code = json.dumps(code)
    html = f"""
    <div id=\"{chart_id}\"></div>
    <script>
    (function() {{
        const definition = {encoded_code};
        const targetId = '{chart_id}';

        const draw = () => {{
            const target = document.getElementById(targetId);
            if (!target || !window.mermaid || !window.mermaid.mermaidAPI) {{
                return;
            }}
            window.mermaid.mermaidAPI.initialize({{
                startOnLoad: false,
                securityLevel: 'loose',
                theme: 'dark'
            }});
            window.mermaid.mermaidAPI.render(
                targetId + '-svg',
                definition,
                svgCode => {{
                    target.innerHTML = svgCode;
                }}
            );
        }};

        const ensureLoaded = () => {{
            if (window.mermaid && window.mermaid.mermaidAPI) {{
                draw();
            }} else {{
                setTimeout(ensureLoaded, 60);
            }}
        }};

        const existing = document.getElementById('mermaid-cdn');
        if (!existing) {{
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
            script.id = 'mermaid-cdn';
            script.onload = ensureLoaded;
            document.body.appendChild(script);
        }} else if (window.mermaid && window.mermaid.mermaidAPI) {{
            draw();
        }} else {{
            existing.addEventListener('load', ensureLoaded, {{ once: true }});
            ensureLoaded();
        }}
    }})();
    </script>
    """
    components.html(html, height=height, scrolling=True)


def _context_key(path: Path) -> str:
    return f"context::{path.resolve()}"


def _build_node_options(payload: Dict[str, object], node_map: Dict[int, str], md_sections: List[Tuple[str, str]]) -> Tuple[List[str], Dict[str, str]]:
    display_options: List[str] = []
    label_counts: Dict[str, int] = {}
    raw_nodes = payload.get("nodes", []) or []
    prepared: List[Tuple[str, Optional[int]]] = []
    for node in raw_nodes:
        node_id = node.get("node_id")
        try:
            numeric_id = int(node_id)
        except Exception:
            numeric_id = None
        if numeric_id is not None:
            label = resolve_label(numeric_id, node_map)
        else:
            label = str(node_id) if node_id is not None else "Unknown"
        prepared.append((label, numeric_id))
        label_counts[label] = label_counts.get(label, 0) + 1

    display_to_base: Dict[str, str] = {}
    for label, numeric_id in prepared:
        if label_counts.get(label, 0) > 1 and numeric_id is not None:
            option = f"{label} [id={numeric_id}]"
        else:
            option = label
        display_options.append(option)
        display_to_base[option] = label

    if not display_options:
        for title, _ in md_sections:
            base = title.split(" (", 1)[0].strip() or title
            if base not in display_to_base:
                display_options.append(base)
                display_to_base[base] = base

    return display_options, display_to_base


def _load_md_sections(report_path: Path) -> List[Tuple[str, str]]:
    if not report_path.exists():
        return []
    return extract_node_md_sections(report_path.read_text(encoding="utf-8"))


def get_window_context(
    selection: Path,
    mapping_path: Optional[Path],
    node_map: Dict[int, str],
) -> Dict[str, object]:
    cache_key = _context_key(selection)
    mapping_key = str(mapping_path.resolve()
                      ) if mapping_path and mapping_path.exists() else "<none>"
    mtime = selection.stat().st_mtime

    cached = st.session_state.get(cache_key)
    if cached and cached.get("mtime") == mtime and cached.get("mapping_key") == mapping_key:
        return cached

    payload = load_payload(selection)
    threshold = float(payload.get("threshold", 0.0))
    graph_rows = summarise_graph_edges(payload, node_map)
    node_rows = summarise_nodes(payload, node_map, threshold)
    gpt_sections = parse_gpt_sections(payload.get("gpt_summary"))
    graph_fig = plot_top_edges_bar(graph_rows, k=10) if graph_rows else None
    node_fig = plot_node_scores_bar(
        node_rows, threshold, k=10) if node_rows else None
    timeline_fig = plot_event_timeline(payload)
    sankey_fig = plot_sankey(graph_rows)
    # Pass mapping to heatmap if we update it, but for now let's just fix the call in main if needed.
    # Actually, I'll update plot_heatmap to take mapping in the next chunk or assume we pass it.
    # Wait, I didn't update plot_heatmap signature in the previous chunk to take mapping.
    # I should have. I'll fix it in the next tool call or just rely on IDs for now if I can't.
    # But user asked for subjects.
    # I will update plot_heatmap signature in a separate call if needed, or just do it here.
    # Let's assume I can't change it easily here without re-writing.
    # I'll rely on the fact that I can map it in the dataframe construction if I had the mapping.
    # I'll update the heatmap_fig call to pass mapping if I update the function.
    # Let's stick to the plan: I updated plot_heatmap but didn't change signature.
    # I will update it now to use mapping.

    heatmap_fig = plot_heatmap(payload, node_map)
    static_graph_fig = plot_static_network(graph_rows, node_rows)

    report_md = compute_report_md_path(payload, EXPLANATION_DIR)
    md_sections = _load_md_sections(report_md)
    md_index = {title.split(
        " (", 1)[0].strip() or title: content for title, content in md_sections}
    node_options, display_to_base = _build_node_options(
        payload, node_map, md_sections)

    context = {
        "payload": payload,
        "threshold": threshold,
        "graph_rows": graph_rows,
        "node_rows": node_rows,
        "gpt_sections": gpt_sections,
        "graph_fig": graph_fig,
        "node_fig": node_fig,
        "timeline_fig": timeline_fig,
        "sankey_fig": sankey_fig,
        "heatmap_fig": heatmap_fig,
        "static_graph_fig": static_graph_fig,
        "md_sections": md_sections,
        "md_index": md_index,
        "node_options": node_options,
        "display_to_base": display_to_base,
        "mtime": mtime,
        "mapping_key": mapping_key,
    }

    st.session_state[cache_key] = context
    return context


def main() -> None:
    st.set_page_config(page_title="PROVEX", layout="wide")
    st.title("PROVEX - SOC Dashboard")
    st.caption("PROVEX XAI Dashboard explaining DARPA TC dataset CADETS E3 host")

    mapping_default = Path(
        getattr(config, "NODE_MAPPING_JSON", EXPLANATION_DIR / "node_mapping.json"))
    mapping_path = Path(os.environ.get(
        "PROVEX_NODE_MAPPING_JSON", mapping_default))
    node_map = load_mapping(mapping_path if mapping_path.exists() else None)

    files = sorted(EXPLANATION_DIR.glob(DEFAULT_PATTERN))
    if not files:
        st.warning("No explanation JSON files found under artifact/explanations.")
        return

    default_file = os.environ.get("PROVEX_EXPLANATION_JSON")
    default_index = 0
    if default_file:
        default_path = Path(default_file).resolve()
        for idx, candidate in enumerate(files):
            if candidate.resolve() == default_path:
                default_index = idx
                break

    selection = st.sidebar.selectbox(
        "Explanation JSON", files, index=default_index, format_func=lambda p: p.name)

    try:
        context = get_window_context(
            selection, mapping_path if mapping_path.exists() else None, node_map)
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to prepare dashboard context: {exc}")
        return

    payload = context["payload"]
    window_name = payload.get("window_path", selection.name)
    events = payload.get("num_events", 0)
    threshold = context["threshold"]

    graph_rows = context["graph_rows"]
    node_rows = context["node_rows"]
    gpt_sections = context["gpt_sections"] or {}
    timeline_fig = context["timeline_fig"]
    heatmap_fig = context.get("heatmap_fig")
    static_graph_fig = context.get("static_graph_fig")
    graph_fig = context.get("graph_fig")
    node_fig = context.get("node_fig")

    # --- Sidebar Settings ---
    st.sidebar.markdown("### Settings")

    # =========================================================================
    # SECTION 1: HEADER & METRICS
    # =========================================================================
    st.markdown(f"### Window: `{window_name}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", f"{events:,}")
    col2.metric("Nodes Involved", len(node_rows))
    col3.metric("High-Loss Events", len(graph_rows))
    col4.metric("Threshold", f"{threshold:.4f}")

    st.divider()

    # =========================================================================
    # SECTION 2: EXECUTIVE SUMMARY & TEMPORAL ANALYSIS (Side by side)
    # =========================================================================
    # st.markdown("## Executive Summary & Timeline")
    
    col_summary, col_timeline = st.columns(2)
    
    with col_summary:
        st.markdown("### Summary")
        summary_text = gpt_sections.get("What happened?")
        if summary_text:
            st.markdown(summary_text)
        else:
            st.info("No summary available.")
    
    with col_timeline:
        st.markdown("### Temporal Analysis")
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("No timeline data available.")

    st.divider()

    # =========================================================================
    # SECTION 3: NETWORK GRAPH
    # =========================================================================
    st.markdown("## Network Graph")
    if static_graph_fig:
        st.plotly_chart(static_graph_fig, use_container_width=True)
    else:
        st.info("No graph data available.")

    st.divider()

    # =========================================================================
    # SECTION 4: KEY INSIGHTS (Who's Involved)
    # =========================================================================
    st.markdown("## Key Insights")
    st.markdown("### Who's Involved?")
    who_text = gpt_sections.get("Who's involved?", "N/A")
    st.markdown(who_text)

    st.divider()

    # =========================================================================
    # SECTION 5: ACTIVITY HEATMAP & WHY FLAGGED (Side by side)
    # =========================================================================
    st.markdown("## Activity Analysis")
    
    col_heatmap, col_why = st.columns(2)
    
    with col_heatmap:
        st.markdown("### Activity Heatmap")
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("No heatmap data available.")
    
    with col_why:
        st.markdown("### Why Flagged?")
        why_text = gpt_sections.get("Why flagged?", "N/A")
        st.markdown(why_text)

    st.divider()

    # =========================================================================
    # SECTION 5: DETAILED ANALYSIS (Tables & Charts)
    # =========================================================================
    st.markdown("## Detailed Analysis")

    # 5a. Top Edges & Top Nodes (side by side)
    col_edges, col_nodes = st.columns(2)
    with col_edges:
        st.markdown("### Top Edges")
        if graph_fig:
            st.plotly_chart(graph_fig, use_container_width=True)
        else:
            st.info("No edge data available.")

    with col_nodes:
        st.markdown("### Top Nodes")
        if node_fig:
            st.plotly_chart(node_fig, use_container_width=True)
        else:
            st.info("No node data available.")

    st.divider()

    # =========================================================================
    # SECTION 6: RISK ASSESSMENT & RECOMMENDATIONS (Writeups)
    # =========================================================================
    st.markdown("## Risk Assessment & Recommendations")

    col_risks, col_recs = st.columns(2)
    with col_risks:
        st.markdown("### Risks & Missing Context")
        risks_text = gpt_sections.get("What's missing or risky?", "N/A")
        st.markdown(risks_text)

    with col_recs:
        st.markdown("### Recommendations")
        recs_text = gpt_sections.get("What next?", "N/A")
        st.markdown(recs_text)


if __name__ == "__main__":
    main()
