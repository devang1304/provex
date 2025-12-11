"""Export nodeid→message mapping to JSON using DB credentials from config.py.

Also exposes ensure_node_mapping() so the explanations pipeline can create the
mapping on first run without failing if the file is already present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import psycopg2

try:
    from ..config import DATABASE, HOST, NODE_MAPPING_JSON, PASSWORD, PORT, USER
except ImportError:  # pragma: no cover
    from config import DATABASE, HOST, NODE_MAPPING_JSON, PASSWORD, PORT, USER  # type: ignore

DEFAULT_OUTPUT = Path(NODE_MAPPING_JSON)


def export_node_mapping(output_path: Optional[Path] = None) -> int:
    """Export node id→label mapping to JSON. Returns number of entries."""
    path = Path(output_path) if output_path else DEFAULT_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    with psycopg2.connect(
        dbname=DATABASE,
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
    ) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT index_id, node_type, msg FROM node2id ORDER BY index_id")
        rows = cur.fetchall()

    mapping = {int(index_id): f"{node_type}: {msg}" for index_id,
               node_type, msg in rows}
    path.write_text(json.dumps(mapping, ensure_ascii=False,
                    indent=2), encoding="utf-8")
    return len(mapping)


def ensure_node_mapping(path: Optional[Path] = None) -> bool:
    """Ensure mapping file exists; create it from DB if missing."""
    target = Path(path) if path else DEFAULT_OUTPUT
    if target.exists():
        return True
    try:
        n = export_node_mapping(target)
        print(
            f"[success] Exported {n} node mapping entries to {target.resolve()}")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Could not export node mapping: {exc}")
        return False


def main() -> None:
    count = export_node_mapping()
    print(f"Exported {count} entries to {DEFAULT_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
