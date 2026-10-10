"""Node-level ground truth for the four CADETS E3 attacks, mapped to KAIROS node ids.

Produced by the vendored ``kairos/cadets_e3/node_gt.py`` from the ORTHRUS/PIDSMaker CSVs.
By default it maps by name: KAIROS keys nodes by the hash of their name, so this is how
KAIROS itself identifies them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set

ATTACK_DAY = {"06": "4_6", "11": "4_11", "12": "4_12", "13": "4_13"}


def load_gt_nodes(gt_dir: str | Path) -> Dict[str, Set[int]]:
    """day -> set of malicious KAIROS node ids (union over attacks on that day)."""
    out: Dict[str, Set[int]] = {}
    for p in sorted(Path(gt_dir).glob("kairos_nodes_*.json")):
        attack = p.stem.split("_")[-1][-2:]
        day = ATTACK_DAY.get(attack)
        if day is None:
            continue
        nodes = json.loads(p.read_text(encoding="utf-8"))
        ids = {int(k) for k in (nodes.keys() if isinstance(nodes, dict) else nodes)}
        out.setdefault(day, set()).update(ids)
    return out