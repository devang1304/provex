"""Day graphs, edge-type labels and node names."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch_geometric.data import TemporalData

EDGE_TYPES = ["EVENT_WRITE", "EVENT_READ", "EVENT_CLOSE", "EVENT_OPEN",
              "EVENT_EXECUTE", "EVENT_SENDTO", "EVENT_RECVFROM"]


def load_day(path: str | Path) -> TemporalData:
    """Load a ``graph_4_X.TemporalData.simple`` file (trusted local artifact)."""
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not bool((data.t[1:] >= data.t[:-1]).all()):
        raise ValueError(f"{path}: events are not time-sorted")
    return data


def edge_type_labels(msg: torch.Tensor, node_embedding_dim: int = 16,
                     num_edge_types: int = len(EDGE_TYPES)) -> torch.Tensor:
    """Event-type label as upstream computes it: position of the 1 in the one-hot slice.

    Upstream uses ``tensor_find(m[16:-16], 1) - 1`` (first index equal to 1). For valid
    one-hot rows that is the argmax; ``check_one_hot`` asserts validity.
    """
    one_hot = msg[..., node_embedding_dim:node_embedding_dim + num_edge_types]
    return one_hot.argmax(-1)


def check_one_hot(msg: torch.Tensor, node_embedding_dim: int = 16,
                  num_edge_types: int = len(EDGE_TYPES)) -> None:
    one_hot = msg[..., node_embedding_dim:node_embedding_dim + num_edge_types]
    if not bool(((one_hot == 0) | (one_hot == 1)).all() and (one_hot.sum(-1) == 1).all()):
        raise ValueError("edge-type slice of msg is not one-hot")


def load_node_names(path: str | Path) -> Dict[int, str]:
    """``node_mapping.json``: global index -> "type: name" (e.g. "subject: nginx")."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def split_label(label: str) -> tuple[str, str]:
    kind, _, name = label.partition(": ")
    return kind, name
