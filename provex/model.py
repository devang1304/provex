"""KAIROS model definitions and checkpoint loading.

The classes mirror upstream KAIROS (DARPA/CADETS_E3/model.py) parameter-for-parameter,
so upstream state dicts load unchanged. Two differences from upstream are deliberate:

* no module-level device/global state: everything runs on the tensors' device;
* ``rel_t_sign`` selects the time-delta convention. Upstream uses
  ``last_update[src] - t`` (+1). The v1 PROVEX fork used ``t - last_update[src]`` (-1);
  it only matters for loading the v1 checkpoint.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator


@dataclass
class KairosHParams:
    num_nodes: int
    raw_msg_dim: int = 39
    memory_dim: int = 100
    time_dim: int = 100
    embedding_dim: int = 100
    neighbor_size: int = 20
    num_edge_types: int = 7
    node_embedding_dim: int = 16
    rel_t_sign: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: nn.Module,
                 rel_t_sign: int = 1):
        super().__init__()
        self.time_enc = time_enc
        self.rel_t_sign = int(rel_t_sign)
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8, dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(out_channels * 8, out_channels, heads=1, concat=False,
                                     dropout=0.0, edge_dim=edge_dim)

    def edge_attr(self, last_update: torch.Tensor, edge_index: torch.Tensor, t: torch.Tensor,
                  msg: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        rel_t = last_update[edge_index[0]] - t
        if self.rel_t_sign < 0:
            rel_t = -rel_t
        return torch.cat([self.time_enc(rel_t.to(dtype)), msg], dim=-1)

    def forward(self, x, last_update, edge_index, t, msg):
        edge_attr = self.edge_attr(last_update, edge_index, t, msg, x.dtype)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)
        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8), nn.Dropout(dropout), nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2), nn.Dropout(dropout), nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)), nn.Dropout(dropout), nn.Tanh(),
            Linear(int(in_channels // 2), out_channels),
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)
        return self.lin_seq(h)


def build(hp: KairosHParams) -> Tuple[TGNMemory, GraphAttentionEmbedding, LinkPredictor]:
    memory = TGNMemory(
        hp.num_nodes, hp.raw_msg_dim, hp.memory_dim, hp.time_dim,
        message_module=IdentityMessage(hp.raw_msg_dim, hp.memory_dim, hp.time_dim),
        aggregator_module=LastAggregator(),
    )
    gnn = GraphAttentionEmbedding(hp.memory_dim, hp.embedding_dim, hp.raw_msg_dim, memory.time_enc,
                                  rel_t_sign=hp.rel_t_sign)
    link_pred = LinkPredictor(hp.embedding_dim, hp.num_edge_types)
    return memory, gnn, link_pred


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu", *, legacy_rel_t_sign: int = -1):
    """Load a KAIROS checkpoint and return ``(memory, gnn, link_pred, hparams)`` in eval mode.

    Supported formats:
      * ``models_state.pt`` written by the patched upstream ``train.py``:
        ``{"hparams": {...}, "memory": sd, "gnn": sd, "link_pred": sd, ...}``
      * the v1 PROVEX pickle ``[memory, gnn, link_pred, neighbor_loader]`` (trusted, legacy).
    """
    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)  # trusted local artifact
    if isinstance(obj, dict) and "hparams" in obj:
        hp = KairosHParams(**{k: v for k, v in obj["hparams"].items() if k in KairosHParams.__dataclass_fields__})
        memory, gnn, link_pred = build(hp)
        memory.load_state_dict(obj["memory"])
        gnn.load_state_dict(obj["gnn"])
        link_pred.load_state_dict(obj["link_pred"])
    elif isinstance(obj, (list, tuple)) and len(obj) >= 3:
        raise ValueError(
            "Legacy pickled checkpoint: load it with load_legacy_checkpoint(), which needs the v1 "
            "model module importable as 'model'.")
    else:
        raise ValueError(f"Unrecognised checkpoint format: {path}")
    return _finish(memory, gnn, link_pred, hp, device)


def load_legacy_checkpoint(path: str | Path, legacy_module, device="cpu", rel_t_sign: int = -1):
    """Load the v1 fork pickle by aliasing ``legacy_module`` as ``model`` and copying weights."""
    sys.modules["model"] = legacy_module
    old_memory, old_gnn, old_lp = torch.load(path, map_location="cpu", weights_only=False)[:3]
    hp = KairosHParams(
        num_nodes=int(old_memory.num_nodes), raw_msg_dim=int(old_memory.raw_msg_dim),
        memory_dim=int(old_memory.memory_dim), time_dim=int(old_memory.time_dim),
        embedding_dim=int(old_gnn.conv2.out_channels), num_edge_types=int(old_lp.lin_seq[-1].out_features),
        neighbor_size=64, rel_t_sign=rel_t_sign,
    )
    memory, gnn, link_pred = build(hp)
    memory.load_state_dict(old_memory.state_dict(), strict=False)
    gnn.load_state_dict(old_gnn.state_dict())
    link_pred.load_state_dict(old_lp.state_dict())
    return _finish(memory, gnn, link_pred, hp, device)


def _finish(memory, gnn, link_pred, hp, device):
    memory, gnn, link_pred = memory.to(device), gnn.to(device), link_pred.to(device)
    memory.eval(); gnn.eval(); link_pred.eval()
    for m in (memory, gnn, link_pred):
        for p in m.parameters():
            p.requires_grad_(False)
    return memory, gnn, link_pred, hp
