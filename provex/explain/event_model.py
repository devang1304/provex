"""Recompute one event's logits from its capture, under gates on neighbour edges and history.

Items are absolute event indices. One gate per item applies everywhere the event appears:
as a neighbour edge in the event's 2-hop computation subgraph (via ``edge_deletion``), and
as a candidate memory message in the history window (via ``FunctionalTGN``). Setting a gate
to 0 therefore means "this past event never happened", with the batch partition held fixed.
After a deletion the neighbour loader is NOT backfilled with older neighbours; this is
documented as a limitation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..replay.capture import EventCapture
from .functional_tgn import FunctionalTGN
from .interventions import edge_deletion


class EventModel(nn.Module):
    def __init__(self, memory, gnn, link_pred, cap: EventCapture, *, use_history: bool = True,
                 dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        super().__init__()
        self.gnn, self.link_pred, self.cap, self.dtype = gnn, link_pred, cap, dtype
        dev = self.device = torch.device(device)
        hist_items = cap.hist.items() if (use_history and cap.hist is not None) else []
        hs = set(hist_items)
        items = sorted(set(cap.edge_ev.tolist()) | hs)
        self.items = torch.tensor(items, dtype=torch.long)
        pos = {ev: k for k, ev in enumerate(items)}
        self.item_pos = pos
        self.edge_item = torch.tensor([pos[int(e)] for e in cap.edge_ev.tolist()], dtype=torch.long, device=dev)
        self.is_history_item = torch.tensor([ev in hs for ev in items], dtype=torch.bool, device=dev)
        self.is_edge_item = torch.zeros(len(items), dtype=torch.bool, device=dev)
        self.is_edge_item[self.edge_item] = True
        if cap.hist is not None:
            t_ref = cap.hist.t_ref
        else:
            t_ref = int(cap.edge_t.min()) if cap.edge_t.numel() else cap.t
        self.t_ref = t_ref
        self.x = cap.x.to(dev, dtype)
        # subtract in int64 first: ns timestamps (~1.5e18) are not exact in float64, and the KAIROS
        self.edge_t_off = (cap.edge_t - t_ref).to(torch.float64).to(dev)
        self.edge_msg = cap.edge_msg.to(dev, dtype)
        self.ei = cap.edge_index.to(dev)
        self.y = torch.tensor([cap.y], device=dev)
        self.ftgn = FunctionalTGN(memory, cap.hist, pos, dtype, dev) if hist_items else None
        pos_g = {n: i for i, n in enumerate(cap.node_gid.tolist())}
        pairs = [(k, pos_g[n]) for k, n in enumerate(cap.hist.R) if n in pos_g] if self.ftgn is not None else []
        self.r_hist = [k for k, _ in pairs]      # rows of the functional replay ...
        self.r_local = [i for _, i in pairs]     # ... written into these subgraph nodes
    @property
    def num_items(self) -> int:
        return int(self.items.numel())

    def _edge_attr(self, lu_off: Tensor, ei: Tensor, t_off: Tensor, msg: Tensor) -> Tensor:
        rel = lu_off[ei[0]] - t_off
        if getattr(self.gnn, "rel_t_sign", 1) < 0:
            rel = -rel
        return torch.cat([self.gnn.time_enc(rel.to(self.dtype)), msg], dim=-1)

    def forward(self, g: Optional[Tensor] = None, *, hard: bool = False,
                history: bool = True, edge_g: Optional[Tensor] = None) -> Tensor:
        """Logits [num_edge_types] with item gates ``g`` (None = all kept).

        hard=True requires binary gates and physically removes edges (exact deletion).
        history=False ignores history gates and keeps the captured memory for u and v.
        edge_g, if given, gates the neighbour-edge channel instead of ``g``; this separates
        the two channels for items that appear in both.
        """
        x, lu = self.x, self.lu_off
        if self.ftgn is not None and history and g is not None:
            mem_r, lu_r = self.ftgn(g)
            x = x.clone(); lu = lu.clone()
            x[self.r_local] = mem_r[self.r_hist]
            lu[self.r_local] = lu_r[self.r_hist]
        ei, t_off, msg = self.ei, self.edge_t_off, self.edge_msg
        eg = edge_g if edge_g is not None else g
        m = None if eg is None else eg[self.edge_item]
        convs = [self.gnn.conv, self.gnn.conv2]
        if hard and m is not None:
            keep = m > 0.5
            ei, t_off, msg, m = ei[:, keep], t_off[keep], msg[keep], None
        edge_attr = self._edge_attr(lu, ei, t_off, msg)
        if m is None:
            z = torch.relu(self.gnn.conv(x, ei, edge_attr))
            z = torch.relu(self.gnn.conv2(z, ei, edge_attr))
        else:
            with edge_deletion(convs, mask=m):
                z = torch.relu(self.gnn.conv(x, ei, edge_attr))
                z = torch.relu(self.gnn.conv2(z, ei, edge_attr))
        s, d = self.cap.src_local, self.cap.dst_local
        return self.link_pred(z[[s]], z[[d]])[0]

    def loss(self, g: Optional[Tensor] = None, **kw) -> Tensor:
        return F.cross_entropy(self.forward(g, **kw).unsqueeze(0), self.y)