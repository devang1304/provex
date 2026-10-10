"""Per-event captures: the exact computation subgraph and memory-update history of an event.

Everything reported downstream is keyed by *global* identifiers: ``node_gid`` for nodes and
``edge_ev`` (absolute event index in the day graph) for neighbour edges and history items.
Local indices never leave this module (fixes v1 defect C1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class Cand:
    """One candidate memory message for node ``node`` in one batch (PyG precedence order)."""
    ev: int                 # absolute event index, or -1 if unknown (stale entry from before the day)
    counterpart: int        # global id of the other endpoint
    t: int                  # event time (ns)
    raw: Tensor             # raw message [raw_msg_dim]
    current: bool           # True if the event is in this batch (False = stale store entry)
    counterpart_mem: Tensor  # counterpart memory *before* this batch's update [memory_dim]


@dataclass
class Step:
    batch: int
    node: int               # global id (member of the capture's R set)
    cands: List[Cand]
    pre_mem: Tensor         # node memory before the batch update (ground truth, for checks)
    pre_lu: int
    post_mem: Tensor        # node memory after the batch update (ground truth, for checks)
    post_lu: int


@dataclass
class HistoryContext:
    R: List[int]                    # global ids whose memory is replayed functionally ({u, v})
    b0: int                         # first batch of the history window
    mem0: Tensor                    # [|R|, memory_dim] memory of R at the start of batch b0
    lu0: Tensor                     # [|R|] last_update (int64 ns)
    steps: List[List[Step]]         # per touched batch, the steps for R nodes in that batch
    t_ref: int                      # time origin for float64 offsets

    def items(self) -> List[int]:
        out = set()
        for batch_steps in self.steps:
            for st in batch_steps:
                for c in st.cands:
                    if c.ev >= 0:
                        out.add(c.ev)
        return sorted(out)


@dataclass
class EventCapture:
    event: int
    batch: int
    pos: int
    src: int
    dst: int
    t: int
    y: int
    loss_batch: float
    logits_batch: Tensor            # [num_edge_types]
    node_gid: Tensor                # [S] global ids, local order
    x: Tensor                       # [S, memory_dim] pre-batch memory of subgraph nodes
    lu: Tensor                      # [S] int64 last_update
    edge_index: Tensor              # [2, E] local indices (neighbour -> centre)
    edge_ev: Tensor                 # [E] absolute event index of each neighbour edge
    edge_t: Tensor                  # [E] int64 event time
    edge_msg: Tensor                # [E, raw_msg_dim]
    layer_rel: Tensor               # [2, E] bool: relevant to conv1 / conv2
    src_local: int
    dst_local: int
    tau: Optional[float] = None     # alert threshold of the event's window (set by alerts)
    hist: Optional[HistoryContext] = None
    meta: Dict = field(default_factory=dict)


def extract_subgraph(edge_index_b: Tensor, a_u: int, a_v: int):
    """Exact receptive field of a 2-layer conv for targets ``a_u``, ``a_v`` (batch-local ids).

    Edges point neighbour -> centre (``LastNeighborLoader`` convention). conv2 at u/v reads
    conv1 outputs of in-neighbours N1 and of u/v themselves; conv1 at any node reads that
    node's in-edges. So the needed edges are exactly the in-edges of T = {u, v} ∪ N1, and the
    needed nodes are T plus the sources of those edges.

    Returns (nodes_b, keep_edge_mask, conv2_mask) with ``nodes_b`` sorted batch-local ids.
    """
    src, tgt = edge_index_b
    centre = torch.tensor([a_u, a_v], dtype=src.dtype, device=src.device).unique()
    into_centre = torch.isin(tgt, centre)
    n1 = src[into_centre].unique()
    targets = torch.cat([centre, n1]).unique()
    keep = torch.isin(tgt, targets)
    nodes = torch.cat([targets, src[keep]]).unique()
    return nodes, keep, into_centre[keep]


def build_capture(*, event: int, batch: int, pos: int, src_g: int, dst_g: int, t: int, y: int,
                  loss: float, logits: Tensor, n_id: Tensor, edge_index_b: Tensor, e_id: Tensor,
                  x_b: Tensor, lu_b: Tensor, a_u: int, a_v: int, data) -> EventCapture:
    nodes, keep, conv2_rel = extract_subgraph(edge_index_b, a_u, a_v)
    relabel = torch.full((n_id.numel(),), -1, dtype=torch.long, device=nodes.device)
    relabel[nodes] = torch.arange(nodes.numel(), device=nodes.device)
    ei = relabel[edge_index_b[:, keep]]
    ev = e_id[keep].cpu()
    layer_rel = torch.stack([torch.ones_like(conv2_rel), conv2_rel]).cpu()
    return EventCapture(
        event=event, batch=batch, pos=pos, src=src_g, dst=dst_g, t=t, y=y, loss_batch=loss,
        logits_batch=logits.detach().cpu().clone(),
        node_gid=n_id[nodes].cpu().clone(), x=x_b[nodes].detach().cpu().clone(),
        lu=lu_b[nodes].cpu().clone(), edge_index=ei.cpu().clone(), edge_ev=ev.clone(),
        edge_t=data.t[ev].clone(), edge_msg=data.msg[ev].clone(), layer_rel=layer_rel,
        src_local=int(relabel[a_u]), dst_local=int(relabel[a_v]),
    )
