"""Ground truth for history gating: full PyG replay with some events deleted.

The batch partition stays fixed (as in FunctionalTGN): deleted events are dropped from their
batch, and every other event keeps its batch. LastNeighborLoader e_ids are insertion
counters, so a map from counter to absolute event index restores message and time lookups.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import LastNeighborLoader

from ..data import edge_type_labels


@torch.no_grad()
def full_replay(memory, gnn, link_pred, hp, data, deleted: Iterable[int], event: int,
                batch_size: int = 1024) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Returns (loss of ``event``, memory[[src, dst]] before its batch, last_update[[src, dst]])."""
    deleted = set(int(d) for d in deleted)
    memory.eval(); gnn.eval(); link_pred.eval()
    memory.reset_state()
    loader = LastNeighborLoader(memory.num_nodes, size=hp.neighbor_size)
    assoc = torch.empty(memory.num_nodes, dtype=torch.long)
    counter_to_ev = []
    b_event = event // batch_size
    for b in range(0, b_event * batch_size + 1, batch_size):
        e = min(b + batch_size, data.num_events)
        keep = torch.tensor([i for i in range(b, e) if i not in deleted or i == event], dtype=torch.long)
        src, dst, t, msg = data.src[keep], data.dst[keep], data.t[keep], data.msg[keep]
        if b // batch_size == b_event:
            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0))
            x, lu = memory(n_id)
            evs = torch.tensor([counter_to_ev[c] for c in e_id.tolist()], dtype=torch.long)
            z = gnn(x, lu, edge_index, data.t[evs], data.msg[evs])
            pos = int((keep == event).nonzero()[0])
            logits = link_pred(z[assoc[src[pos:pos + 1]]], z[assoc[dst[pos:pos + 1]]])
            y = edge_type_labels(msg[pos:pos + 1], hp.node_embedding_dim, hp.num_edge_types)
            r = torch.tensor([int(data.src[event]), int(data.dst[event])])
            return float(F.cross_entropy(logits, y)), memory.memory[r].clone(), memory.last_update[r].clone()
        if keep.numel():
            memory.update_state(src, dst, t, msg)
            loader.insert(src, dst)
            counter_to_ev.extend(keep.tolist())
    raise RuntimeError("event batch not reached")