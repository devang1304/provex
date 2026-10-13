"""Edge deletion inside TransformerConv, exact for binary masks and smooth for soft ones.

PyG's built-in explain hook scales messages *after* the attention softmax, so a mask of 0
does not renormalise attention over the remaining edges: it is not deletion. Here a
message-forward hook rescales each kept message by the attention mass that survives at its
target node:

    out'_e = out_e * m_e / sum_{e' -> i} alpha_e' m_e'      (e -> i)
           * s_i,  s_i = 1 - prod_{e' -> i} (1 - m_e')

With a binary mask this equals running the layer on the graph without the deleted edges.
That includes nodes left with no in-edges, which receive no aggregated message. With
m == 1 everywhere it is the identity, up to rounding in the attention sum.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import scatter


def _make_hook(mask: Tensor):
    def hook(module: TransformerConv, inputs, out: Tensor):
        kw = inputs[0]
        index, size_i = kw["index"], kw["size_i"]
        alpha = module._alpha                                    # [E, H] post-softmax
        m = mask.to(out.dtype)
        n = int(size_i) if size_i is not None else int(index.max()) + 1
        denom = scatter(alpha * m.unsqueeze(-1), index, dim=0, dim_size=n, reduce="sum")   # [N, H]
        keep_any = 1.0 - scatter(1.0 - m, index, dim=0, dim_size=n, reduce="mul")        # [N]
        factor = m.unsqueeze(-1) / denom[index].clamp_min(1e-30)                           # [E, H]
        return out * factor.unsqueeze(-1) * keep_any[index].view(-1, 1, 1)
    return hook


@contextmanager
def edge_deletion(convs: Iterable[TransformerConv], mask: Optional[Tensor] = None,
                  per_layer: Optional[Dict[int, Tensor]] = None):
    """Install deletion hooks on ``convs``.

    Pass either one ``mask`` [E], shared by every layer, or ``per_layer``, which maps a layer
    position to its own mask. Hooks are removed on exit.
    """
    handles = []
    try:
        for li, conv in enumerate(convs):
            m = per_layer.get(li) if per_layer is not None else mask
            if m is not None:
                handles.append(conv.register_message_forward_hook(_make_hook(m)))
        yield
    finally:
        for h in handles:
            h.remove()