"""GraphMask for TransformerConv (fixes v1 defect C3).

Following Schlichtkrull et al. (ICLR 2021), adapted to TransformerConv:
* amortised gate networks, one per layer, read the unmasked forward pass
  (query_i, key_j + e, value_j + e) of each edge and output a hard-concrete log-alpha;
* a gated-off edge is replaced by a learned baseline: score b_s in R^H and value b_v in
  R^{H x C}. The softmax is recomputed with the replaced scores, so nothing leaks through
  the attention denominator;
* the objective is to minimise the expected L0 (number of open gates) subject to
  |L_masked - L_full| / max(L_full, 1) <= allowance, with a persistent Lagrange multiplier;
* gates are trained once on alerts from validation days, then applied per event in one
  forward pass.

GraphMask answers "which neighbour edges does the model *need*" (introspection). It does not
cover the memory channel. ``explain`` reports per-item keep probability.

PyG's own GraphMaskExplainer cannot be used here. TransformerConv calls propagate with
query/key/value instead of x, so the x_i/x_j that PyG's gate hook expects are empty. Its
[E, 1] message scale also mis-broadcasts against [E, H, C] messages.
"""
from __future__ import annotations

import math
import time
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.utils import softmax

from .event_model import EventModel
from .joint_mask import Explanation, expected_l0, hard_concrete_mean, hard_concrete_sample


class _Recorder:
    def __init__(self):
        self.rec = {}

    def hook(self, layer: int):
        def h(module, inputs, out):
            kw = inputs[0]
            e = module.lin_edge(kw["edge_attr"]).view(-1, module.heads, module.out_channels)
            self.rec[layer] = {"q": kw["query_i"].detach(), "k": (kw["key_j"] + e).detach(),
                               "v": (kw["value_j"] + e).detach()}
        return h


class GateNet(nn.Module):
    def __init__(self, heads: int, channels: int, hidden: int = 64):
        super().__init__()
        d = heads * channels
        self.proj = nn.ModuleList([nn.Linear(d, hidden) for _ in range(3)])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, 3.0)        # start with gates open

    def forward(self, q, k, v) -> Tensor:
        h = sum(p(x.flatten(1).to(p.weight.dtype)) for p, x in zip(self.proj, (q, k, v)))
        return self.out(F.relu(self.norm(h))).squeeze(-1)


class GraphMaskTC(nn.Module):
    def __init__(self, gnn, hidden: int = 64, allowance: float = 0.05):
        super().__init__()
        self.convs = [gnn.conv, gnn.conv2]
        self.gates = nn.ModuleList([GateNet(c.heads, c.out_channels, hidden) for c in self.convs])
        self.b_s = nn.ParameterList([nn.Parameter(torch.zeros(c.heads)) for c in self.convs])
        self.b_v = nn.ParameterList([nn.Parameter(torch.zeros(c.heads, c.out_channels)) for c in self.convs])
        self.allowance = allowance
        self.lam = 1.0

    # -- forward helpers ---------------------------------------------------------------
    def _run(self, em: EventModel, hooks) -> Tensor:
        handles = [c.register_message_forward_hook(h) for c, h in zip(self.convs, hooks) if h is not None]
        try:
            ea = em._edge_attr(em.lu_off, em.ei, em.edge_t_off, em.edge_msg)
            z = torch.relu(em.gnn.conv(em.x, em.ei, ea))
            z = torch.relu(em.gnn.conv2(z, em.ei, ea))
            logits = em.link_pred(z[[em.cap.src_local]], z[[em.cap.dst_local]])
        finally:
            for h in handles:
                h.remove()
        return F.cross_entropy(logits, em.y)

    def record(self, em: EventModel):
        r = _Recorder()
        with torch.no_grad():
            self._run(em, [r.hook(0), r.hook(1)])
        return r.rec

    def _replace_hook(self, layer: int, z: Tensor):
        conv = self.convs[layer]

        def h(module, inputs, out):
            kw = inputs[0]
            e = module.lin_edge(kw["edge_attr"]).view(-1, module.heads, module.out_channels)
            k, v = kw["key_j"] + e, kw["value_j"] + e
            s = (kw["query_i"] * k).sum(-1) / math.sqrt(module.out_channels)
            zz = z.to(s.dtype).unsqueeze(-1)
            s = zz * s + (1 - zz) * self.b_s[layer].to(s.dtype)
            alpha = softmax(s, kw["index"], kw["ptr"], kw["size_i"])
            val = zz.unsqueeze(-1) * v + (1 - zz.unsqueeze(-1)) * self.b_v[layer].to(v.dtype)
            return val * alpha.unsqueeze(-1)
        return h

    def log_alpha(self, rec) -> List[Tensor]:
        return [self.gates[l](rec[l]["q"], rec[l]["k"], rec[l]["v"]) for l in range(len(self.convs))]

    # -- training -------------------------------------------------------------------------
    def fit(self, models: Sequence[EventModel], epochs: int = 20, lr: float = 1e-3, dual_lr: float = 0.5,
            seed: int = 0, log=print) -> List[dict]:
        torch.manual_seed(seed)
        recs = [self.record(em) for em in models]
        L_full = [float(self._run(em, [None, None]).detach()) for em in models]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        hist = []
        for ep in range(epochs):
            tot_l0, tot_dev = 0.0, 0.0
            for em, rec, Lf in zip(models, recs, L_full):
                opt.zero_grad()
                la = self.log_alpha(rec)
                zs = [hard_concrete_sample(a) for a in la]
                L = self._run(em, [self._replace_hook(l, z) for l, z in enumerate(zs)])
                dev = (L - Lf).abs() / max(Lf, 1.0)
                l0 = sum(expected_l0(a).mean() for a in la) / len(la)
                (l0 + self.lam * F.softplus(dev - self.allowance)).backward()
                opt.step()
                self.lam = max(0.0, self.lam + dual_lr * (float(dev) - self.allowance))
                tot_l0 += float(l0); tot_dev += float(dev)
            row = {"epoch": ep, "mean_open_frac": tot_l0 / len(models), "mean_rel_dev": tot_dev / len(models),
                   "lambda": self.lam}
            hist.append(row); log(row)
        return hist

    # -- explanation ----------------------------------------------------------------------
    @torch.no_grad()
    def explain(self, em: EventModel, tau: float) -> Explanation:
        t0 = time.perf_counter()
        rec = self.record(em)
        gates = [hard_concrete_mean(a) for a in self.log_alpha(rec)]
        edge_keep = torch.stack(gates).max(0).values.double()       # needed in any layer
        L_masked = float(self._run(em, [self._replace_hook(l, (g > 0.5).double()) for l, g in enumerate(gates)]))
        scores = torch.full((em.num_items,), -1.0, dtype=torch.float64, device=em.device)
        scores.scatter_reduce_(0, em.edge_item, edge_keep, reduce="amax", include_self=True)
        L_full = float(em.loss(torch.ones(em.num_items, dtype=em.dtype, device=em.device), hard=True))
        return Explanation(method="graphmask", mode="introspection", items=em.items.tolist(),
                           scores=scores.tolist(), selected=em.items[(scores > 0.5).cpu()].tolist(),
                           valid=abs(L_masked - L_full) / max(L_full, 1.0) <= self.allowance, repaired=0,
                           L_full=L_full, L_selected=L_masked, tau=tau, runtime_s=time.perf_counter() - t0,
                           extra={"per_layer_open": [float((g > 0.5).double().mean()) for g in gates]})