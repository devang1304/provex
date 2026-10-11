"""GraphMask for TransformerConv (fixes v1 defect C3).

Following Schlichtkrull et al. (ICLR 2021), adapted to TransformerConv:
* amortised gate networks, one per layer, read the unmasked forward pass
  (query_i, key_j + e, value_j + e) of each edge plus the edge's role for the explained event
  (does its target / source touch an endpoint?) and output a hard-concrete log-alpha. Without the
  role an amortised gate cannot tell which edges reach a link prediction (only layer-2 edges into
  the two endpoints do), and training degenerates to one switch per layer;
* a gated-off edge is replaced by a learned baseline: score b_s in R^H and value b_v in
  R^{H x C}. The softmax is recomputed with the replaced scores, so nothing leaks through
  the attention denominator;
* the objective is to minimise the expected L0 (number of open gates) subject to
  |L_masked - L_full| / max(L_full, 1) <= allowance, with a persistent Lagrange multiplier;
* training starts on the feasible side (initial multiplier 10, slow dual step 0.05) and log-alpha
  is bounded by a leaky clamp. With multiplier 1 the gates collapse shut in the first epoch
  (closing every edge costs KAIROS only ~9% of its loss), and a fully closed hard-concrete gate
  gets no gradient, so it never reopens; a tanh bound has the same problem at saturation
  (scripts/diag_graphmask.py). The constraint holds on average over events, as in GraphMask;
  ``evaluate`` reports how often it holds per event. Even so, the gates can collapse later in
  training, so ``fit`` keeps the sparsest epoch whose held-out deviation meets the constraint;
* gates are trained once on alerts from validation days, then applied per event in one
  forward pass.

GraphMask answers "which neighbour edges does the model *need*" (introspection). It does not
cover the memory channel. ``explain`` reports per-item keep probability.

PyG's own GraphMaskExplainer cannot be used here. TransformerConv calls propagate with
query/key/value instead of x, so the x_i/x_j that PyG's gate hook expects are empty. Its
[E, 1] message scale also mis-broadcasts against [E, H, C] messages.
"""
from __future__ import annotations

import copy
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
    def __init__(self, heads: int, channels: int, hidden: int = 64, init_bias: float = 3.0,
                 bounds: tuple = (-4.0, 5.0), n_role: int = 2):
        super().__init__()
        d = heads * channels
        self.proj = nn.ModuleList([nn.Linear(d, hidden) for _ in range(3)])
        self.role = nn.Linear(n_role, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, init_bias)  # start with gates open
        self.lo, self.hi = bounds

    def forward(self, q, k, v, role) -> Tensor:
        h = sum(p(x.flatten(1).to(p.weight.dtype)) for p, x in zip(self.proj, (q, k, v)))
        h = h + self.role(role.to(self.role.weight.dtype))
        raw = self.out(F.relu(self.norm(h))).squeeze(-1)
        # Leaky clamp to [lo, hi] (slope 0.01 outside). At log-alpha -4 a gate still samples open ~8% of
        # the time and keeps a gradient, so it can reopen. A plain clamp or tanh has no gradient at the
        # bound, and with unbounded log-alpha the gates collapse shut and stay shut.
        c = raw.clamp(self.lo, self.hi)
        return c + 0.01 * (raw - c)


class GraphMaskTC(nn.Module):
    def __init__(self, gnn, hidden: int = 64, allowance: float = 0.05, lam0: float = 10.0,
                 init_bias: float = 3.0, bounds: tuple = (-4.0, 5.0)):
        super().__init__()
        self.convs = [gnn.conv, gnn.conv2]
        self.gates = nn.ModuleList([GateNet(c.heads, c.out_channels, hidden, init_bias, bounds)
                                    for c in self.convs])
        self.b_s = nn.ParameterList([nn.Parameter(torch.zeros(c.heads)) for c in self.convs])
        self.b_v = nn.ParameterList([nn.Parameter(torch.zeros(c.heads, c.out_channels)) for c in self.convs])
        self.allowance = allowance
        self.lam = lam0

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
        # edge role for this event; messages arrive in em.ei column order (TransformerConv adds no loops)
        ends = torch.tensor([em.cap.src_local, em.cap.dst_local], device=em.ei.device)
        role = torch.stack([torch.isin(em.ei[1], ends), torch.isin(em.ei[0], ends)], 1).float()
        for l in r.rec:
            r.rec[l]["role"] = role
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
        return [self.gates[l](rec[l]["q"], rec[l]["k"], rec[l]["v"], rec[l]["role"]) for l in range(len(self.convs))]

    # -- training -------------------------------------------------------------------------
    def fit(self, models: Sequence[EventModel], epochs: int = 40, lr: float = 1e-3, dual_lr: float = 0.05,
            seed: int = 0, log=print, holdout_every: int = 5) -> List[dict]:
        """Train on all but every ``holdout_every``-th event. After each epoch the deterministic gates are
        evaluated on the held-out events, and the sparsest state whose mean deviation is within the
        allowance (the GraphMask constraint) is kept. Late in training the gates can collapse shut and not
        recover even as the multiplier grows, so the last epoch is not used blindly; the all-open start
        is the fallback, since it is always feasible."""
        torch.manual_seed(seed)
        recs = [self.record(em) for em in models]
        has_edges = [any(r[l]["q"].shape[0] for l in r) for r in recs]   # nothing to gate otherwise
        models = [m for m, h in zip(models, has_edges) if h]
        recs = [r for r, h in zip(recs, has_edges) if h]
        if not models:
            return [{"final": True, "n_events_trained": 0, "n_events_without_edges": len(has_edges)}]
        L_full = [float(self._run(em, [None, None]).detach()) for em in models]
        hold = list(range(0, len(models), holdout_every)) if holdout_every and len(models) >= 2 * holdout_every else []
        train = [i for i in range(len(models)) if i not in set(hold)]
        sel = hold or train

        def evaluate_sel():
            return self.evaluate([models[i] for i in sel], [recs[i] for i in sel], [L_full[i] for i in sel])

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        best = {"epoch": -1, **evaluate_sel(), "state": copy.deepcopy(self.state_dict())}
        hist = []
        for ep in range(epochs):
            tot_l0, tot_dev = 0.0, 0.0
            for i in train:
                em, rec, Lf = models[i], recs[i], L_full[i]
                opt.zero_grad()
                la = self.log_alpha(rec)
                zs = [hard_concrete_sample(a) for a in la]
                L = self._run(em, [self._replace_hook(l, z) for l, z in enumerate(zs)])
                dev = (L - Lf).abs() / max(Lf, 1.0)
                ne = [a for a in la if a.numel()]                          # a layer can have no edges
                l0 = sum(expected_l0(a).mean() for a in ne) / len(ne)
                (l0 + self.lam * F.softplus(dev - self.allowance)).backward()
                opt.step()
                self.lam = max(0.0, self.lam + dual_lr * (float(dev) - self.allowance))
                tot_l0 += float(l0); tot_dev += float(dev)
            e = evaluate_sel()
            row = {"epoch": ep, "mean_open_frac": tot_l0 / len(train), "mean_rel_dev": tot_dev / len(train),
                   "lambda": self.lam, "heldout_dev_hard": e["dev_hard_mean"], "heldout_open_hard": e["open_frac_hard"]}
            hist.append(row); log(row)
            if e["dev_hard_mean"] <= self.allowance and e["open_frac_hard"] < best["open_frac_hard"]:
                best = {"epoch": ep, **e, "state": copy.deepcopy(self.state_dict())}
        self.load_state_dict(best.pop("state"))
        final = {"final": True, "selected_epoch": best.pop("epoch"), "n_events_trained": len(train),
                 "n_events_heldout": len(hold), "n_events_without_edges": has_edges.count(False), **best}
        hist.append(final); log(final)
        return hist
    @torch.no_grad()
    def evaluate(self, models, recs, L_full) -> dict:
        """Deviation and open fraction with the deterministic gates that ``explain`` uses, and the
        deviation with every edge replaced by the learned baselines (how much the edges matter)."""
        devs, closed, opens = [], [], []
        by_kind = {"l1": [], "l2_into_ends": [], "l2_other": []}
        for em, rec, Lf in zip(models, recs, L_full):
            gates = [(hard_concrete_mean(a) > 0.5).double() for a in self.log_alpha(rec)]
            L = float(self._run(em, [self._replace_hook(l, g) for l, g in enumerate(gates)]))
            L0 = float(self._run(em, [self._replace_hook(l, torch.zeros_like(g)) for l, g in enumerate(gates)]))
            devs.append(abs(L - Lf) / max(Lf, 1.0)); closed.append(abs(L0 - Lf) / max(Lf, 1.0))
            opens += [float(g.mean()) for g in gates if g.numel()]
            into = rec[1]["role"][:, 0] > 0
            by_kind["l1"] += gates[0].tolist()
            by_kind["l2_into_ends"] += gates[1][into].tolist(); by_kind["l2_other"] += gates[1][~into].tolist()
        if not devs:
            return {}
        devs.sort()
        return {"dev_hard_mean": sum(devs) / len(devs), "dev_hard_median": devs[len(devs) // 2],
                "frac_valid_hard": sum(d <= self.allowance for d in devs) / len(devs),
                "open_frac_hard": sum(opens) / len(opens), "dev_all_closed_mean": sum(closed) / len(closed),
                **{f"open_frac_{k}": (sum(v) / len(v) if v else None) for k, v in by_kind.items()},
                "frac_valid_all_closed": sum(d <= self.allowance for d in closed) / len(closed)}

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