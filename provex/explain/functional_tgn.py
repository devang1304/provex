"""Differentiable replay of TGNMemory's eval-mode update path for a small node set R.

This is what makes the explanation *temporal*: deleting (gating) a past event changes the
memory of u and v at the explained event's batch, exactly as PyG would have.

PyG eval-mode semantics reproduced, for node i in a batch:
* Candidate messages are i's entries in the s-store, then in the d-store, each in store
  order. A store holds the current batch's events if i appeared in that role in this batch,
  and stale entries from an earlier batch otherwise.
* Each message is [mem_i, mem_c, raw, time_enc(t - last_update_i)], with mem_c the
  counterpart's memory before this batch's update.
* LastAggregator keeps the message with the largest t. On CPU, ties go to the first in
  precedence order.
* memory_i <- GRU(winner, memory_i), and last_update_i <- the winner's t.
* If none of i's *current* events survive, i is not in the batch at all: memory and
  last_update stay unchanged, even when stale entries exist.

Gates g in [0, 1] per historical event relax this:
    order candidates by (t desc, precedence)
    w_k = g_k * prod_{j<k} (1 - g_j)          (survival-weighted "last")
    P   = 1 - prod_{current k} (1 - g_k)      (node present in the batch)
    mem' = P * sum_k w_k H_k / sum_k w_k + (1 - P) * mem
With binary gates this is exactly the PyG update, including promotion of the next-latest
message when the latest is deleted. Counterparts outside R stay frozen at their recorded
pre-batch memories (first-order scope). That is exact unless a deleted event would also have
changed a counterpart that later feeds R; ``provex.evaluate.local_vs_full`` measures this.

Times are carried as float64 offsets from ``t_ref`` so integer differences stay exact.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..replay.capture import HistoryContext


class FunctionalTGN:
    def __init__(self, memory, hist: HistoryContext, item_pos: Dict[int, int], dtype: torch.dtype,
                 device: torch.device | str = "cpu"):
        self.gru, self.time_enc = memory.gru, memory.time_enc
        self.hist, self.dtype, self.device = hist, dtype, torch.device(device)
        dev = self.device
        self.R = list(hist.R)
        self.r_pos = {n: k for k, n in enumerate(self.R)}
        t_ref = hist.t_ref
        self.mem0 = hist.mem0.to(dev, dtype)
        self.lu0 = (hist.lu0.to(torch.float64) - t_ref).to(dev)
        # Precompute per step: ordered candidate tensors.
        self.plan: List[List[dict]] = []
        for batch_steps in hist.steps:
            plan_b = []
            for st in batch_steps:
                order = sorted(range(len(st.cands)), key=lambda k: (-st.cands[k].t, k))
                cands = [st.cands[k] for k in order]
                plan_b.append({
                    "node": st.node,
                    "t_off": torch.tensor([c.t - t_ref for c in cands], dtype=torch.float64, device=dev),
                    "raw": torch.stack([c.raw for c in cands]).to(dev, dtype),
                    "cp_frozen": torch.stack([c.counterpart_mem for c in cands]).to(dev, dtype),
                    "cp_in_r": [self.r_pos.get(c.counterpart, -1) for c in cands],
                    "item": torch.tensor([item_pos.get(c.ev, -1) if c.ev >= 0 else -1 for c in cands],
                                         device=dev),
                    "current": torch.tensor([c.current for c in cands], device=dev),
                    "step": st,
                })
            self.plan.append(plan_b)

    def __call__(self, g: Optional[Tensor] = None, *, check: bool = False,
                 atol: float = 1e-5) -> Tuple[Tensor, Tensor]:
        dev, dt = self.device, self.dtype
        one = torch.ones((), dtype=dt, device=dev)
        mem = [self.mem0[k] for k in range(len(self.R))]
        lu = [self.lu0[k] for k in range(len(self.R))]
        worst = 0.0
        for plan_b in self.plan:
            pre_mem, pre_lu = list(mem), list(lu)
            for p in plan_b:
                i = self.r_pos[p["node"]]
                cp = p["cp_frozen"].clone()
                for k, rk in enumerate(p["cp_in_r"]):
                    if rk >= 0:
                        cp[k] = pre_mem[rk]
                m_i = pre_mem[i].unsqueeze(0).expand(cp.size(0), -1)
                t_rel = (p["t_off"] - pre_lu[i]).to(dt)
                msg = torch.cat([m_i, cp, p["raw"], self.time_enc(t_rel)], dim=-1)
                H = self.gru(msg, m_i)                                   # [K, D]
                if g is None:
                    gk = torch.ones(cp.size(0), dtype=dt, device=dev)
                else:
                    gk = torch.where(p["item"] >= 0, g[p["item"].clamp_min(0)].to(dt), one)
                surv = torch.cumprod(torch.cat([one.view(1), 1 - gk[:-1]]), 0)
                w = gk * surv
                presence = 1 - torch.prod(torch.where(p["current"], 1 - gk, one))
                s = w.sum().clamp_min(1e-30)
                new_mem = presence * (w.unsqueeze(-1) * H).sum(0) / s + (1 - presence) * pre_mem[i]
                pd = presence.double()
                new_lu = pd * (w.double() * p["t_off"]).sum() / s.double() + (1 - pd) * pre_lu[i]
                mem[i], lu[i] = new_mem, new_lu
                if check and g is None:
                    st = p["step"]
                    worst = max(worst, float((pre_mem[i] - st.pre_mem.to(dev, dt)).abs().max()),
                                float((new_mem - st.post_mem.to(dev, dt)).abs().max()))
                    if abs(float(new_lu) - (st.post_lu - self.hist.t_ref)) > 0.5:
                        raise AssertionError(f"last_update mismatch at batch {st.batch} node {st.node}")
        if check and g is None and worst > atol:
            raise AssertionError(f"functional replay deviates from PyG by {worst:.3e}")
        return torch.stack(mem), torch.stack(lu)