"""KAIROS-exact replay of one day graph.

Semantics match upstream ``test.py``:
- memory and neighbour loader are reset at the start of the day;
- events are processed in batches of 1024, in file order;
- each event is scored by the cross-entropy of its event type;
- memory is updated after each batch is scored;
- a 15-minute window closes at a batch boundary when ``t[-1] > start_time + window``.

Events after the last closed window are scored but, as upstream, belong to no window.

Run replays on CPU. When timestamps tie, the CUDA scatter_max kernel of torch_scatter picks the
last memory message nondeterministically. The CPU kernel deterministically keeps the first
maximum in concatenation order, which provex.explain.functional_tgn emulates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.models.tgn import LastNeighborLoader

from ..data import edge_type_labels
from ..model import KairosHParams
from .capture import Cand, EventCapture, HistoryContext, Step, build_capture

WINDOW_NS = 15 * 60 * 1_000_000_000


@dataclass
class Window:
    start_idx: int   # first event index (inclusive)
    end_idx: int     # last event index (exclusive)
    start_t: int     # upstream file-name start time (ns)
    end_t: int       # upstream file-name end time (ns) = t of the last event in the window


@dataclass
class BatchState:
    index: int
    begin: int
    end: int
    src: Tensor
    dst: Tensor
    t: Tensor
    msg: Tensor
    n_id: Tensor
    edge_index: Tensor
    e_id: Tensor
    x: Tensor
    lu: Tensor
    logits: Tensor
    y: Tensor
    assoc: Tensor


class KairosReplayer:
    def __init__(self, memory, gnn, link_pred, hp: KairosHParams, *, batch_size: int = 1024,
                 window_ns: int = WINDOW_NS, device="cpu"):
        self.memory, self.gnn, self.link_pred, self.hp = memory, gnn, link_pred, hp
        self.batch_size, self.window_ns, self.device = batch_size, window_ns, torch.device(device)

    # ------------------------------------------------------------------ core loop
    @torch.no_grad()
    def _replay(self, data, *, pre_update: Optional[Callable[[BatchState], None]] = None,
                post_update: Optional[Callable[[BatchState], None]] = None,
                stop_after_event: Optional[int] = None) -> None:
        mem, gnn, lp, dev = self.memory, self.gnn, self.link_pred, self.device
        mem.eval(); gnn.eval(); lp.eval()
        mem.reset_state()
        loader = LastNeighborLoader(mem.num_nodes, size=self.hp.neighbor_size, device=dev)
        assoc = torch.empty(mem.num_nodes, dtype=torch.long, device=dev)
        n_events = data.num_events
        for bi, b in enumerate(range(0, n_events, self.batch_size)):
            e = min(b + self.batch_size, n_events)
            src, dst = data.src[b:e].to(dev), data.dst[b:e].to(dev)
            t, msg = data.t[b:e].to(dev), data.msg[b:e].to(dev)
            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=dev)
            x, lu = mem(n_id)
            e_cpu = e_id.cpu()
            z = gnn(x, lu, edge_index, data.t[e_cpu].to(dev), data.msg[e_cpu].to(dev))
            logits = lp(z[assoc[src]], z[assoc[dst]])
            y = edge_type_labels(msg, self.hp.node_embedding_dim, self.hp.num_edge_types)
            st = BatchState(bi, b, e, src, dst, t, msg, n_id, edge_index, e_id, x, lu, logits, y, assoc)
            if pre_update is not None:
                pre_update(st)
            mem.update_state(src, dst, t, msg)
            loader.insert(src, dst)
            if post_update is not None:
                post_update(st)
            if stop_after_event is not None and e > stop_after_event:
                break

    # ------------------------------------------------------------------ pass 1
    def pass1(self, data):
        """Per-event loss for the whole day and upstream-identical window boundaries."""
        losses = torch.empty(data.num_events, dtype=torch.float64)
        windows: List[Window] = []
        state = {"start_t": int(data.t[0]), "start_idx": 0}

        def on_batch(st: BatchState):
            losses[st.begin:st.end] = F.cross_entropy(st.logits, st.y, reduction="none").double().cpu()
            last_t = int(st.t[-1])
            if last_t > state["start_t"] + self.window_ns:
                windows.append(Window(state["start_idx"], st.end, state["start_t"], last_t))
                state["start_t"], state["start_idx"] = last_t, st.end

        self._replay(data, pre_update=on_batch)
        return losses, windows

    # ------------------------------------------------------------------ pass 2
    def pass2(self, data, targets: Iterable[int], *, history_batches: int = 64, scope: str = "closure",
              closure_cap: int = 64) -> Dict[int, EventCapture]:
        """Capture the computation subgraph and memory history of each target event.

        ``targets`` are absolute event indices in ``data``. For each target e = (u, v) in batch B,
        the history window is batches [B - history_batches, B). For every batch in it that
        touches u or v, the candidate memory messages are recorded in PyG precedence order:
        s-store entries, then d-store entries, each in store order. Counterpart memories are
        frozen at their pre-batch values, and ground-truth pre/post memories are kept for checks.
        """
        targets = sorted(set(int(i) for i in targets))
        if not targets:
            return {}
        bs = self.batch_size
        mem = self.memory
        by_batch: Dict[int, List[int]] = {}
        for ev in targets:
            by_batch.setdefault(ev // bs, []).append(ev)
        b0_of = {ev: max(0, ev // bs - history_batches) for ev in targets}
        r_of, exact_of = {}, {}
        for ev in targets:
            r_of[ev], exact_of[ev] = influence_set(data, ev, b0_of[ev], bs, scope, closure_cap)
        watch: Set[int] = set().union(*r_of.values())
        first_b = min(b0_of.values())
        snap_at: Dict[int, List[int]] = {}
        for ev in targets:
            snap_at.setdefault(b0_of[ev], []).append(ev)

        caps: Dict[int, EventCapture] = {}
        hist_steps: Dict[int, List[List[Step]]] = {ev: [] for ev in targets}
        snapshots: Dict[int, tuple] = {}
        last_s: Dict[int, List[int]] = {}    # node -> absolute events currently in its s-store
        last_d: Dict[int, List[int]] = {}
        pending: Dict[int, Dict] = {}
        counterparts: Dict[int, Dict[int, Tensor]] = {}

        def store_entries(store, node, current_events, tracked):
            _, s_dst, s_t, s_raw = store[node]
            n = int(s_t.numel())
            evs = current_events if current_events is not None else tracked.get(node, [-1] * n)
            if len(evs) != n:
                evs = [-1] * n
            return [(evs[k], int(s_dst[k]), int(s_t[k]), s_raw[k].detach().cpu().clone()) for k in range(n)]

        def pre_update(st: BatchState):
            for ev in snap_at.get(st.index, []):
                r = torch.tensor(r_of[ev], device=mem.memory.device)
                snapshots[ev] = (mem.memory[r].detach().cpu().clone(), mem.last_update[r].cpu().clone())
            for ev in by_batch.get(st.index, []):
                pos = ev - st.begin
                a_u, a_v = int(st.assoc[st.src[pos]]), int(st.assoc[st.dst[pos]])
                loss = float(F.cross_entropy(st.logits[pos:pos + 1], st.y[pos:pos + 1]))
                caps[ev] = build_capture(
                    event=ev, batch=st.index, pos=pos, src_g=int(st.src[pos]), dst_g=int(st.dst[pos]),
                    t=int(st.t[pos]), y=int(st.y[pos]), loss=loss, logits=st.logits[pos],
                    n_id=st.n_id, edge_index_b=st.edge_index, e_id=st.e_id, x_b=st.x, lu_b=st.lu,
                    a_u=a_u, a_v=a_v, data=data)
            if st.index < first_b:
                return
            touched = [n for n in torch.cat([st.src, st.dst]).unique().tolist() if n in watch]
            if not touched:
                return
            # Same calls as TGNMemory._update_msg_store, so the store order is reproduced exactly.
            s_sorted, s_perm = st.src.sort()
            d_sorted, d_perm = st.dst.sort()
            s_pairs = list(zip(s_sorted.tolist(), s_perm.tolist()))
            d_pairs = list(zip(d_sorted.tolist(), d_perm.tolist()))
            rec, cps = {}, {}
            for n in touched:
                cur_s = [st.begin + p for v, p in s_pairs if v == n]
                cur_d = [st.begin + p for v, p in d_pairs if v == n]
                rec[n] = {"cur_s": cur_s, "cur_d": cur_d,
                          "pre_mem": mem.memory[n].detach().cpu().clone(), "pre_lu": int(mem.last_update[n])}
                for ev_k in cur_s:
                    cps[int(data.dst[ev_k])] = None
                for ev_k in cur_d:
                    cps[int(data.src[ev_k])] = None
                if not cur_s:
                    cps.update({int(c): None for c in mem.msg_s_store[n][1].tolist()})
                if not cur_d:
                    cps.update({int(c): None for c in mem.msg_d_store[n][1].tolist()})
            for c in cps:
                cps[c] = mem.memory[c].detach().cpu().clone()
            pending[st.index] = rec
            counterparts[st.index] = cps

        def post_update(st: BatchState):
            rec = pending.pop(st.index, None)
            cps = counterparts.pop(st.index, None)
            if rec is None:
                return
            batch_steps: Dict[int, Step] = {}
            for n, r in rec.items():
                s_entries = store_entries(mem.msg_s_store, n, r["cur_s"] or None, last_s)
                d_entries = store_entries(mem.msg_d_store, n, r["cur_d"] or None, last_d)
                cands: List[Cand] = []
                for entries, is_cur in ((s_entries, bool(r["cur_s"])), (d_entries, bool(r["cur_d"]))):
                    for ev_k, c, t_k, raw_k in entries:
                        cands.append(Cand(ev=ev_k, counterpart=c, t=t_k, raw=raw_k, current=is_cur,
                                          counterpart_mem=cps[c]))
                batch_steps[n] = Step(batch=st.index, node=n, cands=cands, pre_mem=r["pre_mem"],
                                      pre_lu=r["pre_lu"], post_mem=mem.memory[n].detach().cpu().clone(),
                                      post_lu=int(mem.last_update[n]))
                if r["cur_s"]:
                    last_s[n] = r["cur_s"]
                if r["cur_d"]:
                    last_d[n] = r["cur_d"]
            for ev in targets:
                if b0_of[ev] <= st.index < ev // bs:
                    steps = [batch_steps[n] for n in r_of[ev] if n in batch_steps]
                    if steps:
                        hist_steps[ev].append(steps)

        self._replay(data, pre_update=pre_update, post_update=post_update, stop_after_event=max(targets))
        t_ref = int(data.t[0])
        for ev, cap in caps.items():
            mem0, lu0 = snapshots[ev]
            cap.hist = HistoryContext(R=r_of[ev], b0=b0_of[ev], mem0=mem0, lu0=lu0,
                                      steps=hist_steps[ev], t_ref=t_ref, scope=scope, exact_scope=exact_of[ev],
                                      ev_lo=b0_of[ev] * bs, ev_hi=(ev // bs) * bs)
        return caps


def influence_set(data, event: int, b0: int, batch_size: int, scope: str = "closure", cap: int = 64):
    """Nodes whose memory can influence u/v's memory at the event's batch, within [b0, B_e).

    first_order: {u, v}. closure: walk batches backwards. If node i is in S and has an event
    (i, c) in batch b, then c's pre-batch memory fed i's update, so c joins S for earlier
    batches. Returns (sorted node list with u, v first, exact). exact is False when the cap
    truncated the closure, or when the scope is first_order and some counterpart would have
    joined.
    """
    u, v = int(data.src[event]), int(data.dst[event])
    S = [u] if u == v else [u, v]
    inS = set(S)
    truncated = False
    for b in range(event // batch_size - 1, b0 - 1, -1):
        lo, hi = b * batch_size, min((b + 1) * batch_size, data.num_events)
        src, dst = data.src[lo:hi].tolist(), data.dst[lo:hi].tolist()
        new = []
        for a, c in zip(src, dst):
            if a in inS and c not in inS:
                new.append(c)
            if c in inS and a not in inS:
                new.append(a)
        for n in dict.fromkeys(new):
            if scope == "first_order" or len(S) >= cap:
                truncated = True
                continue
            S.append(n); inS.add(n)
    return S, not truncated
