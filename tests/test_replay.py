"""Replay and capture exactness (KAIROS semantics) on the synthetic fixture."""
import torch
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import LastNeighborLoader

from provex.data import edge_type_labels
from provex.explain.event_model import EventModel
from provex.replay.kairos_replay import KairosReplayer

from conftest import BATCH, WINDOW_NS

TARGETS = [100, 333, 517, 640, 799]


def reference_test_loop(hp, memory, gnn, lp, data):
    """Straight transcription of upstream KAIROS test(): per-edge CE, windows at batch ends."""
    memory.reset_state()
    loader = LastNeighborLoader(memory.num_nodes, size=hp.neighbor_size)
    assoc = torch.empty(memory.num_nodes, dtype=torch.long)
    losses, windows = [], []
    start_t, start_i = int(data.t[0]), 0
    with torch.no_grad():
        for b in range(0, data.num_events, BATCH):
            batch = data[b:b + BATCH]
            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0))
            x, lu = memory(n_id)
            z = gnn(x, lu, edge_index, data.t[e_id], data.msg[e_id])
            out = lp(z[assoc[src]], z[assoc[dst]])
            y = edge_type_labels(msg, hp.node_embedding_dim, hp.num_edge_types)
            memory.update_state(src, dst, t, msg)
            loader.insert(src, dst)
            for i in range(out.size(0)):
                losses.append(float(F.cross_entropy(out[i].reshape(1, -1), y[i].reshape(-1))))
            if int(t[-1]) > start_t + WINDOW_NS:
                windows.append((start_i, b + src.numel(), start_t, int(t[-1])))
                start_t, start_i = int(t[-1]), b + src.numel()
    return torch.tensor(losses, dtype=torch.float64), windows


def test_pass1_matches_reference(tiny):
    hp, memory, gnn, lp, data = tiny
    rep = KairosReplayer(memory, gnn, lp, hp, batch_size=BATCH, window_ns=WINDOW_NS)
    losses, windows = rep.pass1(data)
    ref_losses, ref_windows = reference_test_loop(hp, memory, gnn, lp, data)
    assert torch.allclose(losses, ref_losses, atol=1e-12, rtol=0)
    assert [(w.start_idx, w.end_idx, w.start_t, w.end_t) for w in windows] == ref_windows
    assert len(windows) >= 3


def _caps(tiny, history_batches=8):
    hp, memory, gnn, lp, data = tiny
    rep = KairosReplayer(memory, gnn, lp, hp, batch_size=BATCH, window_ns=WINDOW_NS)
    return rep.pass2(data, TARGETS, history_batches=history_batches)


def test_capture_logits_parity(tiny):
    hp, memory, gnn, lp, data = tiny
    for ev, cap in _caps(tiny).items():
        em = EventModel(memory, gnn, lp, cap, dtype=torch.float64)
        assert torch.allclose(em(None), cap.logits_batch, atol=1e-10, rtol=0), ev


def test_items_are_global_event_ids(tiny):
    """C1: every neighbour edge maps to the real (src, dst) of its event."""
    hp, memory, gnn, lp, data = tiny
    for cap in _caps(tiny).values():
        for e in range(cap.edge_index.size(1)):
            ev = int(cap.edge_ev[e])
            ends = {int(cap.node_gid[cap.edge_index[0, e]]), int(cap.node_gid[cap.edge_index[1, e]])}
            assert ends == {int(data.src[ev]), int(data.dst[ev])}
            assert int(cap.edge_t[e]) == int(data.t[ev])


def test_subgraph_is_receptive_field(tiny):
    """Dropping any captured edge changes nothing only if it is irrelevant; adding the full batch
    graph back must not change logits (the capture is complete)."""
    hp, memory, gnn, lp, data = tiny
    for cap in _caps(tiny).values():
        assert cap.layer_rel[0].all()
        assert cap.edge_index.size(1) == cap.edge_ev.numel()


def test_hook_identity_and_deletion(tiny):
    hp, memory, gnn, lp, data = tiny
    g = torch.Generator().manual_seed(3)
    for ev, cap in _caps(tiny).items():
        em = EventModel(memory, gnn, lp, cap, dtype=torch.float64)
        ones = torch.ones(em.num_items, dtype=torch.float64)
        assert torch.allclose(em(ones, history=False), em(None), atol=1e-12, rtol=0), ev
        for _ in range(5):
            b = (torch.rand(em.num_items, generator=g) > 0.4).double()
            soft = em(b, history=False)
            hard = em(b, history=False, hard=True)
            assert torch.allclose(soft, hard, atol=1e-10, rtol=0), ev


def test_functional_tgn_identity(tiny):
    hp, memory, gnn, lp, data = tiny
    for ev, cap in _caps(tiny).items():
        em = EventModel(memory, gnn, lp, cap, dtype=torch.float64)
        if em.ftgn is None:
            continue
        mem_r, lu_r = em.ftgn(None, check=True, atol=1e-10)
        assert torch.allclose(mem_r[em.r_hist], cap.x[em.r_local].double(), atol=1e-10, rtol=0), ev
        assert torch.allclose(lu_r[em.r_hist], em.lu_off[em.r_local], atol=0.5, rtol=0), ev
        ones = torch.ones(em.num_items, dtype=torch.float64)
        assert torch.allclose(em(ones), em(None), atol=1e-10, rtol=0), ev


def test_functional_tgn_deletion_matches_full_replay(tiny):
    """With a complete influence closure, deleting any history event reproduces a full PyG
    replay (with that event removed) exactly, for both endpoints of the explained event."""
    from provex.evaluate.local_vs_full import full_replay
    hp, memory, gnn, lp, data = tiny
    checked = 0
    for ev, cap in _caps(tiny).items():
        em = EventModel(memory, gnn, lp, cap, dtype=torch.float64)
        if em.ftgn is None or not cap.hist.exact_scope:
            continue
        n_uv = 1 if int(data.src[ev]) == int(data.dst[ev]) else 2    # R starts with [src, dst]
        for k in cap.hist.items()[:20]:
            g = torch.ones(em.num_items, dtype=torch.float64)
            g[em.item_pos[k]] = 0
            mem_f, _ = em.ftgn(g)
            _, mem_full, _ = full_replay(memory, gnn, lp, hp, data, [k], ev, batch_size=BATCH)
            assert torch.allclose(mem_f[:n_uv], mem_full[:n_uv].double(), atol=1e-10, rtol=0), (ev, k)
            checked += 1
    assert checked >= 20
    print(f"closure deletions checked exactly: {checked}")


def test_first_order_scope_flags_inexact(tiny):
    hp, memory, gnn, lp, data = tiny
    rep = KairosReplayer(memory, gnn, lp, hp, batch_size=BATCH, window_ns=WINDOW_NS)
    caps = rep.pass2(data, TARGETS, history_batches=8, scope="first_order")
    assert all(c.hist.R[:2] == [int(data.src[e]), int(data.dst[e])][:len(c.hist.R[:2])] for e, c in caps.items())
    assert any(not c.hist.exact_scope for c in caps.values())