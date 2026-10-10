"""Real-data smoke test of the v2 explainer stack (uses the v1 fork checkpoint; runs on a VM).

    python scripts/smoke_real.py --ckpt artifact/models/models.pt --day-file artifact/graph_embeddings/graph_4_6.TemporalData.simple
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from provex.data import load_day
from provex.evaluate import metrics as M
from provex.explain import baselines
from provex.explain.event_model import EventModel
from provex.explain.joint_mask import JointMaskExplainer
from provex.explain.variational import VariationalTemporalExplainer
from provex.model import load_legacy_checkpoint
from provex.replay.kairos_replay import KairosReplayer
from provex.replay.alerts import window_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--day-file", required=True)
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--history-batches", type=int, default=16)
    a = ap.parse_args()
    import types
    if "psycopg2" not in sys.modules:   # v1 utils import psycopg2; the DB is not used here
        pg = types.ModuleType("psycopg2"); pg.extras = types.ModuleType("psycopg2.extras")
        sys.modules["psycopg2"], sys.modules["psycopg2.extras"] = pg, pg.extras
    import kairos_tgnn_model as legacy  # v1 module for unpickling
    memory, gnn, lp, hp = load_legacy_checkpoint(a.ckpt, legacy, "cpu", rel_t_sign=-1)
    print("hparams", hp, flush=True)
    data = load_day(a.day_file)
    rep = KairosReplayer(memory, gnn, lp, hp)
    t0 = time.perf_counter(); losses, windows = rep.pass1(data)
    print(f"pass1: {data.num_events} events, {len(windows)} windows, {time.perf_counter() - t0:.1f}s", flush=True)
    # window containing 2018-04-06 11:30 EDT = 15:30 UTC
    t_target = 1523028600 * 10**9
    w = next(w for w in windows if w.start_t <= t_target <= w.end_t)
    tau = window_threshold(losses, w)
    seg = losses[w.start_idx:w.end_idx]
    top = (torch.argsort(seg, descending=True)[:a.n] + w.start_idx).tolist()
    print(f"window [{w.start_idx},{w.end_idx}) tau={tau:.3f} top={top} L={[round(float(losses[i]), 2) for i in top]}", flush=True)
    t0 = time.perf_counter()
    caps = rep.pass2(data, top, history_batches=a.history_batches)
    print(f"pass2 captures: {time.perf_counter() - t0:.1f}s", flush=True)
    for ev in top:
        cap = caps[ev]; cap.tau = tau
        h = cap.hist
        n_steps = sum(len(b) for b in h.steps); n_cands = sum(len(s.cands) for b in h.steps for s in b)
        em = EventModel(memory, gnn, lp, cap)
        par = float((em(None) - cap.logits_batch).abs().max())
        t1 = time.perf_counter()
        try:
            em.ftgn(None, check=True, atol=1e-3); ident = "ok"
        except AssertionError as e:
            ident = str(e)
        t_id = time.perf_counter() - t1
        print(f"ev {ev}: L={cap.loss_batch:.2f} |S|={len(h.R)} exact_scope={h.exact_scope} steps={n_steps} cands={n_cands} "
              f"items={em.num_items} (edge {int(em.is_edge_item.sum())}, hist {int(em.is_history_item.sum())}) "
              f"logit_parity={par:.2e} ftgn_identity={ident} ftgn_time={t_id:.2f}s", flush=True)
        ch = M.channel_decomposition(em); print("  channels", {k: round(v, 2) for k, v in ch.items()}, flush=True)
        for name, fn in (("joint", lambda: JointMaskExplainer("preserve", epochs=100).explain(em, tau)),
                         ("joint-cf", lambda: JointMaskExplainer("counterfactual", epochs=100).explain(em, tau)),
                         ("variational", lambda: VariationalTemporalExplainer(epochs=60, samples=4).explain(em, tau, data.t)),
                         ("gradient", lambda: baselines.gradient(em, tau))):
            t1 = time.perf_counter(); ex = fn(); dt = time.perf_counter() - t1
            print(f"  {name}: valid={ex.valid} L_sel={ex.L_selected:.2f} |sel|={len(ex.selected)} time={dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()