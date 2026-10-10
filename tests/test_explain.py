"""Explainers and metrics on the synthetic fixture."""
import math

import torch

from provex.evaluate import metrics
from provex.explain import baselines
from provex.explain.event_model import EventModel
from provex.explain.gnnexplainer_anomaly import anomaly_gnnexplainer, v1_gnnexplainer
from provex.explain.joint_mask import JointMaskExplainer
from provex.explain.variational import VariationalTemporalExplainer
from provex.replay.kairos_replay import KairosReplayer

from conftest import BATCH, WINDOW_NS


def _top_event(tiny):
    hp, memory, gnn, lp, data = tiny
    rep = KairosReplayer(memory, gnn, lp, hp, batch_size=BATCH, window_ns=WINDOW_NS)
    losses, _ = rep.pass1(data)
    ev = int(torch.argmax(losses[200:])) + 200
    cap = rep.pass2(data, [ev], history_batches=8)[ev]
    em = EventModel(memory, gnn, lp, cap, dtype=torch.float64)
    ch = metrics.channel_decomposition(em)
    tau = 0.5 * (ch["full"] + ch["nothing"])
    return em, tau, data


def test_joint_preserve_and_counterfactual_valid(tiny):
    em, tau, _ = _top_event(tiny)
    for mode in ("preserve", "counterfactual"):
        ex = JointMaskExplainer(mode, epochs=150, seed=0).explain(em, tau)
        assert ex.valid, (mode, ex.L_selected, tau)
        assert len(ex.items) == len(ex.scores) == em.num_items


def test_variational_exports_uncertainty(tiny):
    em, tau, data = _top_event(tiny)
    ex = VariationalTemporalExplainer(epochs=60, samples=4, mc_export=64).explain(em, tau, data.t)
    n = em.num_items
    assert all(0.0 <= p <= 1.0 for p in ex.scores)
    for key in ("mean_m", "std_m", "mu", "sigma", "dt_s"):
        assert len(ex.extra[key]) == n


def test_baselines_and_gnnexplainers_run(tiny):
    em, tau, data = _top_event(tiny)
    for name, fn in baselines.BASELINES.items():
        ex = fn(em, tau, data.t) if name == "recency" else fn(em, tau)
        assert len(ex.scores) == em.num_items, name
    for fn in (anomaly_gnnexplainer, v1_gnnexplainer):
        ex = fn(em, tau, epochs=30)
        assert len(ex.scores) == em.num_items


def test_occlusion_consistent_with_deletion_curve(tiny):
    em, tau, _ = _top_event(tiny)
    occ = baselines.occlusion(em, tau)
    curve = metrics.deletion_curve(em, occ.scores, ks=(1,))
    top = max(s for s in occ.scores if not math.isnan(s))
    assert abs((occ.L_full - curve[1]) - top) < 1e-10


def test_stability_and_agreement_bounds():
    a = [0.9, 0.1, 0.5, 0.3]
    b = [0.8, 0.2, 0.6, 0.1]
    st = metrics.stability([a, b, a], k=2)
    assert 0 <= st["topk_jaccard"] <= 1
    ag = metrics.agreement(a, b, k=2)
    assert ag["topk_jaccard"] == 1.0