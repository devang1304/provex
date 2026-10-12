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

def test_graphmask_fit_and_explain(tiny):
    from provex.explain.graphmask_tc import GraphMaskTC
    hp, memory, gnn, lp, data = tiny
    rep = KairosReplayer(memory, gnn, lp, hp, batch_size=BATCH, window_ns=WINDOW_NS)
    losses, _ = rep.pass1(data)
    evs = torch.argsort(losses[300:], descending=True)[:6].add(300).tolist()
    caps = rep.pass2(data, evs, history_batches=0)
    models = [EventModel(memory, gnn, lp, caps[e], use_history=False, dtype=torch.float64) for e in evs]
    gm = GraphMaskTC(gnn).double()
    hist = gm.fit(models, epochs=3, log=lambda r: None)
    assert len(hist) == 4 and hist[-1]["final"] and 0.0 <= hist[-1]["open_frac_hard"] <= 1.0
    assert all(math.isfinite(r["mean_open_frac"]) for r in hist[:3])     # no NaN from empty gate sets
    ex = gm.explain(models[0], tau=0.0)
    assert len(ex.scores) == models[0].num_items


def test_graphmask_gate_bound_keeps_gradient():
    """A gate pinned at the lower bound must still get gradient, or it can never reopen."""
    from provex.explain.graphmask_tc import GateNet
    g = GateNet(heads=2, channels=3, init_bias=-100.0)
    q = k = v = torch.randn(5, 2, 3)
    a = g(q, k, v, torch.zeros(5, 2))
    assert ((a > -6) & (a < -4)).all()
    a.sum().backward()
    assert g.out.bias.grad.abs().item() > 0


def test_calibration_enumeration_runs(tiny):
    from provex.evaluate.calibration import ece, posterior_calibration
    em, tau, data = _top_event(tiny)
    ex = VariationalTemporalExplainer(epochs=40, samples=4, mc_export=32).explain(em, tau, data.t)
    cal = posterior_calibration(em, ex, tau, n=6)
    assert cal["n"] == 6 and all(0 <= p <= 1 for p in cal["pi_exact"])
    assert 0 <= ece(list(zip(cal["pi_var"], cal["pi_exact"]))) <= 1


def test_joint_names_and_floor_alerts(tiny):
    """Counterfactual records are named joint-cf; when removing every item keeps the alert, preserve returns
    the empty set but counterfactual mode still searches (L is not monotone in the removed set)."""
    em, _, _ = _top_event(tiny)
    assert JointMaskExplainer("preserve").name == "joint"
    assert JointMaskExplainer("counterfactual").name == "joint-cf"
    assert JointMaskExplainer("preserve", history=False).name == "joint-edges"
    L_floor = float(em.loss(torch.zeros(em.num_items, dtype=em.dtype), hard=True))
    low = L_floor - 1.0
    pre = JointMaskExplainer("preserve", epochs=20).explain(em, low)
    assert pre.method == "joint" and pre.valid and pre.selected == [] and pre.extra["floor_alert"]
    cf = JointMaskExplainer("counterfactual", epochs=40, seed=0).explain(em, low)
    assert cf.method == "joint-cf" and cf.extra["floor_alert"]
    assert all(not math.isnan(s) for s in cf.scores)                  # the search ran
    assert cf.extra["cf_found"] == cf.valid
    if cf.valid:                                                      # a reported flip must be real
        g = torch.ones(em.num_items, dtype=em.dtype)
        for e in cf.selected:
            g[em.item_pos[e]] = 0
        assert float(em.loss(g, hard=True)) < low


def test_alert_class_window_first():
    assert metrics.alert_class({"nothing": 1.0, "prior_only": 5.0}, 2.0) == "window"
    assert metrics.alert_class({"nothing": 3.0, "prior_only": 5.0}, 2.0) == "prior"
    assert metrics.alert_class({"nothing": 3.0, "prior_only": 1.0}, 2.0) == "long_memory"

def test_mcts_subset_is_checked(tiny):
    from provex.explain.mcts import mcts_explain
    em, tau, _ = _top_event(tiny)
    ex = mcts_explain(em, tau, n_candidates=12, k_max=4, n_sim=24)
    assert ex.method == "mcts" and len(ex.scores) == em.num_items
    assert len(ex.selected) <= 4 and set(ex.selected) <= set(em.items.tolist())
    g = torch.zeros(em.num_items, dtype=em.dtype)
    for e in ex.selected:
        g[em.item_pos[e]] = 1
    L_keep = float(em.loss(g, hard=True))
    assert abs(L_keep - ex.L_selected) < 1e-9 and ex.valid == (L_keep >= tau)
    top = sorted(range(em.num_items), key=lambda k: -ex.scores[k])[:len(ex.selected)]
    assert {int(em.items[k]) for k in top} == set(ex.selected)          # the subset is ranked first

def test_load_gt_nodes_formats(tmp_path):
    import json as _json
    from provex.evaluate.ground_truth import load_gt_nodes
    (tmp_path / "kairos_nodes_Nginx_Backdoor_06.json").write_text(_json.dumps(
        {"attack": "Nginx_Backdoor_06", "nodes": {"5": {"name": "file: /tmp/x"}}, "rows": [], "stats": {}}))
    (tmp_path / "kairos_nodes_Nginx_Backdoor_12.json").write_text(_json.dumps({"9": "subject: nginx"}))
    assert load_gt_nodes(tmp_path) == {"4_6": {5}, "4_12": {9}}

def test_bounded_occlusion_and_grid_min_cf(tiny):
    em, tau, _ = _top_event(tiny)
    ex = baselines.occlusion(em, tau, top_m=3)
    scored = [s for s in ex.scores if not math.isnan(s)]
    assert len(scored) == min(3, em.num_items) and ex.extra["candidates"] == len(scored)
    k = metrics.min_cf_size(em, ex.scores, tau, k_max=8, grid=[1, 2, 4, 8])
    assert k is None or k in (1, 2, 4, 8)
    if k is not None:                                                  # the reported prefix really flips
        order = metrics._order(ex.scores)
        g = torch.ones(em.num_items, dtype=em.dtype); g[order[:k]] = 0
        assert float(em.loss(g, hard=True)) < tau


def test_explainers_leave_model_in_eval_mode(tiny):
    """PyG's Explainer restores its wrapper's training flag on exit, and a new wrapper starts in training
    mode. That once switched on KAIROS's link-predictor dropout for every later score (step-5 comparison)."""
    em, tau, data = _top_event(tiny)
    ones = torch.ones(em.num_items, dtype=em.dtype)
    L0 = float(em.loss(ones, hard=True))
    for fn in (anomaly_gnnexplainer, v1_gnnexplainer):
        fn(em, tau, epochs=5)
        assert not em.gnn.training and not em.link_pred.training, fn.__name__
        assert float(em.loss(ones, hard=True)) == L0, fn.__name__