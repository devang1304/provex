"""PROVEX v2 command-line pipeline (runs on the GCP VMs, never on the laptop).

    python -m provex.pipeline pass1   --ckpt CKPT --day-file graph_4_6.TemporalData.simple --day 4_6 --out OUT
    python -m provex.pipeline parity  --pass1 OUT/pass1_4_6.pt --kairos-windows windows.json --day 4_6 --out OUT
    python -m provex.pipeline explain --ckpt CKPT --day-file ... --day 4_6 --alerts alerts.json --names node_mapping.json \
                                      --gt-dir GT --methods joint,joint-cf,variational,gnnx-anomaly,gnnx-v1,baselines \
                                      --seeds 0 --device cuda --out OUT
    python -m provex.pipeline graphmask-fit --ckpt CKPT --day-file graph_4_5... --day 4_5 --out OUT/graphmask.pt

Replays (pass 1 and pass 2 captures) always run on CPU, so that tie-breaking is deterministic.
Explainers run on --device.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_day, load_node_names
from .evaluate import metrics as M
from .evaluate.ground_truth import load_gt_nodes
from .evaluate.results import event_record, explanation_record
from .explain import baselines
from .explain.event_model import EventModel
from .explain.gnnexplainer_anomaly import anomaly_gnnexplainer, v1_gnnexplainer
from .explain.graphmask_tc import GraphMaskTC
from .explain.joint_mask import JointMaskExplainer
from .explain.variational import VariationalTemporalExplainer
from .model import load_checkpoint
from .replay import alerts as A
from .replay.kairos_replay import KairosReplayer, Window

KS = (1, 3, 5, 10, 20)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- pass 1
def cmd_pass1(a):
    memory, gnn, lp, hp = load_checkpoint(a.ckpt, "cpu")
    data = load_day(a.day_file)
    t0 = time.perf_counter()
    losses, windows = KairosReplayer(memory, gnn, lp, hp, batch_size=a.batch_size, window_ns=a.window_ns).pass1(data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"losses": losses, "windows": [w.__dict__ for w in windows], "day": a.day,
                "ckpt": str(a.ckpt), "runtime_s": time.perf_counter() - t0}, out / f"pass1_{a.day}.pt")
    rows = [{"start_idx": w.start_idx, "end_idx": w.end_idx, "start_ns": w.start_t, "end_ns": w.end_t,
             "n": w.end_idx - w.start_idx, "threshold": A.window_threshold(losses, w)} for w in windows]
    (out / f"windows_{a.day}.json").write_text(json.dumps(rows, indent=1))
    log(f"pass1 {a.day}: {data.num_events} events, {len(windows)} windows, "
        f"{time.perf_counter() - t0:.1f}s, peak RSS {peak_rss_mb():.0f} MB")


def _load_pass1(path):
    p = torch.load(path, weights_only=False)
    return p["losses"], [Window(**w) for w in p["windows"]]


# ---------------------------------------------------------------------------- parity
def cmd_parity(a):
    losses, windows = _load_pass1(a.pass1)
    kw = [r for r in json.loads(Path(a.kairos_windows).read_text()) if r["day"] == a.day]
    rep = {"day": a.day, "n_provex": len(windows), "n_kairos": len(kw), "mismatches": []}
    for w, k in zip(windows, kw):
        thr = A.window_threshold(losses, w)
        if w.start_t != k["start_ns"] or w.end_t != k["end_ns"] or abs(thr - k["threshold"]) > 1e-4 * max(1, abs(thr)):
            rep["mismatches"].append({"provex": [w.start_t, w.end_t, thr],
                                      "kairos": [k["start_ns"], k["end_ns"], k["threshold"]]})
    rep["ok"] = rep["n_provex"] == rep["n_kairos"] and not rep["mismatches"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"parity_{a.day}.json").write_text(json.dumps(rep, indent=1))
    log(f"parity {a.day}: ok={rep['ok']} ({len(rep['mismatches'])} mismatches)")
    sys.exit(0 if rep["ok"] else 1)


# ---------------------------------------------------------------------------- explain
# Per-method budgets. "bounded" keeps events with tens of thousands of history items affordable;
# the budget used is recorded in every explanation record.
BUDGETS = {
    "full": {"joint_epochs": 300, "joint_repair": 512, "var_epochs": 300, "var_samples": 8, "var_mc": 256,
             "gnnx_epochs": 200, "occlusion_top": None, "ig_steps": 20, "mincf_kmax": 256, "mincf_grid": None},
    "bounded": {"joint_epochs": 150, "joint_repair": 64, "var_epochs": 150, "var_samples": 4, "var_mc": 128,
                "gnnx_epochs": 100, "occlusion_top": 200, "ig_steps": 10, "mincf_kmax": 64,
                "mincf_grid": [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]},
}


def _metrics_for(em, ex, tau, data, gt, exclude, window=True, B=BUDGETS["full"]):
    out = {"deletion": M.deletion_curve(em, ex.scores, KS), "insertion": M.insertion_curve(em, ex.scores, KS)}
    out["flip_at_k"] = {k: M.alert_flip(v, tau) for k, v in out["deletion"].items()}
    out["keep_at_k"] = {k: M.alert_keep(v, tau) for k, v in out["insertion"].items()}
    if window:   # a linear prefix scan; elsewhere removing every item does not flip, so it rarely ends early
        out["min_cf_size"] = M.min_cf_size(em, ex.scores, tau, k_max=B["mincf_kmax"], grid=B["mincf_grid"])
    if gt is not None:
        ent = M.entity_scores(ex.items, ex.scores, data, exclude)
        out["gt"] = {f"p@{k}": M.precision_at_k(ent, gt, k) for k in (1, 3, 5, 10)}
        out["gt"]["auroc"] = M.entity_auroc(ent, gt)
    return out


def _run_methods(em, tau, data, methods, seed, graphmask, B=BUDGETS["full"]):
    """Yields each explanation as soon as it is computed (so the caller can log per-method timing)."""
    if "joint" in methods:
        yield (JointMaskExplainer("preserve", seed=seed, epochs=B["joint_epochs"], max_repair=B["joint_repair"]).explain(em, tau))
    if "joint-cf" in methods:
        yield (JointMaskExplainer("counterfactual", seed=seed, epochs=B["joint_epochs"],
                                  max_repair=B["joint_repair"]).explain(em, tau))
    if "joint-edges" in methods:
        yield (JointMaskExplainer("preserve", history=False, seed=seed, epochs=B["joint_epochs"],
                                  max_repair=B["joint_repair"]).explain(em, tau))
    if "variational" in methods:
        yield (VariationalTemporalExplainer(seed=seed, epochs=B["var_epochs"], samples=B["var_samples"],
                                            mc_export=B["var_mc"]).explain(em, tau, data.t))
    if "graphmask" in methods and graphmask is not None:
        yield (graphmask.explain(em, tau))
    if "gnnx-anomaly" in methods:
        yield (anomaly_gnnexplainer(em, tau, epochs=B["gnnx_epochs"], seed=seed))
    if "gnnx-v1" in methods:
        yield (v1_gnnexplainer(em, tau, epochs=B["gnnx_epochs"], seed=seed))
    if "mcts" in methods:
        from .explain.mcts import mcts_explain
        yield (mcts_explain(em, tau))
    if "baselines" in methods:
        for name, fn in baselines.BASELINES.items():
            if name == "recency":
                yield (fn(em, tau, data.t))
            elif name == "random":
                yield (fn(em, tau, seed=seed))
            elif name == "occlusion":
                yield (fn(em, tau, top_m=B["occlusion_top"]))
            elif name == "integrated_gradients":
                yield (fn(em, tau, steps=B["ig_steps"]))
            else:
                yield (fn(em, tau))


def cmd_explain(a):
    set_seed(0)
    dev = torch.device(a.device)
    memory, gnn, lp, hp = load_checkpoint(a.ckpt, "cpu")
    data = load_day(a.day_file)
    names = load_node_names(a.names) if a.names else {}
    gt = load_gt_nodes(a.gt_dir).get(a.day) if a.gt_dir else None
    if a.pass1 and Path(a.pass1).exists():
        losses, windows = _load_pass1(a.pass1)
    else:
        losses, windows = KairosReplayer(memory, gnn, lp, hp).pass1(data)
    alerts = A.load_alerts(a.alerts, a.day)
    log(f"{a.day}: {len(alerts)} alerted windows")
    targets, tau_of, origin = [], {}, {}
    seen_windows = set()
    for al in alerts:
        w = A.match_window(al, windows)
        if (w.start_idx, w.end_idx) in seen_windows:
            continue
        seen_windows.add((w.start_idx, w.end_idx))
        tau = A.window_threshold(losses, w)
        evs = A.anomalous_events(losses, w, tau)
        top, sample = A.choose_targets(evs, a.top_n, a.sample_n, seed=0)
        for ev in top + sample:
            targets.append(ev); tau_of[ev] = tau
            origin[ev] = {"window": al.window_file, "queue_id": al.queue_id, "rank": "top" if ev in top else "sample",
                          "n_anomalous_in_window": len(evs), "gt_window": al.is_gt_attack_window}
    if a.events:                                      # explicit event ids: "1,2,3" or "@file" (one id per line)
        spec = Path(a.events[1:]).read_text().split() if a.events.startswith("@") else a.events.split(",")
        keep = {int(x) for x in spec if x.strip()}
        targets = [e for e in targets if e in keep]
    if a.subsample and len(targets) > a.subsample:   # seeded sample across all alerted windows (E6 subsets)
        targets = sorted(np.random.default_rng(0).choice(targets, size=a.subsample, replace=False).tolist())
    if a.max_events:
        targets = targets[:a.max_events]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        targets = targets[i::n]
    log(f"{a.day}: {len(targets)} target events; capturing (CPU replay)")
    t0 = time.perf_counter()
    caps = KairosReplayer(memory, gnn, lp, hp).pass2(data, targets, history_batches=a.history_batches)
    log(f"captures done in {time.perf_counter() - t0:.1f}s, peak RSS {peak_rss_mb():.0f} MB")
    memory, gnn, lp = memory.to(dev), gnn.to(dev), lp.to(dev)
    graphmask = None
    if "graphmask" in a.methods and a.graphmask:
        graphmask = GraphMaskTC(gnn).to(dev)
        graphmask.load_state_dict(torch.load(a.graphmask, map_location=dev))
    dtype = getattr(torch, a.dtype)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{a.shard.replace('/', 'of')}" if a.shard else ""
    fout = open(out / f"explanations_{a.day}{suffix}.jsonl", "a", encoding="utf-8")
    methods = set(a.methods.split(","))
    for i, ev in enumerate(targets):
        cap = caps[ev]; cap.tau = tau_of[ev]
        em = EventModel(memory, gnn, lp, cap, dtype=dtype, device=dev)
        exclude = {cap.src, cap.dst}
        if a.only_class:
            ch0 = M.channel_decomposition(em)
            if M.alert_class(ch0, cap.tau) != a.only_class:
                continue
        rec = {"event": event_record(cap, data, names, a.day), "origin": origin[ev],
               "channels": (ch := M.channel_decomposition(em)), "alert_class": M.alert_class(ch, cap.tau),
               "n_items": em.num_items,
               "n_edge_items": int(em.is_edge_item.sum()), "n_history_items": int(em.is_history_item.sum()),
               "gt_endpoint": (gt is not None) and bool(exclude & gt), "explanations": []}
        with torch.no_grad():
            t_f = time.perf_counter()
            em.loss(torch.ones(em.num_items, dtype=em.dtype, device=em.device), hard=True)
            rec["forward_ms"] = 1000 * (time.perf_counter() - t_f)
        hc = cap.hist
        log(f"ev {ev}: class {rec['alert_class']}, items {em.num_items} (edges {rec['n_edge_items']}, history "
            f"{rec['n_history_items']}), closure {len(hc.R) if hc is not None else 0} nodes "
            f"(exact {getattr(hc, 'exact_scope', None)}), one forward {rec['forward_ms']:.0f} ms")
        for seed in [int(s) for s in a.seeds.split(",")]:
            set_seed(seed)
            for ex in _run_methods(em, cap.tau, data, methods, seed, graphmask, B=BUDGETS[a.budget]):
                t_m = time.perf_counter()
                mets = _metrics_for(em, ex, cap.tau, data, gt, exclude, window=rec["alert_class"] == "window",
                                    B=BUDGETS[a.budget])
                if a.calibration and ex.method == "variational" and rec["alert_class"] == "window":
                    from .evaluate.calibration import posterior_calibration
                    mets["calibration"] = posterior_calibration(em, ex, cap.tau, n=a.calibration_n)
                er = explanation_record(
                    cap, ex, data, names, is_edge=em.is_edge_item.tolist(), is_hist=em.is_history_item.tolist(),
                    metrics=mets, seed=seed)
                if a.full_scores:
                    er["items_all"], er["scores_all"] = ex.items, ex.scores
                er["metrics_s"] = time.perf_counter() - t_m
                er["budget"] = a.budget
                rec["explanations"].append(er)
                log(f"  {ex.method}/{ex.mode}: explain {ex.runtime_s:.1f}s, metrics {er['metrics_s']:.1f}s")
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        log(f"[{i + 1}/{len(targets)}] ev {ev} L={cap.loss_batch:.2f} tau={cap.tau:.2f} items={em.num_items} "
            f"peak RSS {peak_rss_mb():.0f} MB" + (f", GPU peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB"
                                                    if dev.type == "cuda" else ""))
    fout.close()


# ---------------------------------------------------------------------------- local-vs-full (E7)
def _lvf_case(args):
    ckpt, day_file, ev, deleted, bs = args
    from .evaluate.local_vs_full import full_replay
    memory, gnn, lp, hp = load_checkpoint(ckpt, "cpu")
    data = load_day(day_file)
    L, _, _ = full_replay(memory, gnn, lp, hp, data, deleted, ev, batch_size=bs)
    return ev, sorted(deleted), L


def cmd_local_vs_full(a):
    """Compare PROVEX deletion (functional replay + no-backfill) with a full re-execution of the
    detector with the same events removed, on real alerts."""
    import multiprocessing as mp
    from scipy.stats import spearmanr
    set_seed(0)
    memory, gnn, lp, hp = load_checkpoint(a.ckpt, "cpu")
    data = load_day(a.day_file)
    recs = [json.loads(l) for l in open(a.explanations, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["alert_class"] == "window"][:a.max_alerts]
    targets = [r["event"]["event"] for r in recs]
    caps = KairosReplayer(memory, gnn, lp, hp).pass2(data, targets, history_batches=a.history_batches)
    rng = np.random.default_rng(0)
    cases, functional = [], {}
    for r in recs:
        ev = r["event"]["event"]; cap = caps[ev]; cap.tau = r["event"]["tau"]
        em = EventModel(memory, gnn, lp, cap)
        ex = next((x for x in r["explanations"] if x["method"] == "joint-cf"), None) or \
            next(x for x in r["explanations"] if x["method"] == "occlusion")
        ranked = [it["ev"] for it in ex["top_items"]]
        for k in (1, 5, 20):
            for kind, dels in (("top", ranked[:k]), ("random", rng.choice(em.items.numpy(), size=min(k, em.num_items), replace=False).tolist())):
                g = torch.ones(em.num_items, dtype=em.dtype)
                for d in dels:
                    if d in em.item_pos:
                        g[em.item_pos[d]] = 0
                Lf = float(em.loss(g, hard=True))
                functional[(ev, tuple(sorted(dels)))] = (Lf, cap.loss_batch, kind, k, cap.hist.exact_scope)
                cases.append((a.ckpt, a.day_file, ev, dels, 1024))
    log(f"local-vs-full: {len(cases)} full replays on {a.workers} workers")
    with mp.get_context("spawn").Pool(a.workers) as pool:
        full = pool.map(_lvf_case, cases)
    rows = []
    for ev, dels, Lt in full:
        Lf, L0, kind, k, exact = functional[(ev, tuple(dels))]
        rows.append({"event": ev, "k": k, "kind": kind, "L_orig": L0, "L_functional": Lf, "L_full_replay": Lt,
                     "exact_scope": exact})
    d_true = np.array([r["L_full_replay"] - r["L_orig"] for r in rows])
    d_f = np.array([r["L_functional"] - r["L_orig"] for r in rows])
    rel = np.abs(d_f - d_true) / np.maximum(np.abs(d_true), 1e-3)
    out = {"n": len(rows), "median_rel_err": float(np.median(rel)), "p90_rel_err": float(np.quantile(rel, 0.9)),
           "spearman": float(spearmanr(d_f, d_true).statistic), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    log(f"local-vs-full: median rel err {out['median_rel_err']:.3f}, Spearman {out['spearman']:.3f}")


# ---------------------------------------------------------------------------- graphmask-fit
def cmd_graphmask_fit(a):
    set_seed(a.seed)
    dev = torch.device(a.device)
    memory, gnn, lp, hp = load_checkpoint(a.ckpt, "cpu")
    data = load_day(a.day_file)
    losses, windows = KairosReplayer(memory, gnn, lp, hp).pass1(data)
    cands = []
    for w in windows:
        tau = A.window_threshold(losses, w)
        cands += A.anomalous_events(losses, w, tau)[:a.per_window]
    # sample across the whole day: the first max_events candidates all come from the early windows
    rng = np.random.default_rng(a.seed)
    targets = sorted(rng.choice(cands, size=min(a.max_events, len(cands)), replace=False).tolist()) if cands else []
    caps = KairosReplayer(memory, gnn, lp, hp).pass2(data, targets, history_batches=0)
    memory, gnn, lp = memory.to(dev), gnn.to(dev), lp.to(dev)
    models = [EventModel(memory, gnn, lp, caps[e], use_history=False, device=dev) for e in targets]
    gm = GraphMaskTC(gnn).to(dev)
    hist = gm.fit(models, epochs=a.epochs, seed=a.seed, log=lambda r: log(json.dumps(r)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gm.state_dict(), a.out)
    Path(str(a.out) + ".json").write_text(json.dumps({"history": hist, "final": hist[-1] if hist else {},
                                                      "n_events": len(models), "n_candidates": len(cands),
                                                      "day": a.day}))


def main(argv=None):
    p = argparse.ArgumentParser(prog="provex")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pass1"); s.add_argument("--ckpt", required=True); s.add_argument("--day-file", required=True)
    s.add_argument("--day", required=True); s.add_argument("--out", required=True)
    s.add_argument("--batch-size", type=int, default=1024); s.add_argument("--window-ns", type=int, default=15 * 60 * 10**9)
    s.set_defaults(fn=cmd_pass1)
    s = sub.add_parser("parity"); s.add_argument("--pass1", required=True); s.add_argument("--kairos-windows", required=True)
    s.add_argument("--day", required=True); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_parity)
    s = sub.add_parser("explain")
    for k in ("--ckpt", "--day-file", "--day", "--alerts", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--names"); s.add_argument("--gt-dir"); s.add_argument("--pass1"); s.add_argument("--graphmask")
    s.add_argument("--methods", default="joint,joint-cf,joint-edges,variational,gnnx-anomaly,gnnx-v1,baselines")
    s.add_argument("--seeds", default="0"); s.add_argument("--device", default="cpu"); s.add_argument("--dtype", default="float32")
    s.add_argument("--top-n", type=int, default=25); s.add_argument("--sample-n", type=int, default=25)
    s.add_argument("--max-events", type=int, default=0); s.add_argument("--history-batches", type=int, default=64)
    s.add_argument("--full-scores", action="store_true"); s.add_argument("--calibration", action="store_true")
    s.add_argument("--calibration-n", type=int, default=10); s.add_argument("--shard", default="", help="i/n")
    s.add_argument("--subsample", type=int, default=0, help="seeded random subset of the targets (0 = all)")
    s.add_argument("--budget", default="full", choices=sorted(BUDGETS), help="per-method compute budget")
    s.add_argument("--events", default="", help="only these alert events: '1,2,3' or '@file' with one id per line")
    s.add_argument("--only-class", default="", help="restrict to alert class (e.g. window), for E6 subsets")
    s.set_defaults(fn=cmd_explain)
    s = sub.add_parser("graphmask-fit")
    for k in ("--ckpt", "--day-file", "--day", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--device", default="cpu"); s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--per-window", type=int, default=20); s.add_argument("--max-events", type=int, default=400)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_graphmask_fit)
    s = sub.add_parser("local-vs-full")
    for k in ("--ckpt", "--day-file", "--explanations", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--max-alerts", type=int, default=30); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--history-batches", type=int, default=64); s.set_defaults(fn=cmd_local_vs_full)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()