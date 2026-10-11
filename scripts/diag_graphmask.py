"""Diagnose GraphMask-TC training on the validation day (a check, not part of the paper pipeline).

    python scripts/diag_graphmask.py --ckpt CKPT --day-file graph_4_5.TemporalData.simple --out DIR \
        [--configs tag:lam0:init_bias:lo:hi:dual_lr,...]

1. exactness: with every gate open, the replacement hook must reproduce the unmasked loss;
2. coverage: how many training events have no neighbour edges (empty gate sets);
3. training curves per optimiser setting. The deviation is measured on samples (what the Lagrangian
   sees) and with deterministic gates (mean > 0.5, what explain() uses).
Captures are cached in DIR/captures.pt, so re-runs skip the day replay.
Finding (2026-10-10): with lam0=1 and unbounded log-alpha, every gate closes in the first epoch and
never reopens; closing all edges costs KAIROS only ~9% of its loss.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provex.data import load_day  # noqa: E402
from provex.explain.event_model import EventModel  # noqa: E402
from provex.explain.graphmask_tc import GraphMaskTC  # noqa: E402
from provex.explain.joint_mask import expected_l0, hard_concrete_sample  # noqa: E402
from provex.model import load_checkpoint  # noqa: E402
from provex.replay import alerts as A  # noqa: E402
from provex.replay.kairos_replay import KairosReplayer  # noqa: E402


def log(out, row):
    line = json.dumps(row)
    print(line, flush=True)
    with open(out, "a") as f:
        f.write(line + "\n")


def train(gm, models, recs, L_full, epochs, lr, dual_lr, out, tag):
    opt = torch.optim.Adam(gm.parameters(), lr=lr)
    for ep in range(epochs):
        t0 = time.perf_counter()
        devs, l0s, below1, amean = [], [], [], []
        for em, rec, Lf in zip(models, recs, L_full):
            opt.zero_grad()
            la = gm.log_alpha(rec)
            ne = [a for a in la if a.numel()]
            zs = [hard_concrete_sample(a) for a in la]
            L = gm._run(em, [gm._replace_hook(l, z) for l, z in enumerate(zs)])
            dev = (L - Lf).abs() / max(Lf, 1.0)
            l0 = sum(expected_l0(a).mean() for a in ne) / len(ne)
            (l0 + gm.lam * F.softplus(dev - gm.allowance)).backward()
            opt.step()
            gm.lam = max(0.0, gm.lam + dual_lr * (float(dev) - gm.allowance))
            devs.append(float(dev)); l0s.append(float(l0))
            below1 += [float((z < 1).float().mean()) for z in zs if z.numel()]
            amean += [float(a.mean()) for a in ne]
        row = {"tag": tag, "epoch": ep, "dev_sample_mean": float(np.mean(devs)), "l0_mean": float(np.mean(l0s)),
               "frac_sampled_below_1": float(np.mean(below1)), "log_alpha_mean": float(np.mean(amean)),
               "lambda": gm.lam, "epoch_s": round(time.perf_counter() - t0, 1)}
        row.update(gm.evaluate(models, recs, L_full))
        log(out, row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--day-file", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--configs", default="role_lam10_d005:10:3:-4:5:0.05,role_lam10_d05:10:3:-4:5:0.5,"
                                         "role_lam50_d005:50:3:-4:5:0.05")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = str(Path(a.out) / "diag_graphmask.jsonl")
    cache = Path(a.out) / "captures.pt"
    torch.manual_seed(0)
    memory, gnn, lp, hp = load_checkpoint(a.ckpt, "cpu")
    if cache.exists():
        targets, caps = torch.load(cache, weights_only=False)
    else:
        t0 = time.perf_counter()
        data = load_day(a.day_file)
        losses, windows = KairosReplayer(memory, gnn, lp, hp).pass1(data)
        cands, cand_w = [], []
        for wi, w in enumerate(windows):
            tau = A.window_threshold(losses, w)
            evs = A.anomalous_events(losses, w, tau)[:20]
            cands += evs; cand_w += [wi] * len(evs)
        rng = np.random.default_rng(0)
        targets = sorted(rng.choice(cands, size=min(a.n_events, len(cands)), replace=False).tolist())
        caps = KairosReplayer(memory, gnn, lp, hp).pass2(data, targets, history_batches=0)
        torch.save((targets, caps), cache)
        log(out, {"stage": "setup", "n_windows": len(windows), "n_candidates": len(cands),
                  "windows_with_candidates": len(set(cand_w)), "windows_in_first_400": len(set(cand_w[:400])),
                  "n_events": len(targets), "setup_s": round(time.perf_counter() - t0, 1)})
    models = [EventModel(memory, gnn, lp, caps[e], use_history=False) for e in targets]
    probe = GraphMaskTC(gnn)
    recs = [probe.record(em) for em in models]
    keep = [i for i, r in enumerate(recs) if any(r[l]["q"].shape[0] for l in r)]
    models, recs = [models[i] for i in keep], [recs[i] for i in keep]
    L_full = [float(probe._run(em, [None, None])) for em in models]
    L_open = [float(probe._run(em, [probe._replace_hook(l, torch.ones(rec[l]["q"].shape[0])) for l in range(2)]))
              for em, rec in zip(models, recs)]
    log(out, {"stage": "checks", "max_abs_open_vs_full": max(abs(x - y) for x, y in zip(L_open, L_full)),
              "events_with_edges": len(models), "events_without_edges": len(targets) - len(models)})
    for spec in a.configs.split(","):
        tag, lam0, bias, lo, hi, dual_lr = spec.split(":")
        torch.manual_seed(0)
        gm = GraphMaskTC(gnn, lam0=float(lam0), init_bias=float(bias), bounds=(float(lo), float(hi)))
        log(out, {"tag": tag, "epoch": -1, **gm.evaluate(models, recs, L_full)})
        train(gm, models, recs, L_full, a.epochs, a.lr, float(dual_lr), out, tag)


if __name__ == "__main__":
    main()