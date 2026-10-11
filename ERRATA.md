# Errata: PROVEX v1 → v2

This file lists the problems found in PROVEX v1 (arXiv:2512.18199v1; this repo at tag `v1.0-arxiv-v1`, commit `07e9bc6`) and what v2 changes. The independent evaluation that found them is summarised in the pinned "v1 errata" issue. v1 stays tagged so it can be reproduced exactly as published.

## Results in the v1 paper that are withdrawn

| v1 item | Problem | Status in v2 |
|---|---|---|
| Table II (top GraphMask edges `nginx→bash`, `bash→/tmp/lib.so`, `bash→78.205.235.65:80`) | These edges do not exist in the CADETS E3 data. There are no `nginx↔bash` or `bash↔78.205.235.65` events on 2018-04-06, and `/tmp/lib.so` is not a node. | Withdrawn. Replaced by tables generated from `results/`. |
| Table IV (edge ablation, "−78%", "Alert: No") | No ablation or alert code existed, and the edges do not exist. | Withdrawn. Replaced by measured deletion and insertion curves and alert-flip rates. |
| Table III (comprehensiveness 0.82, sufficiency 0.15) | Incompatible with the code. The metrics were computed on p(true type), which is about 1e-11 for alerting events, so they are about 0. The paper also read their sign backwards. | Withdrawn. Replaced by anomaly-term metrics. |
| Case study ("bash reads /etc/passwd", "bash → attacker.com:8010") | Not in the data. The real chain on 04-06 is attacker IPs → nginx → `/tmp/vUgefal` → `/etc/passwd`, with C2 traffic to 139.123.0.113. | Rewritten from generated outputs. |
| Appendix JSON | Placeholder values. | Replaced with a real output excerpt. |

## Code defects in v1 and their v2 fixes

| ID | v1 defect | v2 fix |
|---|---|---|
| C1 | Explained edges reported per-event local indices as global node IDs, so every human-readable entity was wrong. | Every item is keyed by absolute event index and global node ID (`provex/replay/capture.py`), and a test checks this. |
| C2 | All explainers minimised CE(f(masked), y_true), which explains why an event looks *normal*. | Anomaly objectives are now preserve and counterfactual (`provex/explain/objectives.py`). |
| C3 | "GraphMask" was a feature-scaling stand-in. Its documented hyperparameters were never passed in. | A real GraphMask for TransformerConv: amortised hard-concrete gates, learned baselines and a Lagrangian (`provex/explain/graphmask_tc.py`). |
| C4 | "VA-TGExplainer" used no time information and discarded its variance. | A variational explainer with a temporal prior, which exports its uncertainty, plus an exact-enumeration calibration check (`provex/explain/variational.py`, `provex/evaluate/calibration.py`). |
| H1 | Scores were recomputed with batch size 1 and a 2 h warm-up, so they differed from KAIROS's (top-50 overlap 0). | KAIROS-exact replay: batch 1024, reset per day, upstream windows (`provex/replay/kairos_replay.py`). |
| H2 | TGN memory (the event history) was never explained. | A differentiable replay of TGNMemory gates past events. On a test fixture, 100 deletions match a full PyG replay exactly (`provex/explain/functional_tgn.py`). |
| H3 | The attack window was hard-coded from ground truth, and node selection compared a cumulative sum against a per-event threshold. | Explanations are driven by KAIROS's own alerts, and nodes are ranked deterministically (`provex/replay/alerts.py`). |
| H4 | The paper describes features that have no code: alerting, ablation, parallel windows, the "60%" memory claim. | Implemented and measured, or removed from the paper. |
| M1–M4 | Unpinned dependencies, `torch.load` failing on torch ≥ 2.6, no tests or seeds, a model forked from KAIROS without documentation, a misdescribed label field. | Pinned `requirements-v2.txt`, a test suite, seeds, upstream KAIROS rebuilt as a documented patch series (`kairos/`), and an accurate schema. |

## Removed in v2

These v1 components are not part of v2. They remain available at tag `v1.0-arxiv-v1`.

| v1 component | Why it was removed |
|---|---|
| `explanations/` (v1 explainers and window analysis), `config.py`, `kairos_tgnn_model.py`, `kairos_utils.py` | Replaced by the `provex/` package and an upstream KAIROS patch series. |
| `reporting/` (report builder, Streamlit dashboard, optional GPT narrative) | Built on the v1 output format. The v2 paper does not use or describe them. |
| `report.pdf` | The v1 paper. It stays on arXiv as v1. |
| `requirements.txt`, `environment.yml`, `.env.example` | v1 environment. v2 pins `requirements-v2.txt`. |
