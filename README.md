# PROVEX

PROVEX explains the alerts of [KAIROS](https://github.com/ProvenanceAnalytics/kairos), a temporal graph network (TGN) intrusion detector for system provenance graphs, on DARPA TC CADETS E3, for analysts in a security operations centre (SOC). It asks what a *faithful* explanation of a learned provenance detector can tell an analyst, and how much it reduces the work of deciding on an alert. The study is organised around NIST's four principles of explainable AI and the analyst tiers reported in *Too Much to Trust?* (arXiv:2503.02065v2).

> **About v1.** The first version (arXiv:2512.18199v1, tag `v1.0-arxiv-v1`) had implementation errors that affected its results. v2 corrects them and re-measures everything; [ERRATA.md](ERRATA.md) has the details.

## What v2 does

- **Replays KAIROS exactly.** `provex.replay` reproduces KAIROS's per-event scores, windows and alert queues (batch size 1024, memory reset per day, 15-minute windows). Explanations are of the scores KAIROS alerts on; nothing is re-scored.
- **Tier 1: an alert story.** For each alert, PROVEX groups the flagged events into actions (subject, event type, object), attaches facts an analyst can act on (rare action, new today, external IP new today, new program), orders them for triage and states the story's limits.
- **Tier 2: why this event.** Anomaly-faithful *preserve* and *counterfactual* explanations in loss space, checked by deletion on the replay; a variational explainer with calibrated inclusion probabilities; GraphMask for TransformerConv; GNNExplainer with our objective and with the v1 objective; MCTS (T-GNNExplainer-style), occlusion, integrated gradients, gradient, attention, recency and random baselines on the same items. A differentiable replay of the TGN memory lets explanations also gate past events in memory history.
- **Tier 3: internals and limits.** Alert class (rare action vs context-dependent), channel scores, alert margin, uncertainty, and flags for truncated history closures.

## Key results (paper v2)

- **Most alert events are about the action itself.** 90% of the events in KAIROS's alerts exceed the threshold from the event type alone. For them no subgraph is a cause, and five *random* edges pass the usual sufficiency test 98% of the time.
- **The story cuts the decision burden.** The two alerts go from 12,829 and 26,819 flagged events to 35 and 85 actions, with an attacker action first in both. A simulated analyst escalates 8 of 20 alerted windows (7 with attacker activity) and misses none. Novelty facts alone, without the detector, would flag 288 of 706 test windows.
- **The cheapest attribution is faithful; standard GNN tools are not.** For context-dependent alert events a plain gradient is as faithful as search-based explainers. GNNExplainer, GraphMask and attention are no better than random under real deletion, and the v1 objective is inverted.
- **Memory history adds cost, not faithfulness.** KAIROS's TGN memory is nearly empty, and the evidence lies in recent neighbour edges.
- **Limits.** KAIROS detects 2 of the 4 attacks under its own alert rule, so the other two have nothing to explain. PROVEX's deletions agree with a full re-execution of the detector on the alert decision in 87% of tested cases.

## Repository layout

| Path | Contents |
|---|---|
| `provex/` | The v2 package:<br>`replay/` (exact KAIROS replay, alerts, event captures)<br>`explain/` (explainers)<br>`evaluate/` (metrics, calibration, local-vs-full replay)<br>`pipeline.py` (CLI) |
| `kairos/` | A patch series over upstream KAIROS at a pinned commit, plus `setup_upstream.sh`, which fetches upstream, verifies it and applies the patches. Upstream ships without a license, so this repository contains only the patches. |
| `ground_truth/cadets_e3/` | Node-level labels from PIDSMaker/ORTHRUS (Apache-2.0) in `orthrus/`, and DARPA's attack indicators and steps in `darpa/`. |
| `scripts/` | `run_experiments.sh` (every experiment stage, including `xai` and `tables`), the analyses (`analyze_*.py`, `evaluate_stories.py`, `make_alert_stories.py`, `kairos_investigation.py`), the paper generators (`make_tables.py`, `make_xai_numbers.py`, `make_story_tex.py`) and diagnostics. |
| `tests/` | A pytest suite on a synthetic float64 model: replay exactness, deletion exactness against full re-execution, explainer validity, model state after explainers, and the end-to-end CLI and analyses. |
| `infra/gcp/` | How the experiments were run on Google Cloud, with cost guardrails: a job runner that stops the VM, an idle watchdog and a budget kill switch. |
| `paper/` | LaTeX for v2 (`provex_v2.tex`, run metadata in `meta.json`) and v1 (`report.tex`). Generated numbers and tables go in `paper/generated/`. |

## Quick start

You need Python 3.10+, PyTorch 2.5.1, PyTorch Geometric 2.7.0 and torch-scatter 2.1.2. CPU wheels are enough; all replays run on CPU because CUDA scatter tie-breaking is nondeterministic.

```bash
pip install torch==2.5.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install -r requirements-v2.txt
python -m pytest -q tests
```

The tests take about ten minutes on a 4-vCPU machine. The same environment is available as a container: `docker build -t provex:2 .` then `docker run --rm provex:2` runs the tests, and the `Dockerfile` header shows how to mount data for the experiments.

### Reproducing the paper

1. **Data.** The CADETS E3 day graphs (`graph_4_*.TemporalData.simple`) and `node_mapping.json` ship in `artifact.tar.gz` (Git LFS). To rebuild them from the raw DARPA data, use KAIROS's database stages: `kairos/cadets_e3/run_pipeline.sh --full-rebuild`.
2. **Detector.** Run `bash kairos/setup_upstream.sh`. Then train and test three seeds with `kairos/cadets_e3/run_pipeline.sh --seed N` for N = 0, 1, 2.
3. **Replay and alert anatomy.** `bash scripts/run_experiments.sh detect`, `parity`, `gmfit`, then `explain DAY W` with `METHODS=none` for the channel decomposition of every alert event (stage 1).
4. **Technique comparison.** `scripts/select_compare_events.py` picks the events. Setting A is `HIST_BATCHES=0 BUDGET=bounded EVENTS=@results/compare/events/A_DAY.txt EXPL_OUT=results/compare/A bash scripts/run_experiments.sh explain DAY 4`. Setting B uses `HIST_BATCHES=16`. Then run `e6 DAY` (stability, calibration) and `e7 DAY` (deletions against full re-execution).
5. **Numbers and paper.** `bash scripts/run_experiments.sh tables` and `bash scripts/run_experiments.sh xai`, then `latexmk -pdf paper/provex_v2.tex`.

`infra/gcp/` contains the scripts used to run all of this on Google Cloud: one CPU VM for replay and explanations, and one L4 GPU VM for KAIROS training. The whole study cost under $10 of compute.

## Limitations

- **No user study.** The evaluation is functionally grounded: items to read, ranks and a simulated analyst stand in for an analyst's work.
- **One detector, one host, two alerts.** We study KAIROS on CADETS E3, where KAIROS raises two alerts. Conclusions may not transfer.
- **Deletion is an intervention on the replay.** History closures are capped and the neighbour loader is not back-filled, so deletions approximate a full re-execution; the paper measures the gap.
- **Ground truth is coarse.** Labels are mapped by entity name, as KAIROS keys nodes (all `nginx` processes share one node), and the public labels contain errors that the paper documents.

## Citation

```bibtex
@misc{provex,
  title         = {{PROVEX}: Faithful, Tiered Explanations of Provenance-Based Intrusion Alerts for {SOC} Analysts},
  author        = {Dhanuka, Devang and Rastogi, Nidhi},
  year          = {2025},
  eprint        = {2512.18199},
  archivePrefix = {arXiv}
}
```

v1 ("PROVEX: Enhancing SOC Analyst Trust with Explainable Provenance-Based IDS") remains available at tag `v1.0-arxiv-v1`, exactly as published.