# PROVEX

PROVEX explains the alerts of [KAIROS](https://github.com/ProvenanceAnalytics/kairos), a temporal graph network (TGN) intrusion detector for system provenance graphs, on DARPA TC CADETS E3. For an alert event it asks two questions. Which past events make KAIROS call this event anomalous? And would removing them clear the alert?

> **v1 errata.** The first version (arXiv:2512.18199v1; tag `v1.0-arxiv-v1`) had defects that invalidate its results:
> - explained edges carried local indices instead of node IDs;
> - every explainer explained why an event looks *normal*;
> - Tables II–IV and the case study cite edges that do not exist in CADETS E3.
>
> Those results are withdrawn. [ERRATA.md](ERRATA.md) lists every problem and what v2 changes. v2 re-measures everything, and every number in the v2 paper is generated from `results/` by `scripts/make_tables.py` and `scripts/make_case_studies.py`.

## What v2 does

- **Explains the scores KAIROS alerts on.** `provex.replay` replays KAIROS exactly (batch size 1024, memory reset per day, 15-minute windows) and reads its alert queues. It explains the alert events themselves; it does not re-score them.
- **Explains the anomaly, not normality.** The objectives are in loss space.
  - A *preserve* explanation is a minimal set of past events that keeps the score above the window threshold τ.
  - A *counterfactual* explanation is a minimal set whose removal drops the score below τ.
  - Every explanation is checked by exact deletion and repaired if needed.
- **Covers memory as well as edges.** A differentiable replay of TGN memory updates gates individual past events (*history*), not only the neighbour edges of the current batch. Its deletions are tested against full re-execution of the detector.
- **Reports uncertainty.** A variational explainer with a temporal prior exports inclusion probabilities. They are checked against exact posterior enumeration.
- **Sorts alerts by cause.** Some alerts stay anomalous with the whole window removed. PROVEX labels those as prior-driven or long-memory-driven instead of returning a misleading mask.
- **Compares against baselines on the same items:**
  - GraphMask for TransformerConv;
  - GNNExplainer, with our objective and with the v1 objective;
  - occlusion, integrated gradients, gradient, attention, recency and random.

## Repository layout

| Path | Contents |
|---|---|
| `provex/` | The v2 package:<br>`replay/` (exact KAIROS replay, alerts, event captures)<br>`explain/` (explainers)<br>`evaluate/` (metrics, calibration, local-vs-full replay)<br>`pipeline.py` (CLI) |
| `kairos/` | A patch series over upstream KAIROS at a pinned commit, plus `setup_upstream.sh`, which fetches upstream, verifies it and applies the patches. Upstream ships without a license, so this repository contains only the patches. |
| `ground_truth/cadets_e3/orthrus/` | Node-level ground truth from PIDSMaker/ORTHRUS (Apache-2.0). |
| `scripts/` | `run_experiments.sh` (every experiment stage), `make_tables.py` and `make_case_studies.py` (paper numbers), and diagnostics. |
| `tests/` | A pytest suite on a synthetic float64 model. It covers replay exactness, deletion exactness against full re-execution, explainer validity and the end-to-end CLI. |
| `infra/gcp/` | How the experiments were run on Google Cloud, with cost guardrails: a job runner that stops the VM, an idle watchdog and a budget kill switch. |
| `paper/` | LaTeX for v2 (`provex_v2.tex`) and v1 (`report.tex`). Generated tables go in `paper/generated/`. |

## Quick start

You need Python 3.10+, PyTorch 2.5.1, PyTorch Geometric 2.7.0 and torch-scatter 2.1.2. CPU wheels are enough; all replays run on CPU because CUDA scatter tie-breaking is nondeterministic.

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install -r requirements-v2.txt
python -m pytest -q tests
```

The tests take a few minutes on a laptop CPU. The same environment is available as a container: `docker build -t provex:2 .` then `docker run --rm provex:2` runs the tests, and the `Dockerfile` header shows how to mount data for the experiments.

### Reproducing the paper

1. **Data.** The CADETS E3 day graphs (`graph_4_*.TemporalData.simple`) and `node_mapping.json` ship in `artifact.tar.gz` (Git LFS). To rebuild them from the raw DARPA data, use KAIROS's database stages: `kairos/cadets_e3/run_pipeline.sh --full-rebuild`.
2. **Detector.** Run `bash kairos/setup_upstream.sh`. Then train and test three seeds with `kairos/cadets_e3/run_pipeline.sh --seed N` for N = 0, 1, 2.
3. **Explanations and tables.** Run `bash scripts/run_experiments.sh <stage>` for each stage in order: `detect`, `parity`, `gmfit`, `explain DAY`, `e6 DAY`, `e7 DAY`, `tables`. The script documents every command and output path.

`infra/gcp/` contains the scripts used to run all of this on Google Cloud: one CPU VM for replay and explanations, and one L4 GPU VM for KAIROS training.

## Limitations

- **One detector and one dataset.** We study KAIROS on CADETS E3 only. Conclusions may not transfer to other detectors or datasets.
- **History is truncated.** History gating covers a window of recent batches and a capped influence closure. Truncation is flagged per event, and the paper measures its error against full re-execution.
- **Ground truth is coarse.** It is mapped by entity name, as KAIROS keys nodes. For example, all `nginx` processes share one node.
- **No user study yet.** Explanations are evaluated by faithfulness, stability, calibration and agreement with ground truth.

## Citation

```bibtex
@misc{provex,
  title         = {{PROVEX}: Faithful, Temporal Explanations for a Provenance-Based Intrusion Detector},
  author        = {Dhanuka, Devang and Rastogi, Nidhi},
  year          = {2025},
  eprint        = {2512.18199},
  archivePrefix = {arXiv}
}
```

v1 ("PROVEX: Enhancing SOC Analyst Trust with Explainable Provenance-Based IDS") remains available at tag `v1.0-arxiv-v1`, exactly as published.