# PROVEX

**Prov**enance Graph **Ex**plainability Framework — an XAI pipeline for Temporal Graph Network (TGN) intrusion detection on the DARPA CADETS provenance graph dataset. PROVEX generates graph-based explanations using multiple methods (GraphMask, VA-TG, GNNExplainer) and produces analyst-friendly reports with optional GPT narratives.

📄 **Research Paper**: [report.pdf](report.pdf)

---

## Architecture Overview

### Target Model (`model.py`)

PROVEX explains predictions from a **Temporal Graph Network (TGN)** with the following architecture:

- **Graph Encoder**: 2-layer Graph Transformer (`TransformerConv`)
  - Layer 1: 8 attention heads
  - Layer 2: 1 attention head
  - Edge Features: Concatenation of relative time encoding (`rel_t_enc`) and raw message features (`msg`)
- **Link Predictor**: 4-layer MLP (`Linear` → `Tanh` → `Dropout`) predicting edge labels from source/destination embeddings
- **Memory**: TGN memory module with `LastAggregator` for incoming messages
- **Dataset**: Optimized for DARPA CADETS provenance graphs

### Explainers

| Explainer        | Goal                                                     | Key Parameters                                    |
| ---------------- | -------------------------------------------------------- | ------------------------------------------------- |
| **GraphMask**    | Learn sparse binary edge mask preserving predictions     | 200 epochs, lr=0.01, sparsity/entropy weight=1e-3 |
| **VA-TG**        | Variational mask distribution capturing uncertainty      | 200 epochs, 10 samples/step, KL weight=1e-3       |
| **GNNExplainer** | Compact subgraph + feature subset via Mutual Information | 200 epochs, lr=0.01                               |

---

## 1) Prerequisites

> **PROVEX is a post-training explainability framework.** You must have already trained a TGN model using the Kairos pipeline.

### Required Artifacts from Kairos

Before running PROVEX, ensure you have the following artifacts from the Kairos training pipeline:

| Artifact             | Path (default)                            | Description                                                     |
| -------------------- | ----------------------------------------- | --------------------------------------------------------------- |
| **Trained Model**    | `artifact/models/`                        | TGN checkpoint (`.pt` file)                                     |
| **Graph Embeddings** | `artifact/graph_embeddings/`              | `*.TemporalData.simple` files with pre-computed node embeddings |
| **Node Mapping**     | `artifact/explanations/node_mapping.json` | _(Optional)_ Node ID → human-readable labels                    |

### Directory Structure

```
artifact/
├── models/
│   └── tgn_checkpoint.pt          # Trained TGN model
├── graph_embeddings/
│   └── *.TemporalData.simple      # Graph windows with embeddings
└── explanations/
    └── node_mapping.json          # Optional: exported from DB
```

> ### **First-Time Clone (Git LFS)**
>
> The `artifact.tar.gz` file is stored using [Git LFS](https://git-lfs.com/). After cloning, pull the LFS files:
>
> ```bash
> # Install Git LFS (if not already installed)
> brew install git-lfs   # macOS
> # apt install git-lfs  # Ubuntu/Debian
>
> # Initialize LFS and pull large files
> git lfs install
> git lfs pull
> ```

> ### **Quick Start**:
>
> Pre-computed artifacts are included. Simply extract:
>
> ```bash
> tar -xzf artifact.tar.gz
> ```

---

## 2) Setup

### Environment

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate provex

# Option B: pip
pip install -r requirements.txt
# + install PyTorch/PyG for your platform (see requirements.txt comments)
```

### Configuration

Edit `config.py`:

```python
ARTIFACT_DIR = "./artifact/"                          # Root artifact path
GRAPHS_DIR   = ARTIFACT_DIR + "graph_embeddings/"     # Graph embeddings
MODELS_DIR   = ARTIFACT_DIR + "models/"               # Model checkpoints

# Optional: PostgreSQL for node label export
DATABASE = 'tc_cadet_dataset_db'
HOST = None
USER = 'postgres'
PASSWORD = 'password'
PORT = '5432'
```

### OpenAI API Key

To enable GPT-powered narrative summaries in reports, set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

> Without this key, reports will still be generated but GPT narrative sections will be skipped.

---

## 3) Quick Start

```bash
make explain    # Run explanations pipeline
make report     # Generate Markdown report
make dashboard  # Launch Streamlit dashboard
```

Artifacts land under `artifact/explanations/`.

---

## 4) Explanations Pipeline

The PROVEX explanations pipeline (`explanations/`) orchestrates:

1. **Window Collection**: Streams temporal graph with 2-hour warmup for TGN memory priming
2. **Thresholding**: Dynamic threshold = μ + 1.5σ of window losses
3. **Event Filtering**: High-loss events (capped at 50 per window)
4. **GraphMask Pass**: Edge importance aggregation across events
5. **Node Selection**: Cumulative loss ≥ threshold (fallback: top 20)
6. **Detailed Explanations**: Per-node GNNExplainer + VA-TGExplainer analysis

### Optional: Export Node Mapping

The pipeline auto-creates node labels if missing. Manual export:

```bash
make node_mapping
```

### Run Explanations

> ⚠️ **Note**: Explanation generation is computationally intensive. Expect **2-3 hours** on a dedicated GPU, longer on CPU.

```bash
make explain
```

### Outputs

- `artifact/explanations/<window>_explanations.json` — Per-window explanation data
- `artifact/explanations/graph_<label>_summary.json` — Global summary
- `artifact/explanations/temporal_explanations.log` — Processing log

---

## 5) Reporting + Dashboard

### Generate Report

Creates Markdown report with optional GPT narrative:

```bash
make report
```

**GPT Sections**: "What happened?", "Who's involved?", "Why flagged?", "What's missing/risky?", "What next?"

### Launch Dashboard

```bash
make dashboard
```

**Dashboard Features**:

- Bar chart of top GraphMask edges (relation-colored)
- Bar chart of top nodes by average loss with threshold line
- Minute-binned event timeline
- GPT narrative sections aligned with relevant charts
- Mermaid.js graph of top interactions

---

## 6) Explanation Metrics

PROVEX evaluates mask quality using:

| Metric                | Description                                                     | Better     |
| --------------------- | --------------------------------------------------------------- | ---------- |
| **Comprehensiveness** | prob_full − prob_drop (explanation contains all necessary info) | Higher     |
| **Sufficiency**       | prob_full − prob_keep (explanation alone is sufficient)         | Lower (≈0) |
| **Sparsity**          | 1 − (edges_kept / edges_total)                                  | Higher     |
| **Entropy**           | Mask ambiguity measure                                          | Lower      |

---

## 7) Key Data Structures

### Explanation JSON Schema

```json
{
  "window_path": "string",
  "threshold": "float",
  "metrics": { "..." },
  "graphmask": {
    "per_event": ["..."],
    "aggregate": [
      { "src": "int", "dst": "int", "relation": "string", "weight": "float", "count": "int", "timestamps": ["..."] }
    ]
  },
  "nodes": [
    {
      "node_id": "int",
      "score": "float (Total Loss)",
      "avg_score": "float",
      "max_event_loss": "float",
      "gnn": ["..."],
      "va_tg": { "events": ["..."], "aggregate": ["..."] }
    }
  ],
  "gpt_summary": "string (added by reporting)"
}
```

### Node Mapping

JSON mapping integer Node IDs to human-readable labels (e.g., `123: "Process: powershell.exe"`).

---

## 8) Logging & Progress

- Explanations log to `artifact/explanations/temporal_explanations.log`
- Console output uses concise prints and `tqdm` progress bars
- GPU memory checks printed only on OOM fallback or low-memory warnings

```bash
# Monitor long runs
tail -f artifact/explanations/temporal_explanations.log
```

---

## 9) Environment Variables

| Variable                    | Purpose                                                 |
| --------------------------- | ------------------------------------------------------- |
| `PROVEX_NODE_MAPPING_JSON`  | Node mapping JSON path (labels)                         |
| `OPENAI_API_KEY`            | Enables GPT narrative in reports                        |
| `PROVEX_CONTEXT_DEVICE`     | Tensor storage mode: `gpu`, `cpu`, `cpu_pin`, or `auto` |
| `PROVEX_EXPLAIN_WARMUP_SEC` | Warmup margin in seconds (default: 7200)                |
| `PROVEX_CONTEXT_GPU_BUFFER` | GPU buffer threshold in bytes (default: 512MB)          |
| `PROVEX_CUDA_LOG_EVERY`     | CUDA memory logging frequency (0 = disabled)            |

---

## 10) Notes

- **Threshold Semantics**: The event-loss threshold is derived from per-event losses. The dashboard's "Top nodes" chart shows average loss per node with a dashed vertical line indicating the event-loss threshold context.
- **Device Management**: GPU memory is checked before each explainer batch (~512MB free required). Falls back to CPU if insufficient.
- **Dashboard**: Visualizations use Plotly with Pandas for data aggregation and NetworkX for graph layout.
