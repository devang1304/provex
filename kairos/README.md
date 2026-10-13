# Upstream KAIROS on DARPA TC CADETS E3 (for PROVEX)

PROVEX explains KAIROS detections after the fact. This directory holds the **upstream** KAIROS pipeline (ProvenanceAnalytics/kairos, IEEE S&P'24) for CADETS E3, plus a small patch series. With it we can:

* retrain upstream KAIROS with the upstream configuration and fixed seeds,
* score every day from 2018-04-03 to 2018-04-13,
* evaluate all four CADETS E3 attacks (04-06, 04-11, 04-12, 04-13).

This is independent of the older fork in the repository root (`kairos_tgnn_model.py`) and of `artifact/models/models.pt`; neither is used here.

## Layout

```
kairos/
  UPSTREAM.md          upstream commit, URLs, blob hashes, license status
  upstream/            byte-exact upstream DARPA/CADETS_E3 (+ README_upstream.md, settings/)
  patches/             01..11, one concern each, applied in order to upstream/
  cadets_e3/           upstream/ + all patches = the code to run (+ node_gt.py, run_pipeline.sh)
ground_truth/cadets_e3/orthrus/
  node_Nginx_Backdoor_{06,11,12,13}.csv, readme.md, LICENSE, SOURCE.md   (PIDSMaker)
```

Upstream commit: `0e0b633beb46a1117c0a6d63be5d2481b59ac0dc`. Upstream has no license file; see `UPSTREAM.md`.

### Patches

| Patch | What it changes |
|---|---|
| `01-config-env` | `config.py` reads `PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE`, `KAIROS_ARTIFACT_DIR`, `KAIROS_RAW_DIR`, `KAIROS_GRAPHS_DIR`; defaults = upstream values |
| `02-seeds-determinism` | `kairos_utils.set_seed()` (random, numpy, torch, CUDA, cudnn deterministic, `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG`). Adds `--seed` to train/test/queues/evaluation/attack_investigation; a seeded run writes to `<artifact>/seed_<n>/` |
| `03-torch-load` | `weights_only=False` on every `torch.load` (torch >= 2.6) |
| `04-save-state-dict` | `train.py` also writes `models/models_state.pt` and `train_log.json`; the pickled `models/models.pt` is kept |
| `05-test-all-days` | `test.py` scores every existing day graph `graph_4_3..graph_4_13` (`config.test_graphs` or `--graphs`), one day at a time; window files unchanged |
| `06-eval-all-attacks` | Queues for `graph_4_5..graph_4_13`. Window-level ground truth for all four attacks, metrics per day, per attack and overall, plus the upstream protocol, written to `detection_metrics.json`; also writes `alerts.json` and `windows.json` |
| `07-node-level-gt` | new `node_gt.py`: PIDSMaker node ground truth -> KAIROS `index_id` |
| `08-fix-cal-anomaly-loss-zero-division` | **fix**: guards the division by zero in `cal_anomaly_loss` when a window has no edge above mean + 1.5*std (upstream would crash; other windows unchanged) |
| `09-no-db` | psycopg2 import optional; `test.py` reads node labels from `KAIROS_NODE_MAP_JSON` when set |
| `10-run-pipeline` | new `run_pipeline.sh` |
| `11-compat-pyg-seq-batches` | **compat fix**: `TemporalData.seq_batches()` no longer exists in torch_geometric >= 2.0.4 (upstream pins 2.0.0). Replaced by an identical helper; batches are unchanged |

To rebuild `cadets_e3/` from upstream and validate the series (e.g. on the VM):

```bash
REPO=$PWD                                  # repository root
tmp=$(mktemp -d)                           # must be outside any git work tree
cp "$REPO"/kairos/upstream/Makefile "$REPO"/kairos/upstream/*.py "$tmp"/
( cd "$tmp" && for p in "$REPO"/kairos/patches/*.patch; do git apply --check "$p" && git apply "$p" || exit 1; done )
diff -r "$tmp" "$REPO"/kairos/cadets_e3 && echo "patch series reproduces kairos/cadets_e3"
python -m py_compile "$REPO"/kairos/cadets_e3/*.py
```

## Environment

There are two options:

* **PROVEX environment (recommended)**: `environment.yml` in the repository root, i.e. Python 3.10, torch 2.2 + CUDA 12.1, PyG 2.7, torch-scatter/sparse, scikit-learn, networkx, xxhash, pytz, tqdm. Add `python-louvain` and `graphviz` (only for `attack_investigation.py`), `psycopg2` (only for the database path) and `gdown` (only for downloading raw data). Patches 03 and 11 make the upstream code run on this stack. The TGN API it uses (`TGNMemory`, `LastNeighborLoader`, `IdentityMessage`, `LastAggregator`) is unchanged in PyG 2.7.
* **Upstream environment**: `upstream/settings/environment-settings.md` + `requirements.txt` (Python 3.9, torch 1.13.1 + cu117, torch_geometric 2.0.0, scikit-learn 1.2.0, networkx 2.8.7, xxhash 3.2.0). Note that the pre-built graphs in `artifact.tar.gz` are loaded by PROVEX under PyG 2.7. They may not unpickle under PyG 2.0.0.

## Default path: no database (pre-built graphs)

The repository's `artifact.tar.gz` (Git LFS) already contains everything train and test need:

* `artifact/graph_embeddings/graph_4_2 .. graph_4_13.TemporalData.simple`: vectorized day graphs, `msg` dimension 39 = 16 (src node) + 7 (edge type) + 16 (dst node).
* `artifact/explanations/node_mapping.json`: `{"<index_id>": "<type>: <name>"}`, 268,242 entries (= upstream `node2id`, `max_node_num` 268,243 in `model.py`). Examples: `"subject: nginx"`, `"file: /tmp/vUgefal"`, `"netflow: 81.49.200.166:43530"`.

The database is not needed:

* `train.py` only loads `graph_4_2..graph_4_4`. Upstream `train.py` never opens the database; it only imported psycopg2 through `kairos_utils`, which patch 09 makes optional.
* `test.py` needs node labels; with patch 09 they come from `KAIROS_NODE_MAP_JSON`.
* `anomalous_queue_construction.py`, `evaluation.py` and `attack_investigation.py` only read the window files and queue lists.
* Only `create_database.py` and `embedding.py` need Postgres.

```bash
cd /path/to/provex                      # repository root, branch v2
git lfs pull                            # fetch artifact.tar.gz
tar -xzf artifact.tar.gz                # -> artifact/graph_embeddings, artifact/explanations, ...

export KAIROS_GRAPHS_DIR=$PWD/artifact/graph_embeddings/
export KAIROS_NODE_MAP_JSON=$PWD/artifact/explanations/node_mapping.json
export KAIROS_ARTIFACT_DIR=$PWD/artifact/kairos_upstream/     # inside artifact/ = git-ignored

for seed in 0 1 2; do
  bash kairos/cadets_e3/run_pipeline.sh --seed "$seed"        # train, test, queues, evaluation, node_gt
done
```

`node_gt` does not depend on the seed. Add `--skip node_gt` after the first run if you like.

Before the full run, check that the pre-built graphs are the upstream ones. After `test` on `graph_4_6`, `<artifact>/seed_<n>/graph_4_6/` must contain the four upstream ground-truth windows, starting with `2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt`. Window boundaries depend only on the event stream and `BATCH=1024`, not on the model. `evaluation.py` reports any missing upstream window under `ground_truth.Nginx_Backdoor_06.missing_windows` and `upstream_protocol.missing_gt_windows`. A quick check is `python test.py --seed 0 --graphs graph_4_6`, which needs a trained model.

## Optional: full rebuild from raw data + PostgreSQL

### Raw data (DARPA TC Engagement 3, CADETS)

The upstream KAIROS README does not link the raw data. It says the paper and supplementary material hold the dataset links, and its own Google Drive folder (`1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C`) only holds pretrained models. The official DARPA TC Engagement 3 release is on Google Drive. PIDSMaker's download script (`dataset_preprocessing/darpa_tc/scripts/download_cadets_e3.sh`, commit `438a05a`) uses the same folder.

* E3 data folder: https://drive.google.com/drive/folders/1fOCY3ERsEmXmvDekG-LUUSjfWs6TRdp- (subfolders `cadets`, `clearscope`, `fivedirections`, `theia`, `trace`)
* `cadets` folder: https://drive.google.com/drive/folders/179uDuz62Aw61Ehft6MoJCpPeBEz16VFy

KAIROS parses the **JSON** archives. Sizes are as Google Drive reports them on the download page; all files were last modified 2018-08-23.

| File | Drive file id | Size |
|---|---|---|
| `ta1-cadets-e3-official.json.tar.gz` | `1AcWrYiBmgAqp7DizclKJYYJJBQbnDMfb` | 559M |
| `ta1-cadets-e3-official-1.json.tar.gz` | `1XLCEhf5DR8xw3S-Fimcj32IKnfzHFPJW` | 906M |
| `ta1-cadets-e3-official-2.json.tar.gz` | `1EycO23tEvZVnN3VxOHZ7gdbSCwqEZTI1` | 342M |

The same folder also has the Avro binaries, which KAIROS does not need: `ta1-cadets-e3-official.bin.tar.gz` (`1DpyVpK9JWNlXzhlzzeuKO_gD8BSosn--`), `-1.bin.tar.gz` (`1sIpBepnKw0kz4yZviUD_dPZWz0EAfrAS`) and `-2.bin.tar.gz` (`159Enm1TChO9nUvUE2UhcAlZH5nv9CZ2e`).

After extraction, `create_database.py` expects exactly these 10 files in `$KAIROS_RAW_DIR`:

```
ta1-cadets-e3-official.json      ta1-cadets-e3-official.json.1    ta1-cadets-e3-official.json.2
ta1-cadets-e3-official-1.json    ta1-cadets-e3-official-1.json.1  ta1-cadets-e3-official-1.json.2
ta1-cadets-e3-official-1.json.3  ta1-cadets-e3-official-1.json.4
ta1-cadets-e3-official-2.json    ta1-cadets-e3-official-2.json.1
```

The extracted JSON is much larger than the 1.8 GB of archives (not measured here), so leave room for it and for the Postgres database.

With gdown:

```bash
pip install gdown
export KAIROS_RAW_DIR=/data/cadets_e3 && mkdir -p "$KAIROS_RAW_DIR" && cd "$KAIROS_RAW_DIR"
gdown --no-cookies 1AcWrYiBmgAqp7DizclKJYYJJBQbnDMfb   # ta1-cadets-e3-official.json.tar.gz   (559M)
gdown --no-cookies 1XLCEhf5DR8xw3S-Fimcj32IKnfzHFPJW   # ta1-cadets-e3-official-1.json.tar.gz (906M)
gdown --no-cookies 1EycO23tEvZVnN3VxOHZ7gdbSCwqEZTI1   # ta1-cadets-e3-official-2.json.tar.gz (342M)
# interrupted or quota-limited: re-run the same command with --continue
for f in ta1-cadets-e3-official*.json.tar.gz; do tar -xzf "$f"; done
ls -la ta1-cadets-e3-official*.json*
```

With rclone, using a Google Drive remote named `gdrive` configured with `rclone config`. This is untested here.

```bash
for id in 1AcWrYiBmgAqp7DizclKJYYJJBQbnDMfb 1XLCEhf5DR8xw3S-Fimcj32IKnfzHFPJW 1EycO23tEvZVnN3VxOHZ7gdbSCwqEZTI1; do
  rclone backend copyid gdrive: "$id" "$KAIROS_RAW_DIR/" -P
done
# or the whole folder (JSON archives only):
rclone copy gdrive: "$KAIROS_RAW_DIR" --drive-root-folder-id 179uDuz62Aw61Ehft6MoJCpPeBEz16VFy --include "*.json.tar.gz" -P
```

### PostgreSQL

The schema follows `upstream/settings/database.md` (section "CADETS E3"). Database `tc_cadet_dataset_db` holds these tables:

| Table | Columns |
|---|---|
| `event_table` | `src_node varchar, src_index_id varchar, operation varchar, dst_node varchar, dst_index_id varchar, timestamp_rec bigint, _id serial` (+ unique index on `_id`) |
| `file_node_table` | `node_uuid varchar, hash_id varchar, path varchar`, PK (`node_uuid`, `hash_id`) |
| `netflow_node_table` | `node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port` (varchar), PK (`node_uuid`, `hash_id`) |
| `subject_node_table` | `node_uuid varchar, hash_id varchar, exec varchar` |
| `node2id` | `hash_id varchar PK, node_type varchar, msg varchar, index_id bigint` |

`hash_id` is the sha256 of the node name: exec for subjects, path for files, `"srcaddr,srcport,dstaddr,dstport"` for netflows. `node2id.msg` holds the exec, the path, or `dstaddr:dstport` for netflows.

Upstream expects about 155,322 netflow, 224,146 subject and 234,245 file rows, 268,242 `node2id` rows and 29,727,441 events. `model.py` hard-codes `max_node_num = 268243` (`node2id` rows + 1); change it if your database differs.

```bash
sudo apt-get install -y postgresql
sudo -u postgres psql -c "alter user postgres with password 'postgres';"   # upstream default credentials
export PGHOST=localhost PGPORT=5432 PGUSER=postgres PGPASSWORD=postgres PGDATABASE=tc_cadet_dataset_db
bash kairos/cadets_e3/run_pipeline.sh --only db_setup           # create database + tables (idempotent)
# with peer authentication instead of a password: PSQL="sudo -u postgres psql" bash ... --only db_setup
```

The upstream default `PGHOST=/var/run/postgresql/` (Unix socket) also works when the connecting user may use peer authentication. See the troubleshooting section in `upstream/settings/environment-settings.md`.

### Full rebuild run

```bash
export KAIROS_RAW_DIR=/data/cadets_e3
export KAIROS_ARTIFACT_DIR=$PWD/artifact/kairos_upstream_rebuild/
unset KAIROS_GRAPHS_DIR KAIROS_NODE_MAP_JSON     # graphs -> <artifact>/graphs/, labels from node2id
bash kairos/cadets_e3/run_pipeline.sh --full-rebuild --seed 0
# later seeds reuse the same database and graphs:
bash kairos/cadets_e3/run_pipeline.sh --seed 1 --skip node_gt   # with KAIROS_GRAPHS_DIR=<artifact>/graphs/
```

`create_database` refuses to run when `node2id` is not empty, because it would insert duplicates; set `FORCE_CREATE_DATABASE=1` to override. To get a node map for later runs without the database, export `node2id` to `{index_id: "<node_type>: <msg>"}`, as `explanations/export_node_mapping.py` does.

## Run order and outputs

`run_pipeline.sh [--seed N] [--full-rebuild] [--skip a,b] [--only a,b]`. It runs from `kairos/cadets_e3/` and logs to `<artifact>/logs/<stage>[_seed<N>].log`. It exports `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=<seed>` and `PYTHONUNBUFFERED=1`.

| # | Stage | Command | Outputs (`<A>` = `$KAIROS_ARTIFACT_DIR`, `<S>` = `<A>/seed_<n>`) |
|---|---|---|---|
| 1 | `db_setup` (full rebuild only) | psql DDL | database + tables |
| 2 | `create_database` (full rebuild only) | `python create_database.py` | rows in the 5 tables |
| 3 | `embedding` (full rebuild only) | `python embedding.py` | `<A>/graphs/graph_4_2..13.TemporalData.simple`, `node2higvec`, `rel2vec`, `embedding.log` |
| 4 | `train` | `python train.py --seed N` | `<S>/models/models.pt` (upstream pickle), `<S>/models/models_state.pt`, `<S>/train_log.json`, `<S>/training.log` |
| 5 | `test` | `python test.py --seed N [--graphs ...]` | `<S>/graph_4_3 .. graph_4_13/<start>~<end>.txt` (one `str(dict)` per edge, sorted by loss), `<S>/reconstruction.log` |
| 6 | `queues` | `python anomalous_queue_construction.py --seed N` | `<S>/node_IDF`, `<S>/graph_4_5 .. graph_4_13_history_list`, `<S>/anomalous_queue.log` |
| 7 | `evaluation` | `python evaluation.py --seed N` | `<S>/detection_metrics.json`, `<S>/alerts.json`, `<S>/windows.json`, `<S>/evaluation.log` |
| 8 | `node_gt` | `python node_gt.py [--mode name\|uuid\|both]` | `<A>/ground_truth/kairos_nodes_Nginx_Backdoor_{06,11,12,13}.json` |
| - | (manual) | `python attack_investigation.py --seed N` | `<S>/graph_visual/` (upstream: 04-06 windows only; needs graphviz + python-louvain) |

You can run the scripts directly from `kairos/cadets_e3/`. Without `--seed` they behave like upstream: unseeded, writing to `<A>/` itself. The upstream `Makefile` is unchanged and runs that unseeded layout.

`models_state.pt` holds `{memory, gnn, link_pred: state_dict, hyperparams: {memory_dim, time_dim, embedding_dim, neighbor_size, raw_msg_dim, max_node_num, include_edge_type, node_embedding_dim, batch_size, lr, eps, weight_decay, epoch_num, time_window_size}, seed, upstream_sha}`. `train_log.json` holds the per-epoch loss of each training day, its mean, and wall time per epoch and in total.

## Evaluation protocol (`evaluation.py`, patch 06)

* Node IDF comes from the windows of `graph_4_3..graph_4_5`, as upstream. `graph_4_5` is the validation day: its largest queue score is reported.
* Queues are built per day for `graph_4_6..graph_4_13`. Queue score = product of (window anomalous-loss average + 1). A queue is anomalous if score > beta, and all its windows are then predicted malicious. Beta is 100 on every day (`beta_day6`, `beta_day7`, `beta_default` in `config.py`).
* Window-level ground truth, in US/Eastern, the convention of `ns_time_to_datetime_US` used in the window names:
  * 04-06 (`graph_4_6`): the four upstream windows `11:18:26.126177915 ~ 14:01:32.489584227`. The windows overlapping PIDSMaker's 11:20:00-12:09:00 are reported as a cross-check.
  * 04-11 (`graph_4_11`): windows overlapping 15:07:00-15:16:00.
  * 04-12 (`graph_4_12`): windows overlapping 13:59:00-14:39:00.
  * 04-13 (`graph_4_13`): windows overlapping 09:03:00-09:16:00.
  * 04-07 to 04-10 have no attack; every window there is benign.
* `detection_metrics.json` contains:
  * `per_day`: tp/fp/fn/tn, precision, recall, fscore, accuracy, fpr, AUC on binary predictions as upstream; ratios are `null` when undefined. Also queues with ids, scores and windows, `window_scores` (largest score of a queue containing the window), GT and flagged windows.
  * `overall`: pooled over all evaluated test days.
  * `per_attack`: detected windows and largest GT window score.
  * `upstream_protocol`: the upstream evaluation exactly, i.e. `graph_4_6` + `graph_4_7`, `beta_day6`/`beta_day7`, the four upstream windows. Comparable with the paper.
  * `ground_truth`, `validation`, `config`, `seed`, `upstream_sha`.
* `alerts.json`: one row per (anomalous queue, window) on the test days, as `{day: "4_6", window_file, start, end, start_ns, end_ns, queue_id, queue_score, beta, is_gt_attack_window}`. `queue_id` is the index in that day's `history_list`. `start_ns`/`end_ns` invert `ns_time_to_datetime_US` (pytz US/Eastern `localize`, plus the 9-digit fraction) with `kairos_utils.ns_time_from_datetime_US`.
* `windows.json`: every window of `graph_4_5..graph_4_13` as `{day, split, window_file, start, end, start_ns, end_ns, n_edges, loss_mean, loss_std, threshold (= loss_mean + 1.5 * loss_std, as cal_anomaly_loss), n_anomalous_edges, anomalous_loss_avg}`. The statistics are recorded by the queue construction, so rerun the `queues` stage if the history lists predate patch 06.
* PIDSMaker excludes 04-11, because the attack largely failed (see `ground_truth/cadets_e3/orthrus/SOURCE.md`). It is kept here and reported separately.

## Node-level ground truth (`node_gt.py`, patch 07)

```bash
python node_gt.py                       # name mapping with $KAIROS_NODE_MAP_JSON (no database)
python node_gt.py --mode both           # + UUID -> hash_id -> node2id through the database
```

The input is `ground_truth/cadets_e3/orthrus/node_Nginx_Backdoor_XX.csv` (`UUID,{'<type>': '<label>'},<PIDSMaker id>`). The last column is PIDSMaker's id, not KAIROS'.

The name mapping works because KAIROS keys nodes by name anyway:

* subject `'None nginx'` maps to `subject: nginx`.
* file paths match exactly; a `'None'` path is unmappable.
* netflow `'la:lp->ra:rp'` maps to every `netflow: ra:rp` id, because KAIROS labels netflows by remote `ip:port` only. Such rows are flagged `ambiguous` when there is more than one id. With the database, the exact sha256 of the 4-tuple is tried first.

Output per attack: `{nodes: {index_id: {name, msg, type, uuids, mapped_by}}, rows, unmapped, stats}`. `msg` is the string used in the window files (`"{'subject': 'nginx'}"`). Note that one KAIROS subject node covers every process with that exec name, for example all `sshd` processes.

## Determinism

`--seed` seeds python, numpy, torch and CUDA, and enables deterministic cudnn and `torch.use_deterministic_algorithms(True, warn_only=True)`; `run_pipeline.sh` exports `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED`. Kernels without a deterministic implementation only warn, and torch_scatter / PyG scatter kernels are not covered, so GPU runs are seeded but not guaranteed bitwise identical. Record the variance across seeds.

The graphs are fixed after `embedding`, so all seeds see the same event order. Upstream's `ORDER BY timestamp_rec` does not break ties, so a database rebuild may reorder events that share a timestamp.

## Upstream quirks (kept unless a patch says otherwise)

1. `anomalous_queue_construction.cal_anomaly_loss` divides by zero when no edge of a window exceeds mean + 1.5*std. Patch 08 fixes this only for those windows.
2. In `test.py`, the `loss` variable is reused: the logged window "Loss" and `time_with_loss[...]['loss']` start from the last edge's loss tensor, and `total_loss` is never used. This affects logging only; the per-edge losses in the window files are correct. Not fixed.
3. `test.py`: `total_edges += BATCH` over-counts the last partial batch (logging only). Edges after the last window flush of a day are never written to a window file, so they are not scored.
4. Windows are cut at batch boundaries (1024 events), when the last event of a batch is more than 15 minutes after the window start. In quiet periods a window can be much longer; the fourth upstream 04-06 window spans 12:03-14:01.
5. Queue construction: the `if added_que_flag: break` sits inside the inner loop. A window can therefore join several queues: the first related queue, then every later queue whose first window is related. `evaluation.py` keeps this behaviour; `alerts.json` has one row per (queue, window).
6. `cal_set_rel` overwrites `node_IDF` for keyword nodes (`netflow`, `usr`, `proc`, `var`, `cadet`, ...). Every such node gets the same value, so this is harmless. The keyword substrings are broad and filter many nodes out of queue linking.
7. Upstream `evaluation.py` only works when both labels occur (`confusion_matrix(...).ravel()` into 4 values) and needs all four hard-coded window names to exist. The new per-day metrics use `labels=[0, 1]`; missing GT windows are counted as false negatives and reported.
8. `model.py` hard-codes `max_node_num = 268243`. Subject identity is the exec name, and netflow labels drop the source address and port.
9. `attack_investigation.py` hard-codes the four 04-06 windows and has dead code (`e['srcnode']` on a tuple inside `try/except`). Its Louvain partition is random unless seeded (`--seed`).
10. `embedding.py` would fail on a day without events (`torch.vstack([])`). This does not happen for CADETS E3 days 4_2..4_13.
