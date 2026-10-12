#!/usr/bin/env bash
# PROVEX v2 experiments (runs on a GCP VM from /data/provex with .venv active).
#   bash scripts/run_experiments.sh <stage> [args]
# stages: pull | detect | parity | gmfit | explain DAY [WORKERS] | e6 DAY [WORKERS] | e7 DAY | tables | push
# explain env: METHODS, TOP_N, SAMPLE_N, MAX_EVENTS (0 = all), EXPL_OUT (default results/expl/seed_$SEED),
#   BUDGET (full|bounded), HIST_BATCHES (64; 0 = neighbour edges only), SUBSAMPLE (0 = all), ONLY_CLASS,
#   EVENTS ('1,2,3' or '@file'). e6 also reads BUDGET, HIST_BATCHES, EVENTS (set E6_EVENTS=0 with EVENTS).
# e7 env: E7_EXPL_DIR (explanations with history), E7_OUT, E7_ALERTS, E7_WORKERS, HIST_BATCHES (match the explain run)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi   # VM venv; Docker uses system python
BUCKET=${BUCKET:-gs://provex-v2-dd1304-artifacts}
SEED=${SEED:-0}
export KAIROS_GRAPHS_DIR=$PWD/artifact/graph_embeddings/ KAIROS_NODE_MAP_JSON=$PWD/artifact/explanations/node_mapping.json
export KAIROS_ARTIFACT_DIR=$PWD/artifact/kairos_upstream/ OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
K=$KAIROS_ARTIFACT_DIR/seed_$SEED
CKPT=$K/models/models_state.pt
G=$KAIROS_GRAPHS_DIR
NAMES=$KAIROS_NODE_MAP_JSON
GT=$KAIROS_ARTIFACT_DIR/ground_truth
R=results
mkdir -p $R
stage=${1:?stage}; shift || true
case "$stage" in
  pull)   gcloud storage rsync -r $BUCKET/kairos_upstream $KAIROS_ARTIFACT_DIR
          [ -d kairos/cadets_e3 ] || bash kairos/setup_upstream.sh ;;
  detect) [ -d kairos/cadets_e3 ] || bash kairos/setup_upstream.sh
          for s in ${SEEDS:-0 1 2}; do
            (bash kairos/cadets_e3/run_pipeline.sh --seed $s --skip train > $R/detect_seed$s.log 2>&1; echo "seed $s exit $?") &
          done; wait
          mkdir -p $R/kairos
          for s in ${SEEDS:-0 1 2}; do mkdir -p $R/kairos/seed_$s
            cp $KAIROS_ARTIFACT_DIR/seed_$s/{detection_metrics.json,alerts.json,windows.json} $R/kairos/seed_$s/ 2>/dev/null || true; done ;;
  parity) for d in 4_5 4_6 4_7 4_8 4_9 4_10 4_11 4_12 4_13; do
            python -m provex.pipeline pass1 --ckpt $CKPT --day-file $G/graph_$d.TemporalData.simple --day $d --out $R/pass1
            python -m provex.pipeline parity --pass1 $R/pass1/pass1_$d.pt --kairos-windows $K/windows.json --day $d --out $R/parity || true
          done ;;
  gmfit)  python -m provex.pipeline graphmask-fit --ckpt $CKPT --day-file $G/graph_4_5.TemporalData.simple --day 4_5 \
            --out $R/graphmask/graphmask_seed$SEED.pt --epochs ${GM_EPOCHS:-40} ;;
  explain) d=${1:?day}; W=${2:-$(nproc)}; O=${EXPL_OUT:-$R/expl/seed_$SEED}; mkdir -p $O
          [ -f $R/pass1/pass1_$d.pt ] || python -m provex.pipeline pass1 --ckpt $CKPT --day-file $G/graph_$d.TemporalData.simple --day $d --out $R/pass1
          pids=()
          for i in $(seq 0 $((W-1))); do
            python -m provex.pipeline explain --ckpt $CKPT --day-file $G/graph_$d.TemporalData.simple --day $d \
              --alerts $K/alerts.json --names $NAMES --gt-dir $GT --pass1 $R/pass1/pass1_$d.pt \
              --graphmask $R/graphmask/graphmask_seed$SEED.pt \
              --methods ${METHODS:-joint,joint-cf,joint-edges,variational,graphmask,gnnx-anomaly,gnnx-v1,mcts,baselines} \
              --top-n ${TOP_N:-25} --sample-n ${SAMPLE_N:-25} --max-events ${MAX_EVENTS:-0} --shard $i/$W \
              --budget ${BUDGET:-full} --history-batches ${HIST_BATCHES:-64} --subsample ${SUBSAMPLE:-0} \
              ${ONLY_CLASS:+--only-class $ONLY_CLASS} ${EVENTS:+--events $EVENTS} \
              --out $O > $O/explain_${d}_shard$i.log 2>&1 &
            pids+=($!)
          done
          fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done   # a crashed shard fails the stage
          [ $fail -eq 0 ] || { echo "explain $d: a shard failed, see $O/explain_${d}_shard*.log"; exit 1; } ;;
  e6)     d=${1:?day}; W=${2:-$(nproc)}; mkdir -p $R/e6
          pids=()
          for i in $(seq 0 $((W-1))); do
            python -m provex.pipeline explain --ckpt $CKPT --day-file $G/graph_$d.TemporalData.simple --day $d \
              --alerts $K/alerts.json --names $NAMES --gt-dir $GT --pass1 $R/pass1/pass1_$d.pt \
              --methods joint,joint-cf,variational,gnnx-anomaly \
              --seeds 0,1,2,3,4 --full-scores --calibration --only-class window --subsample ${E6_EVENTS:-60} \
              --budget ${BUDGET:-full} --history-batches ${HIST_BATCHES:-64} ${EVENTS:+--events $EVENTS} \
              --top-n 25 --sample-n 25 --shard $i/$W --out $R/e6 > $R/e6/e6_${d}_shard$i.log 2>&1 &
            pids+=($!)
          done
          fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
          [ $fail -eq 0 ] || { echo "e6 $d: a shard failed, see $R/e6/e6_${d}_shard*.log"; exit 1; } ;;
  e7)     d=${1:?day}
          cat ${E7_EXPL_DIR:-$R/expl/seed_$SEED}/explanations_${d}*.jsonl > $R/expl_${d}_all.jsonl
          python -m provex.pipeline local-vs-full --ckpt $CKPT --day-file $G/graph_$d.TemporalData.simple \
            --explanations $R/expl_${d}_all.jsonl --out ${E7_OUT:-$R/local_vs_full.json} --max-alerts ${E7_ALERTS:-30} \
            --workers ${E7_WORKERS:-$(nproc)} --history-batches ${HIST_BATCHES:-64} ;;
  tables) python scripts/aggregate_e6.py --records "$R/e6/*.jsonl" --out $R || true
          mkdir -p $R/kairos; python scripts/make_tables.py --results $R --out paper/generated
          python scripts/make_case_studies.py --results $R --gt-dir $GT --out paper/generated ;;
  push)   gcloud storage rsync -r $R $BUCKET/results
          gcloud storage rsync -r paper/generated $BUCKET/paper_generated ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
