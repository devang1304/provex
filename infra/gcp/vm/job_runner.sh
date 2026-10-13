#!/usr/bin/env bash
# =============================================================================
# vm/job_runner.sh - runs ON THE VM as the ssh user (provex); driven by run_job.sh.
#
#   job_runner.sh launch <job> <max_hours> <cmd_b64> <meta_b64>
#        create ~/jobs/<job>/ (command.sh, meta.env) and start the detached tmux session
#        "provex-job", which runs:
#   job_runner.sh run <job>
#        * sampler every 60 s -> monitor.log (UTC time, GPU util/mem, free -m, loadavg)
#        * heartbeat + rsync of ~/jobs/<job> to <bucket>/jobs/<job> every 5 min
#        * cd /data/provex, activate .venv if present, `timeout <max_hours>h bash command.sh`
#          -> job.log (mirrored to the tmux pane); exit code -> exit_code
#        * final rsync; one row in <bucket>/ledger/runs.csv (download, append, upload with
#          if-generation-match, retried); then `sudo shutdown -h now` unless NO_SHUTDOWN=1.
#          The VM stops itself even if the laptop that launched the job is gone.
#   job_runner.sh abort    TERM the active job (the run then uploads, records and powers off)
#   job_runner.sh status   active job + tails of job.log and monitor.log
# =============================================================================
set -uo pipefail   # no -e on purpose: the upload / ledger / shutdown path must always run

SESSION="provex-job"
JOBS_ROOT="${HOME}/jobs"
ACTIVE_FILE="${JOBS_ROOT}/.active"
LAST_END_FILE="/dev/shm/provex-last-job-end"
LEDGER_HEADER="job,vm,zone,machine_type,provisioning_model,start_utc,end_utc,job_hours,billable_hours,exit_code,status,rate_usd_h,est_cost_usd,git_commit"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-60}"
export PATH="${PATH}:/usr/local/bin:/snap/bin"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "job_runner: ERROR: $*" >&2; exit 2; }
valid_job() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$ ]]; }
session_exists() { tmux has-session -t "=${SESSION}" 2>/dev/null; }

# ------------------------------------------------------------------ GCS helpers
gcs_tool() {
  if command -v gcloud >/dev/null 2>&1; then echo gcloud
  elif command -v gsutil >/dev/null 2>&1; then echo gsutil
  else echo none; fi
}
gcs_rsync() {  # gcs_rsync SRC_DIR DST_URL
  case "$(gcs_tool)" in
    gcloud) timeout 600 gcloud storage rsync --recursive --quiet "$1" "$2" ;;
    gsutil) timeout 600 gsutil -m -q rsync -r "$1" "$2" ;;
    *) echo "job_runner: no gcloud/gsutil on this VM" >&2; return 1 ;;
  esac
}
gcs_cp() {  # gcs_cp SRC DST
  case "$(gcs_tool)" in
    gcloud) timeout 120 gcloud storage cp --quiet "$1" "$2" ;;
    gsutil) timeout 120 gsutil -q cp "$1" "$2" ;;
    *) return 1 ;;
  esac
}

# ledger_append CSV_LINE: read-modify-write of <bucket>/ledger/runs.csv guarded by
# if-generation-match (generation 0 = "must not exist yet"); concurrent writers retry.
ledger_append() {
  local line=$1 obj="${BUCKET}/ledger/runs.csv" tool tmp err gen attempt
  tool="$(gcs_tool)"
  tmp="$(mktemp)"
  err="$(mktemp)"
  for attempt in 1 2 3 4 5 6 7 8; do
    gen=""
    if [[ $tool == gcloud ]]; then
      gen="$(timeout 60 gcloud storage objects describe "$obj" --format='value(generation)' 2>/dev/null)" || gen=""
    elif [[ $tool == gsutil ]]; then
      gen="$(timeout 60 gsutil stat "$obj" 2>/dev/null | awk '/Generation:/ { print $2 }')" || gen=""
    else
      break
    fi
    if [[ -n $gen ]]; then
      if ! gcs_cp "${obj}#${gen}" "$tmp" 2>"$err"; then sleep "$attempt"; continue; fi
    else
      gen=0
      printf '%s\n' "$LEDGER_HEADER" > "$tmp"
    fi
    printf '%s\n' "$line" >> "$tmp"
    if [[ $tool == gcloud ]]; then
      if timeout 120 gcloud storage cp --quiet --if-generation-match="$gen" "$tmp" "$obj" 2>"$err"; then
        rm -f "$tmp" "$err"; echo "job_runner: ledger row appended"; return 0
      fi
    else
      if timeout 120 gsutil -q -h "x-goog-if-generation-match:${gen}" cp "$tmp" "$obj" 2>"$err"; then
        rm -f "$tmp" "$err"; echo "job_runner: ledger row appended"; return 0
      fi
    fi
    echo "job_runner: ledger attempt ${attempt} failed ($(tail -n 1 "$err" 2>/dev/null)) - retrying"
    sleep $(( attempt + RANDOM % 5 ))
  done
  # last resort - never lose the row: write it as its own object (status.sh reads these too)
  printf '%s\n%s\n' "$LEDGER_HEADER" "$line" > "$tmp"
  if gcs_cp "$tmp" "${BUCKET}/ledger/pending/${JOB}-$(date +%s).csv"; then
    echo "job_runner: ledger row written to ledger/pending/"
  else
    echo "job_runner: FAILED to record the ledger row: ${line}"
  fi
  rm -f "$tmp" "$err"
  return 1
}

# ------------------------------------------------------------------ background helpers
sampler() {
  local gpu mem load
  while :; do
    gpu="n/a"
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | paste -sd ';' -)" || gpu="error"
    fi
    mem="$(free -m | awk '/^Mem:/ { printf "used=%sMiB total=%sMiB avail=%sMiB", $3, $2, $7 }')"
    load="$(cut -d' ' -f1-3 /proc/loadavg)"
    printf '%s | gpu(util,mem): %s | mem: %s | load: %s\n' "$(ts)" "${gpu:-n/a}" "$mem" "$load" >> "${JOB_DIR}/monitor.log"
    sleep "$SAMPLE_SECONDS"
  done
}
heartbeat() {
  while :; do
    ts > "${JOB_DIR}/heartbeat"
    gcs_rsync "$JOB_DIR" "${BUCKET}/jobs/${JOB}" >/dev/null 2>>"${JOB_DIR}/rsync.err" || true
    sleep "$HEARTBEAT_SECONDS"
  done
}

on_signal() {  # TERM/HUP/INT (abort, tmux kill-session, VM shutdown or Spot preemption)
  INTERRUPTED=$1
  echo "[$(ts)] job_runner: received SIG$1 - stopping the job" >> "${JOB_DIR}/job.log"
  if [[ -n ${CHILD:-} ]]; then kill -TERM "$CHILD" 2>/dev/null; fi
}

finish() {  # runs exactly once: exit code, upload, ledger, power-off
  [[ -n ${FINISHED:-} ]] && return 0
  FINISHED=1
  trap - EXIT TERM HUP INT
  local end_utc end_epoch rc status job_h bill_h cost line p
  end_utc="$(ts)"; end_epoch=$(date +%s)
  rc=${RC:-255}
  case $rc in
    0) status=ok ;;
    124|137) status=timeout ;;
    *) status=failed ;;
  esac
  if [[ -n ${INTERRUPTED:-} && $status != timeout ]]; then status="interrupted-${INTERRUPTED}"; fi
  echo "$rc" > "${JOB_DIR}/exit_code"
  for p in ${SAMPLER_PID:-} ${HEARTBEAT_PID:-} ${TAIL_PID:-}; do kill "$p" 2>/dev/null; done
  job_h=$(awk -v s="$(( end_epoch - ${START_EPOCH:-$end_epoch} ))" 'BEGIN { printf "%.4f", s / 3600 }')
  bill_h=$(awk -v s="$(( end_epoch - ${BILL_START:-$end_epoch} ))" 'BEGIN { printf "%.4f", s / 3600 }')
  cost=$(awk -v h="$bill_h" -v r="${RATE_USD_H:-0}" 'BEGIN { printf "%.4f", h * r }')
  echo "=== finished ${end_utc}: exit_code=${rc} status=${status} job=${job_h}h billable~${bill_h}h est_cost~${cost} USD" >> "${JOB_DIR}/job.log"
  ts > "${JOB_DIR}/heartbeat"
  echo "job_runner: final upload to ${BUCKET:-?}/jobs/${JOB}"
  gcs_rsync "$JOB_DIR" "${BUCKET:-}/jobs/${JOB}" >/dev/null 2>>"${JOB_DIR}/rsync.err" || echo "job_runner: final rsync FAILED"
  line="${JOB},${VM_NAME:-?},${VM_ZONE:-?},${MACHINE_TYPE:-?},${PROVISIONING_MODEL:-?},${START_UTC:-?},${end_utc},${job_h},${bill_h},${rc},${status},${RATE_USD_H:-0},${cost},${GIT_COMMIT:-?}"
  if [[ -n ${BUCKET:-} ]]; then ledger_append "$line"; fi
  date +%s > "$LAST_END_FILE" 2>/dev/null
  rm -f "$ACTIVE_FILE"
  if [[ ${NO_SHUTDOWN:-0} == 1 ]]; then
    echo "job_runner: NO_SHUTDOWN=1 - leaving the VM up (the idle watchdog powers it off later)"
    return 0
  fi
  echo "job_runner: powering the VM off"
  sync
  sudo -n shutdown -h now 2>/dev/null || sudo -n systemctl poweroff 2>/dev/null \
    || echo "job_runner: could not power off - the idle watchdog will"
}

# ------------------------------------------------------------------ commands
cmd_launch() {
  local job=${1:-} hours=${2:-} cmd_b64=${3:-} meta_b64=${4:-} dir self
  valid_job "$job" || die "invalid job name '${job}'"
  [[ $hours =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid max_hours '${hours}'"
  command -v tmux >/dev/null 2>&1 || die "tmux is not installed (has the startup script finished?)"
  if session_exists; then die "a '${SESSION}' tmux session is already running (job: $(cat "$ACTIVE_FILE" 2>/dev/null || echo '?'))"; fi
  dir="${JOBS_ROOT}/${job}"
  [[ ! -e $dir ]] || die "${dir} already exists on this VM - pick a new job name"
  mkdir -p "$dir" || die "cannot create ${dir}"
  printf '%s' "$cmd_b64" | base64 -d > "${dir}/command.sh" || die "bad command payload"
  printf '%s' "$meta_b64" | base64 -d > "${dir}/meta.env" || die "bad meta payload"
  grep -q '^MAX_HOURS=' "${dir}/meta.env" || echo "MAX_HOURS=${hours}" >> "${dir}/meta.env"
  self="$(readlink -f "${BASH_SOURCE[0]}")"
  cp "$self" "${dir}/runner.sh"
  cp "$self" "${JOBS_ROOT}/.runner.sh" 2>/dev/null || true     # for --status / --abort later
  tmux new-session -d -s "$SESSION" -x 220 -y 50 "bash '${dir}/runner.sh' run '${job}'" || die "tmux new-session failed"
  sleep 2
  if session_exists || [[ -f ${dir}/exit_code ]]; then
    echo "PROVEX-LAUNCHED job=${job} dir=${dir}"
  else
    die "the tmux session did not start - see ${dir}"
  fi
}

cmd_run() {
  local up last_end rc
  JOB=${1:-}
  valid_job "$JOB" || die "invalid job name"
  JOB_DIR="${JOBS_ROOT}/${JOB}"
  cd "$JOB_DIR" || die "no job dir ${JOB_DIR}"
  START_UTC="$(ts)"; START_EPOCH=$(date +%s)
  trap finish EXIT
  trap 'on_signal TERM' TERM
  trap 'on_signal HUP' HUP
  trap 'on_signal INT' INT
  # shellcheck source=/dev/null
  source "${JOB_DIR}/meta.env"
  REPO_DIR=${REPO_DIR:-/data/provex}
  # billable time starts at boot, or at the end of the previous job of this boot (--no-shutdown)
  up=$(cut -d. -f1 /proc/uptime)
  BILL_START=$(( START_EPOCH - up ))
  last_end=$(cat "$LAST_END_FILE" 2>/dev/null || echo 0)
  if [[ $last_end =~ ^[0-9]+$ ]] && (( last_end > BILL_START )); then BILL_START=$last_end; fi
  echo "$JOB" > "$ACTIVE_FILE"
  {
    echo "=== PROVEX job ${JOB} on ${VM_NAME:-?} (${MACHINE_TYPE:-?}, ${PROVISIONING_MODEL:-?}, ${VM_ZONE:-?})"
    echo "=== started ${START_UTC}, max ${MAX_HOURS} h, commit ${GIT_COMMIT:-?}, cwd ${REPO_DIR}"
    echo "=== command:"
    sed 's/^/    /' "${JOB_DIR}/command.sh"
    echo "==="
  } >> "${JOB_DIR}/job.log"
  sampler & SAMPLER_PID=$!
  heartbeat & HEARTBEAT_PID=$!
  cd "$REPO_DIR" 2>/dev/null || cd "$HOME" || true
  if [[ -f .venv/bin/activate ]]; then
    set +u
    # shellcheck source=/dev/null
    source .venv/bin/activate
    set -u
    echo "=== venv: $(command -v python)" >> "${JOB_DIR}/job.log"
  fi
  export PROVEX_JOB="$JOB" PROVEX_JOB_DIR="$JOB_DIR" PROVEX_BUCKET="${BUCKET:-}" PYTHONUNBUFFERED=1
  # timeout puts the command in its own process group and kills the whole group on expiry
  timeout --kill-after=120 "${MAX_HOURS}h" bash "${JOB_DIR}/command.sh" >> "${JOB_DIR}/job.log" 2>&1 < /dev/null &
  CHILD=$!
  echo "$CHILD" > "${JOB_DIR}/pid"
  tail -n +1 -F --pid="$CHILD" "${JOB_DIR}/job.log" 2>/dev/null &
  TAIL_PID=$!
  while :; do
    wait "$CHILD"; rc=$?
    # >128 while the child is still alive = wait was interrupted by a trapped signal
    if (( rc > 128 )) && kill -0 "$CHILD" 2>/dev/null; then continue; fi
    break
  done
  RC=$rc
  finish
}

cmd_abort() {
  local job pid
  job="$(cat "$ACTIVE_FILE" 2>/dev/null)" || job=""
  if [[ -z $job ]]; then echo "PROVEX-NO-JOB: no active job on this VM"; return 0; fi
  pid="$(cat "${JOBS_ROOT}/${job}/pid" 2>/dev/null)" || pid=""
  if [[ -n $pid ]] && kill -TERM "$pid" 2>/dev/null; then
    echo "PROVEX-ABORTED job=${job}: TERM sent; logs are uploaded, the ledger row written, then the VM powers off"
  else
    echo "job ${job}: process ${pid:-?} is not running"
  fi
}

cmd_status() {
  local job
  job="$(cat "$ACTIVE_FILE" 2>/dev/null)" || job=""
  if session_exists; then echo "tmux session '${SESSION}': RUNNING (job ${job:-?})"; else echo "tmux session '${SESSION}': none"; fi
  if [[ -z $job ]]; then job="$(ls -1t "$JOBS_ROOT" 2>/dev/null | head -n 1)"; fi
  if [[ -z $job || ! -d ${JOBS_ROOT}/${job} ]]; then echo "no jobs on this VM"; return 0; fi
  echo "--- ${JOBS_ROOT}/${job}/job.log (last 20 lines)"
  tail -n 20 "${JOBS_ROOT}/${job}/job.log" 2>/dev/null
  echo "--- monitor.log (last 3 samples)"
  tail -n 3 "${JOBS_ROOT}/${job}/monitor.log" 2>/dev/null
  if [[ -f ${JOBS_ROOT}/${job}/exit_code ]]; then echo "exit_code: $(cat "${JOBS_ROOT}/${job}/exit_code")"; fi
}

case "${1:-}" in
  launch) shift; cmd_launch "$@" ;;
  run)    shift; cmd_run "$@" ;;
  abort)  cmd_abort ;;
  status) cmd_status ;;
  *) echo "usage: job_runner.sh launch <job> <max_hours> <cmd_b64> <meta_b64> | run <job> | abort | status" >&2; exit 2 ;;
esac
