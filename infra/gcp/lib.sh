# shellcheck shell=bash
# =============================================================================
# infra/gcp/lib.sh - helpers shared by the laptop-side scripts. Sourced, never run.
#
#  * loads config.env (+ the git-ignored config.local.env)
#  * puts the Windows Cloud SDK (GCLOUD_BIN_DIR) on PATH and points its sh launcher
#    at the bundled python
#  * keeps Git Bash (MSYS) from rewriting remote paths such as /data/provex inside
#    --command= / --metadata= arguments
#  * cd's into infra/gcp: local files are handed to gcloud as short RELATIVE paths
#    (no drive letters for gcloud/PuTTY to misparse)
#  * gcv = gcloud with the CR stripped that python.exe appends to every output line
# =============================================================================

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "${INFRA_DIR}/config.env"
cd "$INFRA_DIR" || exit 1

BUILD_DIR=".build"                       # scratch space (git-ignored), relative to INFRA_DIR
mkdir -p "$BUILD_DIR"

# MSYS converts arguments that look like POSIX paths when it starts a native Windows program.
# gcloud's sh launcher relies on that (it hands /c/... paths to python.exe), so conversion
# stays on; only the argument prefixes that carry REMOTE paths or URLs are excluded.
export MSYS2_ARG_CONV_EXCL="--command=;--metadata=;--message=;--ssh-flag=;gs://"

if [[ -n ${GCLOUD_BIN_DIR:-} && -d $GCLOUD_BIN_DIR && ":${PATH}:" != *":${GCLOUD_BIN_DIR}:"* ]]; then
  PATH="${GCLOUD_BIN_DIR}:${PATH}"
  export PATH
fi

# Scope every gcloud call to the PROVEX project without touching the global gcloud config.
export CLOUDSDK_CORE_PROJECT="$PROJECT_ID"
export CLOUDSDK_COMPUTE_REGION="$REGION"

VM_SA_EMAIL="${VM_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KS_SA_EMAIL="${KILLSWITCH_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
JOB_SESSION="provex-job"
STARTED_BY_US=0

# ------------------------------------------------------------------ output / prompts
_ts()  { date +%H:%M:%S; }
log()  { printf '[%s] %s\n' "$(_ts)" "$*"; }
step() { printf '\n[%s] === %s\n' "$(_ts)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(_ts)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(_ts)" "$*" >&2; exit 1; }
loud() {  # loud LINE... -> boxed message on stderr
  local bar='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' l
  printf '\n%s\n' "$bar" >&2
  for l in "$@"; do printf '!!  %s\n' "$l" >&2; done
  printf '%s\n\n' "$bar" >&2
}
confirm() {  # confirm QUESTION -> 0 on y/yes (or ASSUME_YES=1)
  local ans=""
  if [[ ${ASSUME_YES:-0} == 1 ]]; then return 0; fi
  read -r -p "$1 [y/N] " ans || return 1
  [[ $ans =~ ^[Yy]([Ee][Ss])?$ ]]
}
confirm_typed() {  # confirm_typed PROMPT EXACT_TEXT (never skipped by ASSUME_YES / --yes)
  local ans=""
  read -r -p "$1 Type '$2' to confirm (anything else skips): " ans || return 1
  [[ $ans == "$2" ]]
}
retry() {  # retry ATTEMPTS DELAY_S CMD...
  local n=$1 delay=$2 i
  shift 2
  for (( i = 1; ; i++ )); do
    if "$@"; then return 0; fi
    if (( i >= n )); then return 1; fi
    warn "attempt $i/$n failed, retrying in ${delay}s: $*"
    sleep "$delay"
  done
}
usage_from_header() {  # print the comment block at the top of a script
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$1"
}

# ------------------------------------------------------------------ gcloud on Windows
_provex_gcloud_setup() {
  local gc root py
  if ! command -v gcloud >/dev/null 2>&1 && command -v gcloud.cmd >/dev/null 2>&1; then
    # Only gcloud.cmd is available: wrap it (note: cmd.exe caps a command line at ~8 KB).
    gcloud() { gcloud.cmd "$@"; }
  fi
  if [[ -n ${CLOUDSDK_PYTHON:-} ]] || ! command -v cygpath >/dev/null 2>&1; then return 0; fi
  gc="$(type -P gcloud 2>/dev/null || true)"
  if [[ -z $gc ]]; then return 0; fi
  root="$(cd "$(dirname "$gc")/.." 2>/dev/null && pwd)" || return 0
  py="${root}/platform/bundledpython/python.exe"
  if [[ -f $py ]]; then
    # Otherwise the sh launcher picks python3/python from PATH (WindowsApps stub, Python 3.14).
    CLOUDSDK_PYTHON="$(cygpath -m "$py")"
    export CLOUDSDK_PYTHON
  fi
}
_provex_gcloud_setup

gcv() { gcloud "$@" | tr -d '\r'; }   # gcloud whose output is captured / parsed

# ------------------------------------------------------------------ preconditions
require_gcloud() {
  command -v gcloud >/dev/null 2>&1 \
    || die "gcloud not found. Install the Google Cloud CLI or fix GCLOUD_BIN_DIR in config.env (now '${GCLOUD_BIN_DIR:-}')."
  GCLOUD_ACCOUNT="$(gcv auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1)" || GCLOUD_ACCOUNT=""
  [[ -n $GCLOUD_ACCOUNT ]] || die "no active gcloud account - run: gcloud auth login"
}
require_project_id() {
  [[ $PROJECT_ID != *changeme* ]] || die "set PROJECT_ID in infra/gcp/config.env (or config.local.env) first"
  [[ $PROJECT_ID =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "PROJECT_ID '$PROJECT_ID' is not a valid project id"
}
require_project_exists() {
  gcloud projects describe "$PROJECT_ID" --format='value(projectId)' >/dev/null 2>&1 \
    || die "project $PROJECT_ID not found or not accessible by ${GCLOUD_ACCOUNT:-the active account} - run ./bootstrap_project.sh"
}
resolve_billing_account() {  # BILLING_ACCOUNT_ID, else the account the project is linked to
  local name=""
  if [[ -n ${BILLING_ACCOUNT_ID:-} ]]; then printf '%s' "$BILLING_ACCOUNT_ID"; return 0; fi
  name="$(gcv billing projects describe "$PROJECT_ID" --format='value(billingAccountName)' 2>/dev/null)" || name=""
  printf '%s' "${name#billingAccounts/}"
}

# ------------------------------------------------------------------ instances
resolve_vm() {  # cpu|gpu|NAME -> instance name
  case "$1" in
    cpu|CPU) printf '%s' "$CPU_VM" ;;
    gpu|GPU) printf '%s' "$GPU_VM" ;;
    *)       printf '%s' "$1" ;;
  esac
}

# load_instance NAME -> sets I_NAME I_ZONE I_STATUS I_MACHINE I_MODEL I_LASTSTART I_CAP_S
# (I_MODEL: STANDARD|SPOT, I_CAP_S: max-run-duration in seconds or empty). Returns 1 if absent.
load_instance() {
  local want=$1 line=""
  I_NAME="" I_ZONE="" I_STATUS="" I_MACHINE="" I_MODEL="" I_LASTSTART="" I_CAP_S=""
  line="$(gcv compute instances list --filter="name=${want}" \
    --format="csv[no-heading,separator='|'](name,zone.basename(),status,machineType.basename(),scheduling.provisioningModel,lastStartTimestamp,scheduling.maxRunDuration.seconds)" \
    2>/dev/null | awk -F'|' -v n="$want" '$1 == n { print; exit }')" || true
  [[ -n $line ]] || return 1
  IFS='|' read -r I_NAME I_ZONE I_STATUS I_MACHINE I_MODEL I_LASTSTART I_CAP_S <<< "$line"
  I_MODEL=${I_MODEL:-STANDARD}
  return 0
}

iso_to_epoch() { date -d "$1" +%s 2>/dev/null || printf '0'; }
rate_for() {  # rate_for MACHINE_TYPE [STANDARD|SPOT] -> USD/h from the rate table (0 = unknown)
  local var="RATE_${1//-/_}_${2:-STANDARD}"
  printf '%s' "${!var:-0}"
}
disk_rate() {  # disk_rate pd-balanced -> USD per GB-month
  local var="RATE_DISK_${1//-/_}"
  printf '%s' "${!var:-0.10}"
}
cap_minutes_for_hours() {  # ceil(hours*60) + HARD_CAP_GRACE_MIN
  awk -v h="$1" -v g="$HARD_CAP_GRACE_MIN" 'BEGIN { m = h * 60; if (m > int(m)) m = int(m) + 1; printf "%d", m + g }'
}

vm_ssh() {  # vm_ssh VM ZONE COMMAND... (non-interactive; stdin is passed through to the VM)
  local vm=$1 zone=$2
  shift 2
  gcloud compute ssh "${SSH_USER}@${vm}" --zone="$zone" --quiet --command="$*"
}

wait_status() {  # wait_status VM ZONE 'REGEX' TIMEOUT_S -> prints the final status; 1 on timeout
  local vm=$1 zone=$2 want=$3 timeout=$4 s="" end
  end=$(( $(date +%s) + timeout ))
  while :; do
    s="$(gcv compute instances describe "$vm" --zone="$zone" --format='value(status)' 2>/dev/null)" || s="GONE"
    if [[ $s =~ ^(${want})$ ]]; then printf '%s' "$s"; return 0; fi
    if (( $(date +%s) >= end )); then printf '%s' "$s"; return 1; fi
    sleep 10
  done
}

start_vm() {  # start_vm VM ZONE (explains stockouts / quota errors)
  local vm=$1 zone=$2 out=""
  if out="$(gcloud compute instances start "$vm" --zone="$zone" 2>&1)"; then
    log "started $vm"
    return 0
  fi
  printf '%s\n' "$out" | tr -d '\r' >&2
  if grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|resource pool exhausted|stockout' <<< "$out"; then
    warn "no capacity for $vm in $zone right now (common for GPUs, especially Spot)."
    warn "retry later, or move the GPU VM to another zone: ./create_gpu_vm.sh --recreate"
  elif grep -qi 'quota' <<< "$out"; then
    warn "quota problem: ./request_quotas.sh --status. CPUS_ALL_REGIONS=12 fits the CPU VM (4) + L4 VM (8);"
    warn "the A100 VM (12 vCPU) cannot run while the CPU VM runs."
  fi
  return 1
}

stop_vm() {  # stop_vm VM ZONE [async]
  if [[ ${3:-} == async ]]; then
    gcloud compute instances stop "$1" --zone="$2" --quiet --async
  else
    gcloud compute instances stop "$1" --zone="$2" --quiet
  fi
}

apply_hard_cap() {  # apply_hard_cap VM ZONE MINUTES (VM must be stopped; never fatal)
  local vm=$1 zone=$2 minutes=$3
  if [[ ${HARD_CAP:-1} != 1 ]]; then log "HARD_CAP=0: no max-run-duration for $vm"; return 0; fi
  if gcloud compute instances set-scheduling "$vm" --zone="$zone" \
       --max-run-duration="${minutes}m" --instance-termination-action=STOP --quiet \
       >/dev/null 2>"${BUILD_DIR}/set-scheduling.err"; then
    log "hard cap: GCE STOPs $vm ${minutes} min after this start (max-run-duration)"
  else
    warn "could not set max-run-duration on $vm (continuing - VM-side timeout + watchdog still apply):"
    tr -d '\r' < "${BUILD_DIR}/set-scheduling.err" | tail -n 3 >&2 || true
  fi
}

cap_check_running() {  # cap_check_running VM NEED_MIN - uses I_CAP_S/I_LASTSTART from load_instance
  local vm=$1 need_min=$2 started remaining
  [[ ${I_CAP_S:-} =~ ^[0-9]+$ ]] || return 0
  started="$(iso_to_epoch "$I_LASTSTART")"
  remaining=$(( I_CAP_S - ( $(date +%s) - started ) ))
  if (( remaining < need_min * 60 )); then
    die "$vm is RUNNING with a max-run-duration that stops it in $(( remaining / 60 )) min, but this needs up to ${need_min} min. Stop it (./down.sh --stop) so a fresh cap is set on the next start, or ask for fewer hours."
  fi
}

# wait_for_ready VM ZONE [TIMEOUT_MIN]: poll the serial console for the marker the startup
# script prints when it is done (no ssh needed while the VM boots).
wait_for_ready() {
  local vm=$1 zone=$2 tmo=${3:-$READY_TIMEOUT_MIN} start_epoch end line kind epoch
  load_instance "$vm" || return 1
  start_epoch="$(iso_to_epoch "$I_LASTSTART")"
  end=$(( $(date +%s) + tmo * 60 ))
  log "waiting for the startup script on $vm (serial-console marker; first boot takes 3-15 min) ..."
  while :; do
    line="$(gcv compute instances get-serial-port-output "$vm" --zone="$zone" --port=1 2>/dev/null \
      | grep -aoE 'PROVEX-STARTUP-(DONE|FAILED) epoch=[0-9]+' | tail -n 1)" || line=""
    if [[ -n $line ]]; then
      kind=${line#PROVEX-STARTUP-}; kind=${kind%% *}
      epoch=${line##*epoch=}
      if (( epoch + 5 >= start_epoch )); then
        if [[ $kind == DONE ]]; then log "startup script finished on $vm"; return 0; fi
        warn "the startup script FAILED on $vm - last provex lines of its serial console:"
        gcv compute instances get-serial-port-output "$vm" --zone="$zone" --port=1 2>/dev/null \
          | grep -a 'provex' | tail -n 25 >&2 || true
        return 1
      fi
    fi
    if (( $(date +%s) >= end )); then warn "timed out after ${tmo} min waiting for $vm"; return 1; fi
    sleep 15
  done
}

wait_for_ssh() {  # wait_for_ssh VM ZONE [ATTEMPTS] (stderr stays visible: key/host-key prompts)
  local vm=$1 zone=$2 n=${3:-8} i out=""
  for (( i = 1; i <= n; i++ )); do
    out="$(vm_ssh "$vm" "$zone" 'echo PROVEX-SSH-OK')" || true
    if [[ $out == *PROVEX-SSH-OK* ]]; then return 0; fi
    warn "ssh to ${SSH_USER}@${vm} not ready (attempt $i/$n); retrying in 15 s"
    sleep 15
  done
  die "cannot ssh to ${SSH_USER}@${vm}. Run ./ssh_vm.sh ${vm} once interactively (first use creates the SSH key; PuTTY may ask to cache the host key)."
}

startup_state_ssh() {  # DONE | FAILED | RUNNING | UNKNOWN (for a VM that is already up)
  local out=""
  out="$(vm_ssh "$1" "$2" 'if [ -f /run/provex-startup.done ]; then echo PROVEX-STATE-DONE; elif [ -f /run/provex-startup.failed ]; then echo PROVEX-STATE-FAILED; else echo PROVEX-STATE-RUNNING; fi')" || true
  case $out in
    *STATE-DONE*) echo DONE ;;
    *STATE-FAILED*) echo FAILED ;;
    *STATE-RUNNING*) echo RUNNING ;;
    *) echo UNKNOWN ;;
  esac
}

job_session_state() {  # job_session_state VM ZONE -> BUSY | FREE | UNKNOWN
  local out=""
  out="$(vm_ssh "$1" "$2" "tmux has-session -t =${JOB_SESSION} 2>/dev/null && echo PROVEX-BUSY || echo PROVEX-FREE")" || true
  case $out in
    *PROVEX-BUSY*) echo BUSY ;;
    *PROVEX-FREE*) echo FREE ;;
    *) echo UNKNOWN ;;
  esac
}

# ensure_running VM CAP_MIN [nowait]: start the VM if it is stopped (after setting the hard
# cap), then wait for its startup script and for ssh. Sets STARTED_BY_US=1 when it started it.
# Leaves load_instance's I_* variables describing the running VM.
ensure_running() {
  local vm=$1 cap_min=$2 nowait=${3:-}
  load_instance "$vm" || die "VM '$vm' does not exist in $PROJECT_ID (create it: ./create_cpu_vm.sh or ./create_gpu_vm.sh)"
  if [[ $I_STATUS =~ ^(STOPPING|SUSPENDING)$ ]]; then
    log "$vm is $I_STATUS - waiting until it is stopped"
    wait_status "$vm" "$I_ZONE" 'TERMINATED|STOPPED|SUSPENDED' 600 >/dev/null || die "$vm did not stop"
    load_instance "$vm" || die "$vm disappeared"
  fi
  if [[ $I_STATUS =~ ^(PROVISIONING|STAGING)$ ]]; then
    log "$vm is $I_STATUS - waiting for RUNNING"
    wait_status "$vm" "$I_ZONE" 'RUNNING' 600 >/dev/null || die "$vm did not reach RUNNING"
    load_instance "$vm" || die "$vm disappeared"
  fi
  case $I_STATUS in
    RUNNING)
      log "$vm is RUNNING ($I_MACHINE, $I_MODEL, zone $I_ZONE)"
      cap_check_running "$vm" "$cap_min" ;;
    TERMINATED|STOPPED)
      step "Starting $vm ($I_MACHINE, $I_MODEL, zone $I_ZONE, ~$(rate_for "$I_MACHINE" "$I_MODEL") USD/h)"
      apply_hard_cap "$vm" "$I_ZONE" "$cap_min"
      start_vm "$vm" "$I_ZONE" || die "could not start $vm"
      STARTED_BY_US=1
      load_instance "$vm" || die "$vm disappeared" ;;
    SUSPENDED) die "$vm is SUSPENDED - resume it: gcloud compute instances resume $vm --zone=$I_ZONE" ;;
    *) die "$vm is in unexpected state '$I_STATUS'" ;;
  esac
  if [[ $nowait == nowait ]]; then return 0; fi
  if (( STARTED_BY_US )); then
    wait_for_ready "$vm" "$I_ZONE" || die "startup script on $vm did not finish (gcloud compute instances get-serial-port-output $vm --zone=$I_ZONE)"
    wait_for_ssh "$vm" "$I_ZONE"
  else
    wait_for_ssh "$vm" "$I_ZONE"
    case "$(startup_state_ssh "$vm" "$I_ZONE")" in
      DONE) : ;;
      FAILED) die "the startup script on $vm failed - see /var/log/provex-startup.log on the VM" ;;
      *) wait_for_ready "$vm" "$I_ZONE" || die "startup script on $vm did not finish" ;;
    esac
  fi
}

# ------------------------------------------------------------------ VM metadata
prepare_vm_files() {  # LF-normalised copies of the VM-side files for --metadata-from-file
  local f
  for f in startup_common.sh startup_cpu.sh startup_gpu.sh provex-watchdog.sh; do
    tr -d '\r' < "vm/$f" > "${BUILD_DIR}/$f"
  done
}
metadata_files_arg() {  # metadata_files_arg cpu|gpu
  printf 'startup-script=%s/startup_%s.sh,provex-startup-common=%s/startup_common.sh,provex-watchdog=%s/provex-watchdog.sh' \
    "$BUILD_DIR" "$1" "$BUILD_DIR" "$BUILD_DIR"
}
metadata_kv_arg() {  # metadata_kv_arg ROLE [EXTRA_KEY=VALUE,...]
  local s
  s="provex-role=$1,provex-ssh-user=${SSH_USER},provex-idle-minutes=${IDLE_MINUTES},provex-repo-url=${REPO_URL},provex-repo-branch=${REPO_BRANCH},provex-repo-dir=${REMOTE_REPO_DIR},provex-clone-repo=${CLONE_REPO_ON_BOOT},enable-guest-attributes=TRUE,enable-oslogin=FALSE"
  if [[ -n ${2:-} ]]; then s+=",$2"; fi
  printf '%s' "$s"
}

# ------------------------------------------------------------------ code sync
# build_sync_list ROOT OUT: NUL-separated tracked + untracked-not-ignored files, minus Git-LFS
# files (the VM has them from its own clone) and files bigger than SYNC_MAX_FILE_MB.
build_sync_list() {
  local root=$1 out=$2 cand="${INFRA_DIR}/${BUILD_DIR}/sync-candidates.lst"
  local max_bytes path _attr info size line
  max_bytes=$(( SYNC_MAX_FILE_MB * 1024 * 1024 ))
  SYNC_FILES=0 SYNC_BYTES=0 SYNC_SKIPPED_LFS=0 SYNC_SKIPPED_BIG=0
  : > "$cand"
  : > "$out"
  while IFS= read -r -d '' path && IFS= read -r -d '' _attr && IFS= read -r -d '' info; do
    if [[ $info == lfs ]]; then SYNC_SKIPPED_LFS=$(( SYNC_SKIPPED_LFS + 1 )); continue; fi
    [[ -f "${root}/${path}" ]] || continue          # deleted locally / not a regular file
    printf '%s\0' "$path" >> "$cand"
  done < <(git -C "$root" ls-files -co --exclude-standard -z | git -C "$root" check-attr --stdin -z filter)
  [[ -s $cand ]] || return 0
  # one stat call for all files (process creation is slow in Git Bash)
  while IFS= read -r line; do
    size=${line%% *}; path=${line#* }
    if (( size > max_bytes )); then
      warn "sync: skipping ${path} ($(( size / 1048576 )) MB > SYNC_MAX_FILE_MB=${SYNC_MAX_FILE_MB})"
      SYNC_SKIPPED_BIG=$(( SYNC_SKIPPED_BIG + 1 ))
      continue
    fi
    printf '%s\0' "$path" >> "$out"
    SYNC_FILES=$(( SYNC_FILES + 1 )); SYNC_BYTES=$(( SYNC_BYTES + size ))
  done < <(cd "$root" && xargs -0 stat -c '%s %n' < "$cand")
}

# sync_working_tree VM ZONE [pipe|scp]: push the local working tree to REMOTE_REPO_DIR.
#   git ls-files -co --exclude-standard -z | tar --null -T - -czf - | gcloud compute ssh VM --command 'tar -xzf - -C /data/provex'
# Files deleted locally are not deleted on the VM. Sets SYNC_COMMIT.
sync_working_tree() {
  local vm=$1 zone=$2 mode=${3:-$SYNC_METHOD} root list commit dirty="" out="" stamp remote_x
  root="$(git -C "$INFRA_DIR" rev-parse --show-toplevel)" || die "infra/gcp is not inside a git work tree"
  list="${INFRA_DIR}/${BUILD_DIR}/sync-files.lst"
  build_sync_list "$root" "$list"
  (( SYNC_FILES > 0 )) || die "sync: nothing to send"
  commit="$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo none)"
  if [[ -n "$(git -C "$root" status --porcelain 2>/dev/null)" ]]; then dirty="-dirty"; fi
  SYNC_COMMIT="${commit}${dirty}"
  stamp="commit=${SYNC_COMMIT} files=${SYNC_FILES} synced_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log "sync: ${SYNC_FILES} files, $(( SYNC_BYTES / 1024 )) KiB, commit ${SYNC_COMMIT} (skipped ${SYNC_SKIPPED_LFS} LFS, ${SYNC_SKIPPED_BIG} large) -> ${vm}:${REMOTE_REPO_DIR}"
  remote_x="mkdir -p ${REMOTE_REPO_DIR} && tar -xzf - -C ${REMOTE_REPO_DIR} && echo '${stamp}' > ~/.provex-sync && echo PROVEX-SYNC-OK"
  if [[ $mode == pipe ]]; then
    out="$( (cd "$root" && tar --null -T "$list" -czf -) | vm_ssh "$vm" "$zone" "$remote_x" )" || out=""
    if [[ $out == *PROVEX-SYNC-OK* ]]; then log "sync done (ssh pipe)"; return 0; fi
    warn "sync through the ssh stdin pipe failed - retrying with a tarball + gcloud compute scp"
    mode=scp
  fi
  if [[ $mode == scp ]]; then
    (cd "$root" && tar --null -T "$list" -czf "${INFRA_DIR}/${BUILD_DIR}/provex-sync.tgz") || die "tar failed"
    # relative local path, remote path relative to the home dir: nothing for MSYS/PuTTY to misparse
    gcloud compute scp --zone="$zone" --quiet "${BUILD_DIR}/provex-sync.tgz" "${SSH_USER}@${vm}:provex-sync.tgz" \
      || die "gcloud compute scp failed"
    out="$(vm_ssh "$vm" "$zone" "mkdir -p ${REMOTE_REPO_DIR} && tar -xzf ~/provex-sync.tgz -C ${REMOTE_REPO_DIR} && rm -f ~/provex-sync.tgz && echo '${stamp}' > ~/.provex-sync && echo PROVEX-SYNC-OK")" || out=""
    [[ $out == *PROVEX-SYNC-OK* ]] || die "remote extract failed"
    log "sync done (scp)"
    return 0
  fi
  die "unknown SYNC_METHOD '$mode' (pipe|scp)"
}

ledger_cat() { gcv storage cat "${BUCKET}/ledger/runs.csv" 2>/dev/null; }
