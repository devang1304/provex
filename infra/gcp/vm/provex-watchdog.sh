#!/bin/bash
# =============================================================================
# provex-watchdog - powers this VM off once it has been idle for IDLE_MINUTES.
# Installed by startup_common.sh as /usr/local/sbin/provex-watchdog and run as root by
# provex-watchdog.timer (first check 10 min after boot, then every 5 min).
#
# The VM is BUSY (idle clock reset) when ANY of these holds:
#   * a tmux session named exactly "provex-job" exists (any user) -> never shut down mid-job
#   * the startup script is still running (only within STARTUP_GRACE_MINUTES of boot)
#   * nvidia-smi exists and the max GPU utilisation >= GPU_UTIL_THRESHOLD %
#   * the 1-minute load average >= LOAD_THRESHOLD
# Interactive ssh sessions are deliberately ignored. To keep a VM up while you work in it,
# work inside `tmux new -s provex-job` and exit that session when you are done.
#
# Settings: /etc/provex-watchdog.defaults (rewritten at boot from instance metadata), then
#           /etc/provex-watchdog.conf (your overrides, never touched), e.g. IDLE_MINUTES=30
# State:    /var/lib/provex-watchdog/count (consecutive idle checks), .../since (first idle epoch)
# Log:      journalctl -t provex-watchdog ; the power-off line also goes to the serial console
# =============================================================================
set -euo pipefail

IDLE_MINUTES=15
GPU_UTIL_THRESHOLD=5
LOAD_THRESHOLD=0.5
STARTUP_GRACE_MINUTES=60
SESSION=provex-job
DRY_RUN=0
STATE_DIR=/var/lib/provex-watchdog

for f in /etc/provex-watchdog.defaults /etc/provex-watchdog.conf; do
  if [[ -f $f ]]; then
    # shellcheck source=/dev/null
    source "$f"
  fi
done

log() { logger -t provex-watchdog -- "$*" 2>/dev/null || true; echo "provex-watchdog: $*"; }

mkdir -p "$STATE_DIR"
NOW=$(date +%s)
UPTIME_S=$(cut -d. -f1 /proc/uptime)
BOOT_EPOCH=$(( NOW - UPTIME_S ))
read -r LOAD1 _ < /proc/loadavg
GPU_MAX="n/a"

job_session_active() {  # checks every user's tmux server, as that user
  local sock owner
  command -v tmux >/dev/null 2>&1 || return 1
  shopt -s nullglob
  for sock in /tmp/tmux-*/*; do
    [[ -S $sock ]] || continue
    owner=$(stat -c %U "$sock" 2>/dev/null) || continue
    if runuser -u "$owner" -- tmux -S "$sock" has-session -t "=${SESSION}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

startup_in_progress() {
  [[ -e /run/provex-startup.done || -e /run/provex-startup.failed ]] && return 1
  (( UPTIME_S < STARTUP_GRACE_MINUTES * 60 ))
}

gpu_busy() {
  local util
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -n 1 | tr -d ' ') || return 1
  [[ $util =~ ^[0-9]+$ ]] || return 1
  GPU_MAX=$util
  (( util >= GPU_UTIL_THRESHOLD ))
}

load_busy() { awk -v l="$LOAD1" -v t="$LOAD_THRESHOLD" 'BEGIN { exit !(l + 0 >= t + 0) }'; }

REASON=""
if job_session_active; then
  REASON="tmux session '${SESSION}' exists"
elif startup_in_progress; then
  REASON="startup script still running"
elif gpu_busy; then
  REASON="GPU utilisation ${GPU_MAX}% >= ${GPU_UTIL_THRESHOLD}%"
elif load_busy; then
  REASON="load average ${LOAD1} >= ${LOAD_THRESHOLD}"
fi

COUNT_FILE="${STATE_DIR}/count"
SINCE_FILE="${STATE_DIR}/since"
SINCE=$(cat "$SINCE_FILE" 2>/dev/null || true)
[[ $SINCE =~ ^[0-9]+$ ]] || SINCE=""
if [[ -n $SINCE ]] && (( SINCE < BOOT_EPOCH )); then SINCE=""; fi   # left over from an earlier boot

if [[ -n $REASON ]]; then
  echo 0 > "$COUNT_FILE"
  rm -f "$SINCE_FILE"
  log "busy: ${REASON} (idle clock reset)"
  exit 0
fi

COUNT=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
[[ $COUNT =~ ^[0-9]+$ ]] || COUNT=0
COUNT=$(( COUNT + 1 ))
echo "$COUNT" > "$COUNT_FILE"
if [[ -z $SINCE ]]; then SINCE=$NOW; echo "$SINCE" > "$SINCE_FILE"; fi
IDLE_MIN=$(( (NOW - SINCE) / 60 ))
log "idle check #${COUNT}: idle for ${IDLE_MIN} min (power-off at ${IDLE_MINUTES}; load ${LOAD1}, gpu ${GPU_MAX})"

if (( IDLE_MIN >= IDLE_MINUTES )); then
  MSG="VM idle for ${IDLE_MIN} min >= IDLE_MINUTES=${IDLE_MINUTES} -> powering off"
  log "$MSG"
  echo "provex-watchdog: ${MSG}" > /dev/ttyS0 2>/dev/null || true
  if [[ $DRY_RUN == 1 ]]; then log "DRY_RUN=1 - not powering off"; exit 0; fi
  echo 0 > "$COUNT_FILE"
  rm -f "$SINCE_FILE"
  wall "provex-watchdog: ${MSG}" 2>/dev/null || true
  shutdown -h now
fi
