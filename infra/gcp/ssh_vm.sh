#!/usr/bin/env bash
# ssh_vm.sh - open a shell on a PROVEX VM as provex@<vm> (starting it if it is stopped), or run
# one command there.
#
# Usage: ./ssh_vm.sh <cpu|gpu|VM> [command...]
#
# First use from Windows creates the SSH key (%USERPROFILE%\.ssh\google_compute_engine*) and
# gcloud uploads it to the project metadata; PuTTY may ask once to cache the host key (answer y).
# The idle watchdog IGNORES ssh sessions: keep a VM up while you work by working inside
# `tmux new -s provex-job` (exit that session when done), otherwise it powers off after
# IDLE_MINUTES of low load. A VM started here also gets a hard cap of INTERACTIVE_MAX_HOURS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[[ $# -ge 1 && ! ${1:-} =~ ^(-h|--help)$ ]] || { usage_from_header "${INFRA_DIR}/ssh_vm.sh"; exit 0; }
VM="$(resolve_vm "$1")"
shift

require_gcloud
require_project_id
ensure_running "$VM" $(( INTERACTIVE_MAX_HOURS * 60 ))
if (( STARTED_BY_US )); then
  log "${VM} was started for you: watchdog stops it after ${IDLE_MINUTES} idle min; hard cap ${INTERACTIVE_MAX_HOURS} h. Stop it now with ./down.sh"
fi

if (( $# )); then
  vm_ssh "$VM" "$I_ZONE" "$*"
else
  log "interactive shell on ${SSH_USER}@${VM} (repo: ${REMOTE_REPO_DIR}; venv: source ${REMOTE_REPO_DIR}/.venv/bin/activate)"
  gcloud compute ssh "${SSH_USER}@${VM}" --zone="$I_ZONE"
fi
