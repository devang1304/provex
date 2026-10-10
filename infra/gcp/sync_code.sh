#!/usr/bin/env bash
# sync_code.sh - push your LOCAL working tree (tracked + untracked-but-not-ignored files,
# uncommitted edits included) into /data/provex on a VM. Locally only git, tar and gcloud run:
#
#   git ls-files -co --exclude-standard -z | tar --null -T - -czf - \
#     | gcloud compute ssh provex@VM --command 'tar -xzf - -C /data/provex'
#
# Git-LFS files (artifact.tar.gz) are skipped - the VM has them from its own clone - and so is
# anything bigger than SYNC_MAX_FILE_MB. Files deleted locally are NOT deleted on the VM.
# run_job.sh does this automatically before every job (unless --no-sync).
#
# Usage: ./sync_code.sh <cpu|gpu|VM> [--scp] [--start] [--force] [--list]
#   --scp    tarball + gcloud compute scp instead of piping through ssh stdin
#   --start  start the VM if it is stopped (hard cap INTERACTIVE_MAX_HOURS; watchdog applies)
#   --force  sync even while a provex-job is running (normally refused)
#   --list   only print what would be sent (no gcloud call)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

VM_ARG="" MODE="$SYNC_METHOD" START=0 FORCE=0 LIST=0
for arg in "$@"; do
  case $arg in
    --scp) MODE=scp ;;
    --start) START=1 ;;
    --force) FORCE=1 ;;
    --list) LIST=1 ;;
    -h|--help) usage_from_header "${INFRA_DIR}/sync_code.sh"; exit 0 ;;
    -*) die "unknown option: $arg" ;;
    *) VM_ARG=$arg ;;
  esac
done

if (( LIST )); then
  ROOT="$(git -C "$INFRA_DIR" rev-parse --show-toplevel)"
  build_sync_list "$ROOT" "${INFRA_DIR}/${BUILD_DIR}/sync-files.lst"
  tr '\0' '\n' < "${INFRA_DIR}/${BUILD_DIR}/sync-files.lst"
  log "${SYNC_FILES} files, $(( SYNC_BYTES / 1024 )) KiB (skipped ${SYNC_SKIPPED_LFS} LFS, ${SYNC_SKIPPED_BIG} large)"
  exit 0
fi
[[ -n $VM_ARG ]] || die "usage: ./sync_code.sh <cpu|gpu|VM> [--scp] [--start] [--force] [--list]"
VM="$(resolve_vm "$VM_ARG")"

require_gcloud
require_project_id
if (( START )); then
  ensure_running "$VM" $(( INTERACTIVE_MAX_HOURS * 60 ))
else
  load_instance "$VM" || die "VM ${VM} not found"
  [[ $I_STATUS == RUNNING ]] || die "${VM} is ${I_STATUS}. Start it with --start, ./ssh_vm.sh ${VM_ARG}, or just use ./run_job.sh (it syncs too)."
fi
if (( ! FORCE )); then
  case "$(job_session_state "$VM" "$I_ZONE")" in
    BUSY) die "a provex-job is running on ${VM}; syncing would change code under it (use --force to do it anyway)" ;;
    UNKNOWN) die "cannot reach ${VM} over ssh (run ./ssh_vm.sh ${VM_ARG} once interactively)" ;;
  esac
fi
step "Syncing working tree -> ${VM}:${REMOTE_REPO_DIR} (${MODE})"
sync_working_tree "$VM" "$I_ZONE" "$MODE"
log "VM-side stamp: $(vm_ssh "$VM" "$I_ZONE" 'cat ~/.provex-sync' 2>/dev/null || echo '?')"
