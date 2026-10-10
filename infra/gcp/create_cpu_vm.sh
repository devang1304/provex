#!/usr/bin/env bash
# create_cpu_vm.sh - the CPU dev/test VM (CPU_MACHINE_TYPE, default e2-standard-4, Ubuntu 22.04)
# plus its persistent data disk (DATA_DISK_SIZE_GB pd-balanced, mounted at /data; the repo lives
# in /data/provex and survives VM re-creation).
#
# Usage: ./create_cpu_vm.sh [--keep-running] [--recreate] [--no-wait]
#   (default)       create the data disk + VM if missing - otherwise refresh the VM metadata
#                   (startup scripts, watchdog) and start it - wait for the startup script
#                   (mount /data, packages, uv, user provex, clone + git lfs pull + extract
#                   artifact.tar.gz), then STOP the VM again. Jobs start it on demand.
#   --keep-running  leave it running (the idle watchdog still powers it off when idle)
#   --recreate      delete the VM first (boot disk goes, the data disk is KEPT)
#   --no-wait       return right after create/start (the watchdog stops it when idle)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

KEEP_RUNNING=0 RECREATE=0 NO_WAIT=0
for arg in "$@"; do
  case $arg in
    --keep-running) KEEP_RUNNING=1 ;;
    --recreate) RECREATE=1 ;;
    --no-wait) NO_WAIT=1 ;;
    -h|--help) usage_from_header "${INFRA_DIR}/create_cpu_vm.sh"; exit 0 ;;
    *) die "unknown option: $arg" ;;
  esac
done

require_gcloud
require_project_id
require_project_exists
VM=$CPU_VM
gcloud iam service-accounts describe "$VM_SA_EMAIL" >/dev/null 2>&1 \
  || die "service account ${VM_SA_EMAIL} missing - run ./bootstrap_project.sh first"

DISK_MONTH="$(awk -v g="$DATA_DISK_SIZE_GB" -v r="$(disk_rate "$DATA_DISK_TYPE")" 'BEGIN { printf "%.2f", g * r }')"
BOOT_MONTH="$(awk -v g="$CPU_BOOT_DISK_GB" -v r="$(disk_rate "$BOOT_DISK_TYPE")" 'BEGIN { printf "%.2f", g * r }')"
log "storage while stopped: data disk ~${DISK_MONTH} USD/month + boot disk ~${BOOT_MONTH} USD/month; running: ~$(rate_for "$CPU_MACHINE_TYPE" STANDARD) USD/h"
if (( DATA_DISK_SIZE_GB > 100 )) && [[ $INSTALL_POSTGRES != 1 ]]; then
  warn "DATA_DISK_SIZE_GB=${DATA_DISK_SIZE_GB} without Postgres - 50 GB is plenty and disks can only grow"
fi

# ------------------------------------------------------------------ data disk
step "Data disk ${DATA_DISK} (${DATA_DISK_SIZE_GB} GB ${DATA_DISK_TYPE}, ${ZONE})"
if gcloud compute disks describe "$DATA_DISK" --zone="$ZONE" --format='value(name)' >/dev/null 2>&1; then
  log "exists ($(gcv compute disks describe "$DATA_DISK" --zone="$ZONE" --format='value(sizeGb)') GB) - kept as is"
else
  gcloud compute disks create "$DATA_DISK" --zone="$ZONE" --size="${DATA_DISK_SIZE_GB}GB" \
    --type="$DATA_DISK_TYPE" --labels=purpose=provex
fi

# ------------------------------------------------------------------ VM
prepare_vm_files
MD_FILES="$(metadata_files_arg cpu)"
MD_KV="$(metadata_kv_arg cpu "provex-install-postgres=${INSTALL_POSTGRES},provex-pg-db=${PG_DB},provex-pg-user=${PG_USER},provex-data-device=${DATA_DISK_DEVICE}")"

if (( RECREATE )) && load_instance "$VM"; then
  confirm "Delete VM ${VM} in ${I_ZONE} (its boot disk is deleted; data disk ${DATA_DISK} is KEPT)?" || die "aborted"
  gcloud compute instances delete "$VM" --zone="$I_ZONE" --keep-disks=data --quiet
fi

STARTED_BY_US=0
if load_instance "$VM"; then
  VM_ZONE=$I_ZONE
  step "VM ${VM} exists (${I_STATUS}, ${I_MACHINE}, ${VM_ZONE}) - refreshing its metadata"
  [[ $VM_ZONE == "$ZONE" ]] || warn "${VM} is in ${VM_ZONE} but ZONE=${ZONE} (the data disk must be in the VM's zone)"
  gcloud compute instances add-metadata "$VM" --zone="$VM_ZONE" --metadata-from-file="$MD_FILES" --metadata="$MD_KV" >/dev/null
  ATTACHED="$(gcv compute instances describe "$VM" --zone="$VM_ZONE" --format='value(disks[].source.basename())')" || ATTACHED=""
  if [[ ";${ATTACHED};" != *";${DATA_DISK};"* ]]; then
    log "attaching data disk ${DATA_DISK}"
    gcloud compute instances attach-disk "$VM" --zone="$VM_ZONE" --disk="$DATA_DISK" \
      --device-name="$DATA_DISK_DEVICE" --mode=rw
  fi
  if [[ $I_MACHINE != "$CPU_MACHINE_TYPE" ]]; then
    if [[ $I_STATUS == TERMINATED ]]; then
      log "changing machine type ${I_MACHINE} -> ${CPU_MACHINE_TYPE}"
      gcloud compute instances set-machine-type "$VM" --zone="$VM_ZONE" --machine-type="$CPU_MACHINE_TYPE"
    else
      warn "machine type is ${I_MACHINE}, config says ${CPU_MACHINE_TYPE}: stop the VM (./down.sh) and re-run to change it"
    fi
  fi
else
  step "Creating VM ${VM} (${CPU_MACHINE_TYPE}, ${CPU_IMAGE_FAMILY}, ${CPU_BOOT_DISK_GB} GB boot) in ${ZONE}"
  gcloud compute instances create "$VM" \
    --zone="$ZONE" \
    --machine-type="$CPU_MACHINE_TYPE" \
    --image-family="$CPU_IMAGE_FAMILY" --image-project="$CPU_IMAGE_PROJECT" \
    --boot-disk-size="${CPU_BOOT_DISK_GB}GB" --boot-disk-type="$BOOT_DISK_TYPE" \
    --disk="name=${DATA_DISK},device-name=${DATA_DISK_DEVICE},mode=rw,boot=no,auto-delete=no" \
    --service-account="$VM_SA_EMAIL" --scopes=cloud-platform \
    --metadata-from-file="$MD_FILES" --metadata="$MD_KV" \
    --labels=purpose=provex,role=cpu \
    --no-restart-on-failure
  STARTED_BY_US=1
fi

if (( NO_WAIT )); then
  ensure_running "$VM" $(( INTERACTIVE_MAX_HOURS * 60 )) nowait
  log "not waiting (--no-wait). The watchdog powers ${VM} off after ${IDLE_MINUTES} idle minutes."
  exit 0
fi
ensure_running "$VM" $(( INTERACTIVE_MAX_HOURS * 60 ))
vm_ssh "$VM" "$I_ZONE" 'cat /run/provex-startup.done; df -h /data | tail -n 1; cat ~/.provex-sync 2>/dev/null; ls /data/provex 2>/dev/null | head -n 20' || true

if (( KEEP_RUNNING )) || (( STARTED_BY_US == 0 )); then
  log "${VM} left RUNNING. The watchdog powers it off after ${IDLE_MINUTES} idle min (work inside 'tmux new -s provex-job' to keep it up)."
else
  step "First-boot setup verified - stopping ${VM} (it costs nothing but disk while stopped)"
  stop_vm "$VM" "$I_ZONE"
  log "stopped. Use it with: ./run_job.sh cpu <hours> <job> -- '<cmd>'   or   ./ssh_vm.sh cpu"
fi
