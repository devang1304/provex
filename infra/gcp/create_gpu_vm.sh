#!/usr/bin/env bash
# create_gpu_vm.sh - the GPU VM on a Deep Learning VM image (CUDA 12.x; the NVIDIA driver is
# installed on first boot via metadata install-nvidia-driver=True). No data disk: the repo is
# cloned to /data/provex on the 100 GB boot disk.
#
# Usage: ./create_gpu_vm.sh [--spot|--ondemand] [--a100] [--keep-running] [--recreate] [--no-wait]
#   --spot / --ondemand  provisioning model (default GPU_PROVISIONING=SPOT: ~60 % cheaper; may be
#                        preempted, and is then STOPPED - never deleted)
#   --a100               a2-highgpu-1g + 1x A100 instead of g2-standard-8 + 1x L4
#                        (12 vCPU: with CPUS_ALL_REGIONS=12 the CPU VM must be stopped)
#   --recreate           delete the existing GPU VM first: needed to change machine type or
#                        provisioning model, or to move zones after a stockout
#   --keep-running / --no-wait   as in create_cpu_vm.sh
# Creation tries every zone in GPU_ZONES until one has capacity (ZONE_RESOURCE_POOL_EXHAUSTED).
# An existing VM is (re)started in its zone. After the first-boot setup the VM is stopped again.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODEL=${GPU_PROVISIONING^^}
A100=0 KEEP_RUNNING=0 RECREATE=0 NO_WAIT=0
for arg in "$@"; do
  case $arg in
    --spot) MODEL=SPOT ;;
    --ondemand|--on-demand) MODEL=STANDARD ;;
    --a100) A100=1 ;;
    --keep-running) KEEP_RUNNING=1 ;;
    --recreate) RECREATE=1 ;;
    --no-wait) NO_WAIT=1 ;;
    -h|--help) usage_from_header "${INFRA_DIR}/create_gpu_vm.sh"; exit 0 ;;
    *) die "unknown option: $arg" ;;
  esac
done
[[ $MODEL =~ ^(SPOT|STANDARD)$ ]] || die "GPU_PROVISIONING must be SPOT or STANDARD"
if (( A100 )); then MT=$GPU_A100_MACHINE_TYPE; ACC=$GPU_A100_ACCELERATOR_TYPE; else MT=$GPU_MACHINE_TYPE; ACC=$GPU_ACCELERATOR_TYPE; fi
VM=$GPU_VM
RATE="$(rate_for "$MT" "$MODEL")"

require_gcloud
require_project_id
require_project_exists
gcloud iam service-accounts describe "$VM_SA_EMAIL" >/dev/null 2>&1 \
  || die "service account ${VM_SA_EMAIL} missing - run ./bootstrap_project.sh first"
if [[ $BILLING_UPGRADED != 1 ]]; then
  warn "BILLING_UPGRADED=${BILLING_UPGRADED}: Free Trial accounts cannot create GPU VMs"
fi
log "${MT} + ${ACC} (${MODEL}): ~${RATE} USD/h while running; boot disk ~$(awk -v g="$GPU_BOOT_DISK_GB" -v r="$(disk_rate "$BOOT_DISK_TYPE")" 'BEGIN { printf "%.2f", g * r }') USD/month while it exists"

# ------------------------------------------------------------------ image family
step "Image family ${GPU_IMAGE_FAMILY} (${GPU_IMAGE_PROJECT})"
IMG="$(gcv compute images describe-from-family "$GPU_IMAGE_FAMILY" --project="$GPU_IMAGE_PROJECT" --format='value(name)' 2>/dev/null)" || IMG=""
if [[ -n $IMG ]]; then
  log "latest image: ${IMG}"
else
  warn "image family ${GPU_IMAGE_FAMILY} not found. Ubuntu 22.04 families available now:"
  gcv compute images list --project="$GPU_IMAGE_PROJECT" --no-standard-images \
    --filter="family~ubuntu-2204" --format='value(family)' 2>/dev/null | sort -u | head -n 40 >&2 || true
  die "set GPU_IMAGE_FAMILY in config.env (a 'common-cu12x-ubuntu-2204-...' family; any CUDA 12 driver runs the cu124 torch wheels)"
fi

prepare_vm_files
MD_FILES="$(metadata_files_arg gpu)"
MD_KV="$(metadata_kv_arg gpu "install-nvidia-driver=True")"

if (( RECREATE )) && load_instance "$VM"; then
  confirm "Delete GPU VM ${VM} (${I_MACHINE}, ${I_MODEL}, ${I_ZONE}) incl. its boot disk (venv is rebuilt with setup_python.sh)?" || die "aborted"
  gcloud compute instances delete "$VM" --zone="$I_ZONE" --quiet
fi

STARTED_BY_US=0
if load_instance "$VM"; then
  if [[ $I_MACHINE != "$MT" || $I_MODEL != "$MODEL" ]]; then
    die "${VM} exists as ${I_MACHINE}/${I_MODEL} but you asked for ${MT}/${MODEL}. Re-run with --recreate."
  fi
  step "VM ${VM} exists (${I_STATUS}, ${I_ZONE}) - refreshing its metadata"
  gcloud compute instances add-metadata "$VM" --zone="$I_ZONE" --metadata-from-file="$MD_FILES" --metadata="$MD_KV" >/dev/null
else
  ARGS=(--machine-type="$MT"
        --image-family="$GPU_IMAGE_FAMILY" --image-project="$GPU_IMAGE_PROJECT"
        --boot-disk-size="${GPU_BOOT_DISK_GB}GB" --boot-disk-type="$BOOT_DISK_TYPE"
        --maintenance-policy=TERMINATE --no-restart-on-failure
        --service-account="$VM_SA_EMAIL" --scopes=cloud-platform
        --metadata-from-file="$MD_FILES" --metadata="$MD_KV"
        --labels=purpose=provex,role=gpu)
  if [[ $GPU_ATTACH_ACCELERATOR_FLAG == 1 ]]; then
    ARGS+=(--accelerator="type=${ACC},count=${GPU_ACCELERATOR_COUNT}")
  fi
  if [[ $MODEL == SPOT ]]; then
    ARGS+=(--provisioning-model=SPOT --instance-termination-action=STOP)
  fi
  IFS=',' read -r -a ZONES_TO_TRY <<< "$GPU_ZONES"
  CREATED=""
  for z in "${ZONES_TO_TRY[@]}"; do
    z=${z//[[:space:]]/}
    [[ -n $z ]] || continue
    step "Creating ${VM} in ${z} (${MT} + ${ACC}, ${MODEL})"
    if OUT="$(gcloud compute instances create "$VM" --zone="$z" "${ARGS[@]}" 2>&1)"; then
      printf '%s\n' "$OUT" | tr -d '\r'
      CREATED=$z
      break
    fi
    printf '%s\n' "$OUT" | tr -d '\r' >&2
    if grep -qiE 'ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|resource pool exhausted|stockout' <<< "$OUT"; then
      warn "no ${ACC} capacity in ${z} - trying the next zone"; continue
    fi
    if grep -qiE "was not found|does not exist in zone|not available in|Invalid value for field 'resource.guestAccelerators" <<< "$OUT"; then
      warn "${MT}/${ACC} not offered in ${z} - trying the next zone"; continue
    fi
    if grep -qi 'quota' <<< "$OUT"; then
      die "quota exceeded - ./request_quotas.sh --status. CPUS_ALL_REGIONS=12: stop the CPU VM before an A100 VM."
    fi
    if grep -qiE 'free trial|upgrade your' <<< "$OUT"; then
      die "GPUs need an upgraded (paid) billing account - Console > Billing > Overview > Upgrade"
    fi
    die "instance creation failed in ${z} (see above)"
  done
  [[ -n $CREATED ]] || die "no zone in GPU_ZONES (${GPU_ZONES}) had capacity. Retry later, try --ondemand / --spot, or add zones to GPU_ZONES."
  STARTED_BY_US=1
fi

if (( NO_WAIT )); then
  ensure_running "$VM" $(( INTERACTIVE_MAX_HOURS * 60 )) nowait
  log "not waiting (--no-wait). The watchdog powers ${VM} off after ${IDLE_MINUTES} idle minutes."
  exit 0
fi
READY_TIMEOUT_MIN=$(( READY_TIMEOUT_MIN > 30 ? READY_TIMEOUT_MIN : 30 ))   # driver install on first boot
ensure_running "$VM" $(( INTERACTIVE_MAX_HOURS * 60 ))
vm_ssh "$VM" "$I_ZONE" 'cat /run/provex-startup.done; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "nvidia-smi FAILED"' || true

if (( KEEP_RUNNING )) || (( STARTED_BY_US == 0 )); then
  log "${VM} left RUNNING (~${RATE} USD/h). The watchdog powers it off after ${IDLE_MINUTES} idle min."
else
  step "First-boot setup verified - stopping ${VM}"
  stop_vm "$VM" "$I_ZONE"
  log "stopped. Next: ./run_job.sh gpu 1 venv-gpu-1 -- 'bash infra/gcp/vm/setup_python.sh'"
fi
