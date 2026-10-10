#!/bin/bash
# =============================================================================
# vm/startup_gpu.sh - startup script of the GPU VM (Deep Learning VM image). Root, every boot.
#   1. fetch + source startup_common.sh; idle watchdog first
#   2. common setup: packages, uv, user provex, repo in /data/provex on the boot disk
#      (git clone + git lfs pull + extract artifact.tar.gz on first boot)
#   3. verify nvidia-smi - the DLVM installs the NVIDIA driver on first boot because of the
#      instance metadata install-nvidia-driver=True (can take ~5-10 min)
# =============================================================================
set -Eeuo pipefail

mkdir -p /opt/provex
for _i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS -H 'Metadata-Flavor: Google' -o /opt/provex/startup_common.sh \
       'http://metadata.google.internal/computeMetadata/v1/instance/attributes/provex-startup-common'; then
    break
  fi
  sleep 3
done
if [[ ! -s /opt/provex/startup_common.sh ]]; then
  echo "PROVEX-STARTUP-FAILED epoch=$(date +%s) role=gpu reason=metadata-provex-startup-common-missing" > /dev/ttyS0
  exit 1
fi
# shellcheck source=startup_common.sh
source /opt/provex/startup_common.sh

verify_gpu() {  # waits up to 15 min for the driver; never fails the startup (status is reported)
  local i
  for (( i = 1; i <= 60; i++ )); do
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      log "GPU OK: $(nvidia-smi -L | tr '\n' ' ')"
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
      PROVEX_STATUS_EXTRA="gpu=ok"
      return 0
    fi
    if (( i == 1 )); then log "waiting for the NVIDIA driver (installed by the DLVM on first boot) ..."; fi
    sleep 15
  done
  log "ERROR: nvidia-smi still fails after 15 min - GPU jobs will not work. See /var/log/syslog (driver install)."
  PROVEX_STATUS_EXTRA="gpu=MISSING"
}

provex_begin "$(md_attr provex-role gpu)"
mkdir -p /data
common_setup
verify_gpu
provex_done
