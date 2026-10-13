#!/usr/bin/env bash
# request_quotas.sh - file the GPU quota PROVEX needs (preferred value 1 each) through the
# Cloud Quotas API, then print the state of every preference and the effective limits.
#
#   GPUS-ALL-REGIONS (global), NVIDIA-L4-GPUS and NVIDIA-A100-GPUS (region=$REGION)
#   + PREEMPTIBLE-NVIDIA-L4-GPUS / PREEMPTIBLE-NVIDIA-A100-GPUS when REQUEST_PREEMPTIBLE_QUOTA=1
#     (Spot VMs use preemptible quota when it exists, otherwise the normal quota)
#
# Exact quota IDs are discovered with `gcloud beta quotas info list`. Idempotent: a quota whose
# limit is already >= 1, or that already has a preference (filed here or in the console), is
# skipped. Needs: gcloud components install beta ; an UPGRADED (non Free Trial) billing account.
#
# Usage: ./request_quotas.sh [--status] [--yes]
#   --status  only show preferences + limits, file nothing
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STATUS_ONLY=0
for arg in "$@"; do
  case $arg in
    --status) STATUS_ONLY=1 ;;
    --yes) ASSUME_YES=1 ;;
    -h|--help) usage_from_header "${INFRA_DIR}/request_quotas.sh"; exit 0 ;;
    *) die "unknown option: $arg" ;;
  esac
done

require_gcloud
require_project_id
require_project_exists
QFLAGS=(--project="$PROJECT_ID" --billing-project="$PROJECT_ID")
EMAIL="${CONTACT_EMAIL:-$GCLOUD_ACCOUNT}"

if (( ! STATUS_ONLY )) && [[ $BILLING_UPGRADED != 1 ]]; then
  loud "BILLING_UPGRADED=${BILLING_UPGRADED}: Free Trial billing accounts cannot get GPU quota." \
       "Upgrade in Console > Billing > Overview (credit is kept) and set BILLING_UPGRADED=1."
  confirm "File the requests anyway?" || exit 1
fi

# limit of a compute quota id, e.g. NVIDIA-L4-GPUS-per-project-region -> metric NVIDIA_L4_GPUS
metric_limit() {
  local metric="${1%%-per-*}"
  metric="${metric//-/_}"
  if [[ $1 == *per-project-region* ]]; then
    gcv compute regions describe "$REGION" --flatten=quotas --filter="quotas.metric=${metric}" \
      --format='value(quotas.limit)' 2>/dev/null | head -n 1
  else
    gcv compute project-info describe --flatten=quotas --filter="quotas.metric=${metric}" \
      --format='value(quotas.limit)' 2>/dev/null | head -n 1
  fi
}

if (( ! STATUS_ONLY )); then
  step "Discovering compute quota IDs (gcloud beta quotas info list)"
  IDS_FILE="${BUILD_DIR}/compute-quota-ids.txt"
  gcv beta quotas info list --service=compute.googleapis.com "${QFLAGS[@]}" --format='value(quotaId)' > "$IDS_FILE" \
    || die "quotas info list failed (run: gcloud components install beta)"
  log "$(wc -l < "$IDS_FILE" | tr -d ' ') compute quota ids"

  PREFS="$(gcv beta quotas preferences list "${QFLAGS[@]}" \
    --format="csv[no-heading,separator='|'](name.basename(),quotaId,dimensions.region,quotaConfig.preferredValue,quotaConfig.grantedValue,reconciling)" 2>/dev/null)" || PREFS=""

  TARGETS=("GPUS-ALL-REGIONS:global" "NVIDIA-L4-GPUS:region" "NVIDIA-A100-GPUS:region")
  if [[ $REQUEST_PREEMPTIBLE_QUOTA == 1 ]]; then
    TARGETS+=("PREEMPTIBLE-NVIDIA-L4-GPUS:region" "PREEMPTIBLE-NVIDIA-A100-GPUS:region")
  fi

  for t in "${TARGETS[@]}"; do
    prefix=${t%%:*}
    scope=${t##*:}
    region_dim=""
    if [[ $scope == region ]]; then region_dim=$REGION; fi
    where=${region_dim:-global}
    qid="$(grep -E "^${prefix}-per-project" "$IDS_FILE" | head -n 1)" || qid=""
    if [[ -z $qid ]]; then
      warn "no quota id starting with ${prefix} - similar ids:"
      grep -i "${prefix#PREEMPTIBLE-}" "$IDS_FILE" | head -n 10 >&2 || true
      continue
    fi
    limit="$(metric_limit "$qid")" || limit=""
    if [[ -n $limit && ${limit%%.*} =~ ^[0-9]+$ ]] && (( ${limit%%.*} >= 1 )); then
      log "${qid} (${where}): limit already ${limit} - nothing to request"
      continue
    fi
    existing="$(awk -F'|' -v q="$qid" -v r="$region_dim" '$2 == q && $3 == r { print; exit }' <<< "$PREFS")"
    if [[ -n $existing ]]; then
      IFS='|' read -r pname _ _ pref granted recon <<< "$existing"
      log "${qid} (${where}): preference ${pname} already filed (preferred=${pref} granted=${granted:-?} reconciling=${recon:-?}) - skipping"
      continue
    fi
    pid="provex-$(tr '[:upper:]' '[:lower:]' <<< "$prefix")-${where}"
    args=(--preference-id="$pid" --service=compute.googleapis.com --quota-id="$qid" --preferred-value=1
          --justification="$QUOTA_JUSTIFICATION" --email="$EMAIL")
    if [[ -n $region_dim ]]; then args+=(--dimensions="region=${region_dim}"); fi
    log "requesting ${qid} = 1 (${where}) as preference ${pid}"
    gcloud beta quotas preferences create "${args[@]}" "${QFLAGS[@]}" || warn "request for ${qid} failed (see above)"
  done
fi

step "Quota preferences in ${PROJECT_ID}"
gcloud beta quotas preferences list "${QFLAGS[@]}" \
  --format="table(name.basename():label=PREFERENCE,quotaId,dimensions.region:label=REGION,quotaConfig.preferredValue:label=PREFERRED,quotaConfig.grantedValue:label=GRANTED,reconciling)" \
  || warn "could not list preferences (gcloud components install beta)"

step "Effective limits"
gcloud compute project-info describe --flatten=quotas --filter="quotas.metric~ALL_REGIONS" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" || true
gcloud compute regions describe "$REGION" --flatten=quotas --filter="quotas.metric~GPUS" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" || true
log "note: CPUS_ALL_REGIONS=12 fits the CPU VM (4 vCPU) + L4 VM (8); the A100 VM (12) needs the CPU VM stopped."
