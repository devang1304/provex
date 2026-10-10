#!/usr/bin/env bash
# down.sh - stop or tear down PROVEX resources.
#
# Usage: ./down.sh [--stop | --delete-vms | --all] [--yes]
#   --stop        (default) stop every running VM in the project; disks, data and bucket stay
#   --delete-vms  delete every VM with its boot disk; KEEP the data disk and the bucket
#   --all         clean teardown: delete VMs, then ALL disks, then (typed confirmation) the
#                 bucket with every object version, then (typed confirmation) the budget and
#                 the whole project (recoverable for 30 days with `gcloud projects undelete`)
#   --yes         skip the y/N questions (typed confirmations are never skipped)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODE=stop
for arg in "$@"; do
  case $arg in
    --stop) MODE=stop ;;
    --delete-vms) MODE=delete-vms ;;
    --all) MODE=all ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) usage_from_header "${INFRA_DIR}/down.sh"; exit 0 ;;
    *) die "unknown option: $arg" ;;
  esac
done
require_gcloud
require_project_id

instance_rows() { gcv compute instances list --format="csv[no-heading,separator='|'](name,zone.basename(),status)" 2>/dev/null || true; }

stop_all() {
  local row name zone status n=0
  declare -A by_zone=()
  while IFS= read -r row; do
    [[ -n $row ]] || continue
    IFS='|' read -r name zone status <<< "$row"
    if [[ $status =~ ^(RUNNING|STAGING|PROVISIONING|REPAIRING)$ ]]; then
      by_zone[$zone]+="${name} "
      n=$(( n + 1 ))
    fi
  done < <(instance_rows)
  if (( n == 0 )); then log "no running VMs"; return 0; fi
  for zone in "${!by_zone[@]}"; do
    log "stopping in ${zone}: ${by_zone[$zone]}"
    # shellcheck disable=SC2086
    gcloud compute instances stop ${by_zone[$zone]} --zone="$zone" --quiet || warn "stop failed in ${zone}"
  done
}

delete_vms() {
  local row name zone status n=0
  declare -A by_zone=()
  while IFS= read -r row; do
    [[ -n $row ]] || continue
    IFS='|' read -r name zone status <<< "$row"
    by_zone[$zone]+="${name} "
    n=$(( n + 1 ))
  done < <(instance_rows)
  if (( n == 0 )); then log "no VMs"; return 0; fi
  for zone in "${!by_zone[@]}"; do log "  ${zone}: ${by_zone[$zone]}"; done
  confirm "Delete these ${n} VM(s) and their boot disks? (data disks are kept)" || { log "skipped"; return 1; }
  for zone in "${!by_zone[@]}"; do
    # shellcheck disable=SC2086
    gcloud compute instances delete ${by_zone[$zone]} --zone="$zone" --keep-disks=data --quiet \
      || warn "delete failed in ${zone}"
  done
}

delete_disks() {
  local row name zone users n=0
  declare -A by_zone=()
  while IFS= read -r row; do
    [[ -n $row ]] || continue
    IFS='|' read -r name zone users <<< "$row"
    if [[ -n $users ]]; then warn "disk ${name} is still attached to ${users} - skipped"; continue; fi
    by_zone[$zone]+="${name} "
    n=$(( n + 1 ))
  done < <(gcv compute disks list --format="csv[no-heading,separator='|'](name,zone.basename(),users.map().basename().list())" 2>/dev/null || true)
  if (( n == 0 )); then log "no unattached disks"; return 0; fi
  for zone in "${!by_zone[@]}"; do log "  ${zone}: ${by_zone[$zone]}"; done
  confirm "Delete these ${n} disk(s)? The data disk holds /data/provex (code, artifacts, venv)." || { log "disks kept"; return 0; }
  for zone in "${!by_zone[@]}"; do
    # shellcheck disable=SC2086
    gcloud compute disks delete ${by_zone[$zone]} --zone="$zone" --quiet || warn "disk delete failed in ${zone}"
  done
}

delete_bucket() {
  if ! gcloud storage buckets describe "$BUCKET" --format='value(name)' >/dev/null 2>&1; then
    log "bucket ${BUCKET} does not exist"; return 0
  fi
  gcloud storage du "$BUCKET" --summarize --readable-sizes 2>/dev/null || true
  if confirm_typed "Delete bucket ${BUCKET} with ALL objects and versions (job logs, ledger, results)?" "${BUCKET#gs://}"; then
    gcloud storage rm --recursive --all-versions "$BUCKET"
    log "bucket deleted"
  else
    log "bucket kept"
  fi
}

delete_project() {
  local ba budgets b
  if ! confirm_typed "FINAL STEP: delete the whole project ${PROJECT_ID} (function, topic, IAM, everything left)?" "$PROJECT_ID"; then
    log "project kept"; return 0
  fi
  ba="$(resolve_billing_account)"
  if [[ -n $ba ]]; then
    budgets="$(gcv billing budgets list --billing-account="$ba" --billing-project="$PROJECT_ID" \
      --filter="displayName=${BUDGET_NAME}" --format='value(name)' 2>/dev/null)" || budgets=""
    for b in $budgets; do
      gcloud billing budgets delete "$b" --billing-project="$PROJECT_ID" --quiet && log "deleted budget ${b}" \
        || warn "could not delete budget ${b} (delete it in Console > Billing > Budgets)"
    done
  fi
  gcloud projects delete "$PROJECT_ID" --quiet
  log "project ${PROJECT_ID} shut down: billing stops now; it is purged after 30 days (undo: gcloud projects undelete ${PROJECT_ID})"
}

case $MODE in
  stop)
    step "Stopping every running VM in ${PROJECT_ID}"
    stop_all ;;
  delete-vms)
    step "Deleting every VM in ${PROJECT_ID} (data disk + bucket are kept)"
    delete_vms || true ;;
  all)
    step "Teardown of ${PROJECT_ID}"
    stop_all
    delete_vms || die "VMs not deleted - teardown stopped"
    step "Disks"
    delete_disks
    step "Bucket"
    delete_bucket
    step "Project"
    delete_project ;;
esac
log "done. Check with ./status.sh"
