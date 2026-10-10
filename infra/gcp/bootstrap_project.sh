#!/usr/bin/env bash
# bootstrap_project.sh - one-time, idempotent setup of the PROVEX GCP project.
#
#   1 project (create if missing)          6 Pub/Sub topic budget-alerts
#   2 billing link + closed/Free-Trial check 7 budget: gross cost (credits EXCLUDED), credit
#   3 APIs                                     window CREDIT_START_DATE..CREDIT_EXPIRY, 25..100 %
#   4 service accounts provex-vm (GCS          alerts -> Pub/Sub
#     write on the bucket) + provex-killswitch 8 kill-switch Cloud Function (gen2, python312)
#     (compute.instanceAdmin.v1)             9 GPU / CPU quota sanity check
#   5 bucket: uniform access, versioning, lifecycle (tmp/ 7 d, old versions), no soft delete
#
# Safe to re-run: every step checks before it creates. Usage: ./bootstrap_project.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[[ ${1:-} =~ ^(-h|--help)$ ]] && { usage_from_header "${INFRA_DIR}/bootstrap_project.sh"; exit 0; }

require_gcloud
require_project_id
log "gcloud account: ${GCLOUD_ACCOUNT}   project: ${PROJECT_ID}   region: ${REGION}"

# ------------------------------------------------------------------ 1. project
step "1/9 Project ${PROJECT_ID}"
if gcloud projects describe "$PROJECT_ID" --format='value(projectId)' >/dev/null 2>&1; then
  log "project exists"
else
  log "creating project ${PROJECT_ID}"
  gcloud projects create "$PROJECT_ID" --name="$PROJECT_NAME"
fi
PROJECT_NUMBER="$(gcv projects describe "$PROJECT_ID" --format='value(projectNumber)')"
log "project number: ${PROJECT_NUMBER}"

# ------------------------------------------------------------------ 2. billing
step "2/9 Billing"
LINKED="$(gcv billing projects describe "$PROJECT_ID" --format='value(billingAccountName)' 2>/dev/null)" || LINKED=""
LINKED=${LINKED#billingAccounts/}
BA="${BILLING_ACCOUNT_ID:-$LINKED}"
if [[ -z $BA ]]; then
  mapfile -t OPEN_ACCOUNTS < <(gcv billing accounts list --filter=open=true --format='value(name.basename())' 2>/dev/null || true)
  if (( ${#OPEN_ACCOUNTS[@]} == 1 )) && [[ -n ${OPEN_ACCOUNTS[0]} ]]; then
    BA=${OPEN_ACCOUNTS[0]}
    log "using the only open billing account: ${BA}"
  else
    gcloud billing accounts list || true
    die "set BILLING_ACCOUNT_ID in infra/gcp/config.local.env (found ${#OPEN_ACCOUNTS[@]} open accounts)"
  fi
fi
BA_OPEN="$(gcv billing accounts describe "$BA" --format='value(open)' 2>/dev/null)" || BA_OPEN=""
case ${BA_OPEN,,} in
  true) log "billing account ${BA} is open" ;;
  false)
    loud "Billing account ${BA} is CLOSED." \
         "A Free Trial account closes when its credit is used up or expires." \
         "Re-open / upgrade it in Console > Billing before going on."
    exit 1 ;;
  *) warn "could not read billing account ${BA} (needs Billing Account Viewer) - continuing" ;;
esac
if [[ $LINKED != "$BA" ]]; then
  log "linking ${PROJECT_ID} to billing account ${BA}"
  gcloud billing projects link "$PROJECT_ID" --billing-account="$BA" >/dev/null
else
  log "project already linked to ${BA}"
fi
if [[ $BILLING_UPGRADED != 1 ]]; then
  loud "BILLING_UPGRADED=${BILLING_UPGRADED} in config.env." \
       "If this billing account is still a FREE TRIAL, GPUs are not available and GPU quota" \
       "requests are rejected. Click 'Upgrade' in Console > Billing > Overview (the remaining" \
       "credit is kept), then set BILLING_UPGRADED=1. CPU-only work is fine without it."
fi

# ------------------------------------------------------------------ 3. APIs
step "3/9 APIs"
APIS=(compute.googleapis.com storage.googleapis.com cloudquotas.googleapis.com
      billingbudgets.googleapis.com cloudbilling.googleapis.com pubsub.googleapis.com
      cloudfunctions.googleapis.com run.googleapis.com cloudbuild.googleapis.com
      eventarc.googleapis.com artifactregistry.googleapis.com logging.googleapis.com
      iam.googleapis.com cloudresourcemanager.googleapis.com serviceusage.googleapis.com)
ENABLED="$(gcv services list --enabled --format='value(config.name)' 2>/dev/null)" || ENABLED=""
MISSING=()
for api in "${APIS[@]}"; do
  grep -qx "$api" <<< "$ENABLED" || MISSING+=("$api")
done
if (( ${#MISSING[@]} )); then
  log "enabling: ${MISSING[*]}"
  gcloud services enable "${MISSING[@]}"
else
  log "all ${#APIS[@]} APIs already enabled"
fi

# ------------------------------------------------------------------ 4. service accounts
step "4/9 Service accounts"
ensure_sa() {  # ensure_sa NAME DISPLAY_NAME
  local email="$1@${PROJECT_ID}.iam.gserviceaccount.com"
  if gcloud iam service-accounts describe "$email" >/dev/null 2>&1; then
    log "service account ${email} exists"
  else
    gcloud iam service-accounts create "$1" --display-name="$2"
  fi
}
bind_project_role() {  # bind_project_role MEMBER ROLE (idempotent; retried for IAM propagation)
  retry 6 10 gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="$1" --role="$2" --condition=None --quiet >/dev/null \
    || die "could not grant $2 to $1"
  log "granted $2 to $1"
}
ensure_sa "$VM_SA_NAME" "PROVEX VMs (job logs + ledger in GCS)"
ensure_sa "$KILLSWITCH_SA_NAME" "PROVEX budget kill switch"
bind_project_role "serviceAccount:${KS_SA_EMAIL}" roles/compute.instanceAdmin.v1
bind_project_role "serviceAccount:${KS_SA_EMAIL}" roles/run.invoker
bind_project_role "serviceAccount:${VM_SA_EMAIL}" roles/logging.logWriter
bind_project_role "serviceAccount:${VM_SA_EMAIL}" roles/monitoring.metricWriter
# Cloud Functions gen2 builds run as the Compute Engine default service account (2024+ default).
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
if retry 12 10 gcloud iam service-accounts describe "$COMPUTE_SA" >/dev/null 2>&1; then
  bind_project_role "serviceAccount:${COMPUTE_SA}" roles/cloudbuild.builds.builder
else
  warn "default compute service account not visible yet; if the function build fails with a permission error, re-run this script"
fi

# ------------------------------------------------------------------ 5. bucket
step "5/9 Bucket ${BUCKET}"
if gcloud storage buckets describe "$BUCKET" --format='value(name)' >/dev/null 2>&1; then
  log "bucket exists"
else
  gcloud storage buckets create "$BUCKET" --location="$REGION" --default-storage-class=STANDARD \
    --uniform-bucket-level-access --public-access-prevention
fi
cat > "${BUILD_DIR}/lifecycle.json" <<EOF
{
  "rule": [
    {"action": {"type": "Delete"}, "condition": {"age": ${TMP_PREFIX_TTL_DAYS}, "matchesPrefix": ["tmp/"]}},
    {"action": {"type": "Delete"}, "condition": {"isLive": false, "matchesPrefix": ["tmp/"], "daysSinceNoncurrentTime": 1}},
    {"action": {"type": "Delete"}, "condition": {"isLive": false, "numNewerVersions": ${NONCURRENT_VERSIONS_KEEP}}},
    {"action": {"type": "Delete"}, "condition": {"isLive": false, "daysSinceNoncurrentTime": ${NONCURRENT_TTL_DAYS}}}
  ]
}
EOF
gcloud storage buckets update "$BUCKET" --versioning --lifecycle-file="${BUILD_DIR}/lifecycle.json" >/dev/null
log "versioning on; lifecycle: tmp/ objects deleted after ${TMP_PREFIX_TTL_DAYS} d; keep ${NONCURRENT_VERSIONS_KEEP} old versions, none older than ${NONCURRENT_TTL_DAYS} d"
if gcloud storage buckets update "$BUCKET" --clear-soft-delete >/dev/null 2>&1; then
  log "soft delete disabled (versioning already keeps history; soft-deleted bytes would stay billable 7 d)"
else
  warn "could not disable soft delete on ${BUCKET} (deleted objects stay billable for 7 days)"
fi
for role in roles/storage.objectAdmin roles/storage.legacyBucketReader; do
  retry 6 10 gcloud storage buckets add-iam-policy-binding "$BUCKET" \
    --member="serviceAccount:${VM_SA_EMAIL}" --role="$role" >/dev/null \
    || die "could not grant ${role} on ${BUCKET}"
  log "granted ${role} on ${BUCKET} to ${VM_SA_EMAIL}"
done

# ------------------------------------------------------------------ 6. Pub/Sub topic
step "6/9 Pub/Sub topic ${BUDGET_TOPIC}"
if gcloud pubsub topics describe "$BUDGET_TOPIC" >/dev/null 2>&1; then
  log "topic exists"
else
  gcloud pubsub topics create "$BUDGET_TOPIC"
fi

# ------------------------------------------------------------------ 7. budget
step "7/9 Budget ${BUDGET_NAME}: ${BUDGET_USD} ${BUDGET_CURRENCY}, ${CREDIT_START_DATE}..${CREDIT_EXPIRY}"
# --billing-project: the Budget API rejects end-user credentials without a quota project.
EXISTING="$(gcv billing budgets list --billing-account="$BA" --billing-project="$PROJECT_ID" \
  --filter="displayName=${BUDGET_NAME}" --format='value(name)' 2>/dev/null | head -n 1)" || EXISTING=""
if [[ -n $EXISTING ]]; then
  log "budget exists: ${EXISTING}"
  gcloud billing budgets describe "$EXISTING" --billing-account="$BA" --billing-project="$PROJECT_ID" \
    --format="yaml(displayName,amount,budgetFilter.creditTypesTreatment,budgetFilter.customPeriod,budgetFilter.calendarPeriod,thresholdRules,notificationsRule.pubsubTopic)" || true
  log "to change it: gcloud billing budgets delete ${EXISTING} --billing-project=${PROJECT_ID}  (then re-run)"
else
  # exclude-all-credits: track GROSS cost. With credits included the Free-Trial credit nets the
  # cost to ~0 and neither the alerts nor the kill switch would ever fire.
  BUDGET_ARGS=(--billing-account="$BA" --display-name="$BUDGET_NAME"
    --budget-amount="${BUDGET_USD}${BUDGET_CURRENCY}"
    --filter-projects="projects/${PROJECT_NUMBER}"
    --credit-types-treatment=exclude-all-credits
    --notifications-rule-pubsub-topic="projects/${PROJECT_ID}/topics/${BUDGET_TOPIC}"
    --billing-project="$PROJECT_ID")
  for t in 0.25 0.5 0.75 0.9 1.0; do BUDGET_ARGS+=(--threshold-rule="percent=${t}"); done
  if OUT="$(gcloud billing budgets create "${BUDGET_ARGS[@]}" --start-date="$CREDIT_START_DATE" --end-date="$CREDIT_EXPIRY" 2>&1)"; then
    log "budget created (custom period ${CREDIT_START_DATE}..${CREDIT_EXPIRY}, alerts at 25/50/75/90/100 %)"
  else
    printf '%s\n' "$OUT" | tr -d '\r' >&2
    if grep -qiE 'unrecognized arguments|start-date|end-date' <<< "$OUT"; then
      warn "custom budget period not accepted - creating a MONTHLY budget instead (it resets every month!)"
      gcloud billing budgets create "${BUDGET_ARGS[@]}" --calendar-period=month
    else
      die "budget creation failed (see above)"
    fi
  fi
fi

# ------------------------------------------------------------------ 8. kill switch
step "8/9 Kill-switch function ${KILLSWITCH_FUNCTION} (stops every VM at ${KILL_AT} x budget)"
retry 3 30 gcloud functions deploy "$KILLSWITCH_FUNCTION" \
  --gen2 --region="$REGION" --runtime=python312 \
  --source=killswitch --entry-point=on_budget_alert \
  --trigger-topic="$BUDGET_TOPIC" \
  --service-account="$KS_SA_EMAIL" --trigger-service-account="$KS_SA_EMAIL" \
  --set-env-vars="KILL_AT=${KILL_AT},GCP_PROJECT_ID=${PROJECT_ID}" \
  --memory=256Mi --timeout=120s --min-instances=0 --max-instances=1 --quiet \
  || die "function deploy failed (open the Cloud Build log URL printed above)"
log "kill switch deployed; test it with ./test_guardrails.sh --only d"

# ------------------------------------------------------------------ 9. quota sanity check
step "9/9 Quota check"
GPU_LIMIT="$(gcv compute project-info describe --flatten=quotas --filter='quotas.metric=GPUS_ALL_REGIONS' --format='value(quotas.limit)' 2>/dev/null | head -n 1)" || GPU_LIMIT=""
CPU_LIMIT="$(gcv compute project-info describe --flatten=quotas --filter='quotas.metric=CPUS_ALL_REGIONS' --format='value(quotas.limit)' 2>/dev/null | head -n 1)" || CPU_LIMIT=""
log "GPUS_ALL_REGIONS limit: ${GPU_LIMIT:-unknown}    CPUS_ALL_REGIONS limit: ${CPU_LIMIT:-unknown}"
if [[ -z $GPU_LIMIT || ${GPU_LIMIT%%.*} == 0 ]]; then
  loud "The global GPU quota (GPUS_ALL_REGIONS) is 0 - no GPU VM can start yet." \
       "Free Trial billing accounts never get GPU quota: upgrade first (Console > Billing > Overview)," \
       "then run ./request_quotas.sh (or check filed requests: ./request_quotas.sh --status)."
fi

step "Done"
cat <<EOF
Project ${PROJECT_ID} is ready. Next:
  ./request_quotas.sh --status      # GPU quota preferences + effective limits
  ./test_guardrails.sh              # prove the 4 money guardrails (~45-60 min, < 0.10 USD)
  ./create_cpu_vm.sh                # CPU dev/test VM + data disk (stops itself after setup)
  ./status.sh                       # what runs, what it costs
EOF
