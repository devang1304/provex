#!/usr/bin/env bash
# status.sh - what is running, what it costs, how much credit is left.
#   * instances: zone, status, machine type, provisioning model, last start, live uptime and the
#     estimated cost since that start (rate table in config.env, + external IP)
#   * disks: size, type, attached-to, monthly cost (billed even while VMs are stopped)
#   * bucket size, cumulative estimated compute spend from the job ledger
#   * days left on the credit (CREDIT_EXPIRY), the configured budget, console links
# Exit status 1 with a loud WARNING when a VM is RUNNING without a provex-job tmux session
# (checked over ssh), or the estimated spend exceeds BUDGET_WARN_FRACTION of BUDGET_USD.
#
# Usage: ./status.sh [--no-ssh]     (--no-ssh: skip the per-VM tmux check; running VMs then warn)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NO_SSH=0
for arg in "$@"; do
  case $arg in
    --no-ssh) NO_SSH=1 ;;
    -h|--help) usage_from_header "${INFRA_DIR}/status.sh"; exit 0 ;;
    *) die "unknown option: $arg" ;;
  esac
done
require_gcloud
require_project_id
NOW=$(date +%s)
WARNINGS=()
RUNNING_VMS=()
add() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", a + b }'; }

# ------------------------------------------------------------------ instances
step "Instances in ${PROJECT_ID}"
mapfile -t ROWS < <(gcv compute instances list \
  --format="csv[no-heading,separator='|'](name,zone.basename(),status,machineType.basename(),scheduling.provisioningModel,lastStartTimestamp)" 2>/dev/null || true)
RUN_COST=0
FMT='%-18s %-15s %-11s %-15s %-9s %-19s %8s %7s %8s\n'
# shellcheck disable=SC2059
printf "$FMT" NAME ZONE STATUS MACHINE MODEL LAST_START UPTIME USD/H EST_USD
for row in "${ROWS[@]}"; do
  [[ -n $row ]] || continue
  IFS='|' read -r name zone status mt model last <<< "$row"
  model=${model:-STANDARD}
  rate="$(rate_for "$mt" "$model")"
  uptime="-" cost="-"
  if [[ $status == RUNNING ]]; then
    up_s=$(( NOW - $(iso_to_epoch "$last") ))
    uptime="$(awk -v s="$up_s" 'BEGIN { printf "%dh%02dm", s / 3600, (s % 3600) / 60 }')"
    cost="$(awk -v s="$up_s" -v r="$rate" -v ip="$RATE_EXTERNAL_IP_H" 'BEGIN { printf "%.2f", s / 3600 * (r + ip) }')"
    RUN_COST="$(add "$RUN_COST" "$cost")"
    RUNNING_VMS+=("${name}|${zone}")
  fi
  # shellcheck disable=SC2059
  printf "$FMT" "$name" "$zone" "$status" "$mt" "$model" "${last:0:19}" "$uptime" "$rate" "$cost"
done
(( ${#ROWS[@]} )) || log "no instances"

# ------------------------------------------------------------------ disks
step "Disks (billed every hour, VM running or not)"
mapfile -t DROWS < <(gcv compute disks list \
  --format="csv[no-heading,separator='|'](name,zone.basename(),sizeGb,type.basename(),users.map().basename().list(),creationTimestamp)" 2>/dev/null || true)
STORE_MONTH=0 STORE_ACCRUED=0
DFMT='%-22s %-15s %8s %-13s %-22s %10s\n'
# shellcheck disable=SC2059
printf "$DFMT" NAME ZONE SIZE_GB TYPE ATTACHED_TO USD/MONTH
for row in "${DROWS[@]}"; do
  [[ -n $row ]] || continue
  IFS='|' read -r name zone size type users created <<< "$row"
  month="$(awk -v g="$size" -v r="$(disk_rate "$type")" 'BEGIN { printf "%.2f", g * r }')"
  days="$(awk -v n="$NOW" -v c="$(iso_to_epoch "$created")" 'BEGIN { d = (n - c) / 86400; printf "%.2f", (d < 0 ? 0 : d) }')"
  STORE_MONTH="$(add "$STORE_MONTH" "$month")"
  STORE_ACCRUED="$(add "$STORE_ACCRUED" "$(awk -v m="$month" -v d="$days" 'BEGIN { printf "%.2f", m * d / 30.4 }')")"
  # shellcheck disable=SC2059
  printf "$DFMT" "$name" "$zone" "$size" "$type" "${users:--}" "$month"
done
log "storage burn: ${STORE_MONTH} USD/month = $(awk -v m="$STORE_MONTH" 'BEGIN { printf "%.2f", m / 30.4 }') USD/day; ~${STORE_ACCRUED} USD accrued since the disks were created"

# ------------------------------------------------------------------ bucket
step "Bucket ${BUCKET}"
if BSIZE="$(gcv storage du "$BUCKET" --summarize --readable-sizes 2>/dev/null)"; then
  printf '%s\n' "$BSIZE"
else
  warn "cannot read ${BUCKET}"
fi

# ------------------------------------------------------------------ ledger
step "Job ledger (${BUCKET}/ledger/runs.csv)"
LEDGER="$(ledger_cat)" || LEDGER=""
PENDING="$(gcv storage cat "${BUCKET}/ledger/pending/*.csv" 2>/dev/null | grep -v '^job,')" || PENDING=""
RUNS=0 LEDGER_COST=0 LEDGER_HOURS=0
if [[ -n $LEDGER || -n $PENDING ]]; then
  read -r RUNS LEDGER_COST LEDGER_HOURS < <(printf '%s\n%s\n' "$LEDGER" "$PENDING" \
    | awk -F, '$1 != "job" && NF >= 13 { n++; c += $13; h += $9 } END { printf "%d %.2f %.2f\n", n, c, h }')
  log "runs: ${RUNS}   billable VM hours: ${LEDGER_HOURS}   estimated compute cost: ${LEDGER_COST} USD"
  log "last 5 runs:"
  printf '%s\n' "$LEDGER" | awk -F, 'NR > 1 && NF >= 13' | tail -n 5 \
    | awk -F, '{ printf "  %-30s %-12s %-14s %-8s %s  exit=%-4s %-12s %6.2fh %7.2f USD\n", $1, $2, $4, $5, $6, $10, $11, $9, $13 }'
else
  log "no ledger rows yet"
fi

# ------------------------------------------------------------------ credit & budget
step "Credit & budget"
EXP_EPOCH="$(iso_to_epoch "$CREDIT_EXPIRY")"
DAYS_LEFT=$(( (EXP_EPOCH - NOW) / 86400 ))
log "credit window ${CREDIT_START_DATE} .. ${CREDIT_EXPIRY}: ${DAYS_LEFT} days left"
(( DAYS_LEFT >= 0 )) || WARNINGS+=("the credit expired on ${CREDIT_EXPIRY} - everything is now billed to the card")
BA="$(resolve_billing_account)"
BUDGET_LINE=""
if [[ -n $BA ]]; then
  BUDGET_LINE="$(gcv billing budgets list --billing-account="$BA" --billing-project="$PROJECT_ID" \
    --filter="displayName=${BUDGET_NAME}" \
    --format='value(amount.specifiedAmount.units,amount.specifiedAmount.currencyCode,budgetFilter.creditTypesTreatment)' 2>/dev/null | head -n 1)" || BUDGET_LINE=""
fi
log "budget '${BUDGET_NAME}': ${BUDGET_LINE:-NOT FOUND} (config: ${BUDGET_USD} ${BUDGET_CURRENCY}; kill switch at ${KILL_AT})"
[[ -n $BUDGET_LINE ]] || WARNINGS+=("budget ${BUDGET_NAME} not found - run ./bootstrap_project.sh")
EST_TOTAL="$(awk -v a="$LEDGER_COST" -v b="$RUN_COST" -v c="$STORE_ACCRUED" 'BEGIN { printf "%.2f", a + b + c }')"
PCT="$(awk -v t="$EST_TOTAL" -v b="$BUDGET_USD" 'BEGIN { printf "%.1f", (b > 0 ? 100 * t / b : 0) }')"
log "ESTIMATED spend: ${EST_TOTAL} USD = ledger ${LEDGER_COST} + running now ${RUN_COST} + storage ${STORE_ACCRUED} -> ${PCT}% of ${BUDGET_USD} USD"
log "(estimate: ignores idle minutes before watchdog shutdowns, bucket and network; the billing console is authoritative)"
if awk -v t="$EST_TOTAL" -v b="$BUDGET_USD" -v f="$BUDGET_WARN_FRACTION" 'BEGIN { exit !(t > b * f) }'; then
  WARNINGS+=("estimated spend ${EST_TOTAL} USD is above $(awk -v f="$BUDGET_WARN_FRACTION" 'BEGIN { printf "%d", f * 100 }')% of the ${BUDGET_USD} USD budget")
fi

# ------------------------------------------------------------------ running VMs without a job
if (( ${#RUNNING_VMS[@]} )); then
  step "Running VMs: is a provex-job active?"
  for entry in "${RUNNING_VMS[@]}"; do
    name=${entry%%|*}
    zone=${entry##*|}
    if (( NO_SSH )); then
      WARNINGS+=("${name} is RUNNING (tmux check skipped with --no-ssh)")
      continue
    fi
    case "$(job_session_state "$name" "$zone")" in
      BUSY) log "${name}: provex-job running ($(vm_ssh "$name" "$zone" 'cat ~/jobs/.active 2>/dev/null' 2>/dev/null || echo '?')) - OK" ;;
      FREE) WARNINGS+=("${name} is RUNNING with NO provex-job tmux session - an idle VM costs money. Stop it: ./down.sh --stop") ;;
      *) WARNINGS+=("${name} is RUNNING and its tmux session could not be checked over ssh") ;;
    esac
  done
fi

# ------------------------------------------------------------------ links
step "Links"
echo "  billing reports : https://console.cloud.google.com/billing/${BA:-<billing-account>}/reports?project=${PROJECT_ID}"
echo "  credits         : https://console.cloud.google.com/billing/${BA:-<billing-account>}/credits"
echo "  budgets         : https://console.cloud.google.com/billing/${BA:-<billing-account>}/budgets"
echo "  VM instances    : https://console.cloud.google.com/compute/instances?project=${PROJECT_ID}"

if (( ${#WARNINGS[@]} )); then
  loud "WARNING" "${WARNINGS[@]}"
  exit 1
fi
log "OK - nothing is burning money unexpectedly."
