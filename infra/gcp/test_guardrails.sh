#!/usr/bin/env bash
# test_guardrails.sh - prove the four money guardrails on a throw-away e2-micro VM.
#   (a) run_job 'sleep 120; exit 1'     -> the VM ends TERMINATED and the ledger has the row (exit 1)
#   (b) watchdog with IDLE_MINUTES=2    -> an idle VM powers itself off
#                                          (first check 10 min after boot, then every 5 min)
#   (c) run_job with max_hours=0.01     -> the command is killed by timeout (exit 124) and the VM stops
#   (d) fake budget alert (0.96 x budget) published to the budget topic -> the kill switch stops the VM
# The test VM (TEST_VM, default provex-guardtest) is created first and deleted at the end.
# Takes ~45-60 min and costs well under 0.10 USD. Prints PASS/FAIL per check; exit 1 on any FAIL.
#
# Usage: ./test_guardrails.sh [--only a,b,c,d] [--keep-vm] [--allow-stop-others]
#   --allow-stop-others  run (d) even if other VMs are RUNNING - the kill switch stops ALL VMs
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TEST_VM="${TEST_VM:-provex-guardtest}"
ONLY="a,b,c,d" KEEP_VM=0 ALLOW_OTHERS=0
while (( $# )); do
  case $1 in
    --only) ONLY=${2:?--only needs a list like a,c}; shift ;;
    --keep-vm) KEEP_VM=1 ;;
    --allow-stop-others) ALLOW_OTHERS=1 ;;
    -h|--help) usage_from_header "${INFRA_DIR}/test_guardrails.sh"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done
require_gcloud
require_project_id
require_project_exists
gcloud iam service-accounts describe "$VM_SA_EMAIL" >/dev/null 2>&1 || die "run ./bootstrap_project.sh first"

declare -A RESULT=()
TS="$(date +%Y%m%d-%H%M%S)"
want() { [[ ",${ONLY}," == *",$1,"* ]]; }
record() {
  RESULT[$1]="$2"
  if [[ $2 == PASS* ]]; then log "($1) $2"; else warn "($1) $2"; fi
}
ledger_exit_code() {  # exit_code column of the ledger row for job $1 ("" if missing)
  ledger_cat | awk -F, -v j="$1" '$1 == j { c = $10 } END { print c }' || true
}
running_others() {  # names of RUNNING VMs other than the test VM, space-separated
  gcv compute instances list --filter="status=RUNNING" --format='value(name)' 2>/dev/null \
    | awk -v t="$TEST_VM" '$1 != t' | tr '\n' ' ' || true
}

cleanup() {
  local rc=$?
  if (( KEEP_VM == 0 )) && load_instance "$TEST_VM"; then
    log "deleting test VM ${TEST_VM}"
    gcloud compute instances delete "$TEST_VM" --zone="$I_ZONE" --quiet >/dev/null 2>&1 \
      || warn "could not delete ${TEST_VM} - delete it by hand: ./down.sh --delete-vms"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

OTHERS="$(running_others)"
if [[ -n ${OTHERS// /} ]]; then
  warn "other VMs are RUNNING: ${OTHERS}(they count against CPUS_ALL_REGIONS=12, and (d) will be skipped)"
fi

# ------------------------------------------------------------------ test VM
step "Test VM ${TEST_VM} (e2-micro, ${ZONE})"
if load_instance "$TEST_VM"; then
  log "exists (${I_STATUS}) - reusing it"
else
  prepare_vm_files
  gcloud compute instances create "$TEST_VM" --zone="$ZONE" --machine-type=e2-micro \
    --image-family="$CPU_IMAGE_FAMILY" --image-project="$CPU_IMAGE_PROJECT" \
    --boot-disk-size=10GB --boot-disk-type=pd-standard \
    --service-account="$VM_SA_EMAIL" --scopes=cloud-platform \
    --metadata-from-file="$(metadata_files_arg cpu)" \
    --metadata="$(CLONE_REPO_ON_BOOT=0 metadata_kv_arg test "provex-install-postgres=0")" \
    --labels=purpose=provex,role=guardtest --no-restart-on-failure
fi

# ------------------------------------------------------------------ (a)
test_a() {
  local job="guardtest-a-${TS}" st code
  step "(a) run_job 'sleep 120; exit 1' -> VM powers off + ledger row"
  if ! bash "${INFRA_DIR}/run_job.sh" "$TEST_VM" 1 "$job" -- 'sleep 120; exit 1'; then
    record a "FAIL - run_job.sh failed"; return
  fi
  if ! st="$(wait_status "$TEST_VM" "$ZONE" 'TERMINATED|STOPPED' 900)"; then
    record a "FAIL - VM still ${st} 15 min after the launch"; return
  fi
  code="$(ledger_exit_code "$job")"
  if [[ $code == 1 ]]; then record a "PASS - VM ${st}, ledger row exit_code=1"
  else record a "FAIL - VM ${st} but ledger exit_code='${code:-missing}'"; fi
}

# ------------------------------------------------------------------ (b)
test_b() {
  local st out
  step "(b) idle watchdog with IDLE_MINUTES=2"
  if ! ( ensure_running "$TEST_VM" $(( INTERACTIVE_MAX_HOURS * 60 )) ); then
    record b "FAIL - could not start / reach the VM"; return
  fi
  load_instance "$TEST_VM" || { record b "FAIL - VM vanished"; return; }
  out="$(vm_ssh "$TEST_VM" "$I_ZONE" "echo IDLE_MINUTES=2 | sudo tee /etc/provex-watchdog.conf >/dev/null && echo PROVEX-OK")" || out=""
  [[ $out == *PROVEX-OK* ]] || { record b "FAIL - could not configure the watchdog"; return; }
  log "watchdog set to 2 idle minutes; waiting up to 35 min for the VM to power itself off ..."
  if ! st="$(wait_status "$TEST_VM" "$I_ZONE" 'TERMINATED|STOPPED' 2100)"; then
    record b "FAIL - VM still ${st} after 35 min"; return
  fi
  if gcv compute instances get-serial-port-output "$TEST_VM" --zone="$I_ZONE" --port=1 2>/dev/null | grep -aq 'provex-watchdog: VM idle'; then
    record b "PASS - powered off by the watchdog (${st})"
  else
    record b "PASS - VM ${st} within the window (watchdog line not found on the serial console)"
  fi
}

# ------------------------------------------------------------------ (c)
test_c() {
  local job="guardtest-c-${TS}" st code
  step "(c) timeout: max_hours=0.01 (36 s) running 'sleep 600'"
  if ! bash "${INFRA_DIR}/run_job.sh" "$TEST_VM" 0.01 "$job" -- 'sleep 600'; then
    record c "FAIL - run_job.sh failed"; return
  fi
  if ! st="$(wait_status "$TEST_VM" "$ZONE" 'TERMINATED|STOPPED' 900)"; then
    record c "FAIL - VM still ${st} 15 min after the launch"; return
  fi
  code="$(ledger_exit_code "$job")"
  if [[ $code == 124 || $code == 137 ]]; then record c "PASS - killed by timeout (exit ${code}), VM ${st}"
  else record c "FAIL - VM ${st} but ledger exit_code='${code:-missing}' (expected 124)"; fi
}

# ------------------------------------------------------------------ (d)
test_d() {
  local st cost msg others
  step "(d) budget kill switch: fake alert at 0.96 x ${BUDGET_USD} ${BUDGET_CURRENCY}"
  others="$(running_others)"
  if [[ -n ${others// /} ]] && (( ! ALLOW_OTHERS )); then
    record d "SKIP - other VMs are RUNNING (${others}); the kill switch would stop them too. Re-run: --only d --allow-stop-others"
    return
  fi
  if ! ( ensure_running "$TEST_VM" $(( INTERACTIVE_MAX_HOURS * 60 )) nowait ); then
    record d "FAIL - could not start the VM"; return
  fi
  load_instance "$TEST_VM" || { record d "FAIL - VM vanished"; return; }
  cost="$(awk -v b="$BUDGET_USD" 'BEGIN { printf "%.2f", b * 0.96 }')"
  msg="{\"budgetDisplayName\":\"${BUDGET_NAME}\",\"alertThresholdExceeded\":0.9,\"costAmount\":${cost},\"costIntervalStart\":\"${CREDIT_START_DATE}T00:00:00Z\",\"budgetAmount\":${BUDGET_USD},\"budgetAmountType\":\"SPECIFIED_AMOUNT\",\"currencyCode\":\"${BUDGET_CURRENCY}\",\"provexGuardrailTest\":true}"
  gcloud pubsub topics publish "$BUDGET_TOPIC" --message="$msg" >/dev/null \
    || { record d "FAIL - could not publish to ${BUDGET_TOPIC}"; return; }
  log "published costAmount=${cost} budgetAmount=${BUDGET_USD}; waiting up to 7 min for the kill switch ..."
  if st="$(wait_status "$TEST_VM" "$I_ZONE" 'STOPPING|TERMINATED|STOPPED' 420)"; then
    record d "PASS - kill switch stopped the VM (${st})"
  else
    record d "FAIL - VM still ${st} 7 min after the fake alert"
  fi
  log "kill-switch log (newest first):"
  gcv functions logs read "$KILLSWITCH_FUNCTION" --gen2 --region="$REGION" --limit=10 2>/dev/null || true
}

for t in a b c d; do
  if want "$t"; then "test_${t}"; fi
done

step "Results"
FAILED=0
for t in a b c d; do
  want "$t" || continue
  printf '  (%s) %s\n' "$t" "${RESULT[$t]:-NOT RUN}"
  if [[ ${RESULT[$t]:-FAIL} == FAIL* ]]; then FAILED=1; fi
done
if (( FAILED )); then
  loud "At least one guardrail check FAILED - do not run long jobs until it is fixed."
  exit 1
fi
log "all selected guardrail checks passed (SKIP = not run, see above)"
