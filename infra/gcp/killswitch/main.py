"""PROVEX budget kill switch (Cloud Functions 2nd gen, Pub/Sub trigger).

A Cloud Billing budget publishes a JSON notification to the ``budget-alerts`` topic several
times a day, for example::

    {"budgetDisplayName": "provex-v2-budget", "costAmount": 266.10, "budgetAmount": 280.0,
     "alertThresholdExceeded": 0.9, "currencyCode": "USD", ...}

When ``costAmount / budgetAmount >= KILL_AT`` (env var, default 0.95) this function STOPS every
RUNNING / STAGING / PROVISIONING Compute Engine instance of the project, in every zone
(aggregated list). Stopping keeps disks and data; only the compute billing stops. Everything it
does is logged as one JSON line per event (structured Cloud Logging).

Billing data lags by hours, so this is the last line of defence; the VM-side guardrails (job
timeout + self-shutdown, idle watchdog, GCE max-run-duration) act first.

Local tests - no cloud access and no extra packages needed (a fake Compute client is used)::

    python main.py --test 0.96      # cost = 0.96 x budget -> every active VM must be stopped
    python main.py --test 0.50      # below KILL_AT -> nothing may be stopped
    python main.py --selftest       # the whole matrix (thresholds, malformed input, API errors)

Read-only check against the real project (needs google-cloud-compute and
``gcloud auth application-default login``); lists what WOULD be stopped, stops nothing::

    python main.py --dry-run --project provex-v2-dd1304
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Any, Callable, Iterable

try:  # provided by the Cloud Functions runtime (requirements.txt); optional for local tests
    import functions_framework
except ImportError:  # pragma: no cover - local runs without the package
    functions_framework = None

DEFAULT_KILL_AT = 0.95
ACTIVE_STATES = frozenset({"RUNNING", "STAGING", "PROVISIONING"})


def log(severity: str, message: str, **fields: Any) -> None:
    """Print one JSON line; Cloud Logging turns it into a structured entry."""
    entry = {"severity": severity, "message": message, "component": "provex-killswitch"}
    entry.update(fields)
    print(json.dumps(entry, default=str), flush=True)


def kill_threshold(raw: str | None = None) -> float:
    """KILL_AT from the environment (or ``raw``), falling back to 0.95 when invalid."""
    raw = os.environ.get("KILL_AT", str(DEFAULT_KILL_AT)) if raw is None else raw
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log("ERROR", f"invalid KILL_AT={raw!r}; using {DEFAULT_KILL_AT}")
        return DEFAULT_KILL_AT
    if not 0.0 < value <= 10.0:
        log("ERROR", f"KILL_AT={value} outside (0, 10]; using {DEFAULT_KILL_AT}")
        return DEFAULT_KILL_AT
    return value


def decode_pubsub_payload(event_data: Any) -> dict:
    """Return the budget notification carried by a Pub/Sub CloudEvent's ``data``."""
    if not isinstance(event_data, dict):
        raise ValueError("CloudEvent data is not an object")
    message = event_data.get("message") or {}
    raw = message.get("data")
    if not raw:
        raise ValueError("Pub/Sub message has no data")
    payload = json.loads(base64.b64decode(raw).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("budget notification is not a JSON object")
    return payload


def spend_ratio(payload: dict) -> float:
    """costAmount / budgetAmount (raises KeyError / ValueError / TypeError on bad input)."""
    cost = float(payload["costAmount"])
    budget = float(payload["budgetAmount"])
    if budget <= 0:
        return float("inf") if cost > 0 else 0.0
    return cost / budget


def resolve_project() -> str:
    for var in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"):
        if os.environ.get(var):
            return os.environ[var]
    import google.auth  # installed with google-cloud-compute

    _, project = google.auth.default()
    if not project:
        raise RuntimeError("cannot determine the project id; set GCP_PROJECT_ID")
    return project


def make_client():
    from google.cloud import compute_v1

    return compute_v1.InstancesClient()


def list_active_instances(client, project: str) -> list[tuple[str, str, str]]:
    """(zone, name, status) of every instance in ACTIVE_STATES, all zones (aggregated list)."""
    found = []
    request = {"project": project}  # no return_partial_success: not a field in every compute_v1 release
    for scope, scoped_list in client.aggregated_list(request=request):
        for inst in getattr(scoped_list, "instances", None) or []:
            if inst.status in ACTIVE_STATES:
                found.append((scope.rsplit("/", 1)[-1], inst.name, inst.status))  # zones/us-central1-a
    return sorted(found)


def stop_instances(client, project: str, instances, dry_run: bool = False):
    """Request a stop for every instance; one failure never protects the others."""
    stopped, failed = [], []
    for zone, name, status in instances:
        if dry_run:
            log("WARNING", f"[dry-run] would stop {name}", zone=zone, status=status)
            stopped.append((zone, name))
            continue
        try:
            operation = client.stop(project=project, zone=zone, instance=name)
        except Exception as exc:  # noqa: BLE001 - keep going with the other VMs
            log("ERROR", f"failed to stop {name}: {exc}", zone=zone, status=status)
            failed.append((zone, name, str(exc)))
            continue
        log("WARNING", f"stop requested for {name}", zone=zone, previous_status=status,
            operation=getattr(operation, "name", None))
        stopped.append((zone, name))
    return stopped, failed


def handle_budget_payload(payload: dict, client, project: str, kill_at: float, dry_run: bool = False) -> dict:
    ratio = spend_ratio(payload)
    summary = {
        "budget": payload.get("budgetDisplayName"),
        "cost": payload.get("costAmount"),
        "budget_amount": payload.get("budgetAmount"),
        "currency": payload.get("currencyCode"),
        "ratio": "inf" if ratio == float("inf") else round(ratio, 4),
        "kill_at": kill_at,
        "project": project,
        "action": "none",
        "stopped": [],
        "failed": [],
    }
    if ratio < kill_at:
        log("INFO", f"spend at {ratio:.1%} of the budget < kill threshold {kill_at:.0%}: nothing to do", **summary)
        return summary
    log("CRITICAL", f"spend at {ratio:.1%} of the budget >= kill threshold {kill_at:.0%}: stopping ALL instances", **summary)
    instances = list_active_instances(client, project)
    if not instances:
        log("WARNING", "no running instances found", project=project)
    stopped, failed = stop_instances(client, project, instances, dry_run=dry_run)
    summary.update(action="dry-run" if dry_run else "stop", stopped=stopped, failed=failed)
    log("CRITICAL" if failed else "WARNING",
        f"kill switch done: {len(stopped)} stopped, {len(failed)} failed", **summary)
    return summary


def handle_event(cloud_event, client_factory: Callable = make_client,
                 project_resolver: Callable = resolve_project, kill_at: float | None = None) -> dict | None:
    """Decode + validate the notification, then act. Malformed messages are logged and ignored."""
    try:
        payload = decode_pubsub_payload(getattr(cloud_event, "data", None))
        spend_ratio(payload)  # validate before touching any API
    except (ValueError, KeyError, TypeError) as exc:
        log("ERROR", f"ignoring malformed budget notification: {exc}")
        return None
    threshold = kill_threshold() if kill_at is None else kill_at
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    return handle_budget_payload(payload, client_factory(), project_resolver(), threshold, dry_run=dry_run)


def on_budget_alert(cloud_event) -> None:
    """Cloud Functions entry point: gcloud functions deploy ... --entry-point=on_budget_alert."""
    handle_event(cloud_event)


if functions_framework is not None:
    on_budget_alert = functions_framework.cloud_event(on_budget_alert)


# ----------------------------------------------------------------------------- local tests
class _FakeInstance:
    def __init__(self, name: str, status: str):
        self.name = name
        self.status = status


class _FakeScopedList:
    def __init__(self, instances):
        self.instances = instances


class _FakeOperation:
    def __init__(self, name: str):
        self.name = name


class _FakeCloudEvent:
    def __init__(self, data):
        self.data = data


class FakeInstancesClient:
    """Stand-in for compute_v1.InstancesClient: aggregated_list() + stop()."""

    def __init__(self, inventory: dict, fail_on: Iterable[str] = ()):
        self.inventory = {zone: [_FakeInstance(n, s) for n, s in items] for zone, items in inventory.items()}
        self.fail_on = set(fail_on)
        self.stop_calls: list[tuple[str, str]] = []

    def aggregated_list(self, request):
        if not request.get("project"):
            raise ValueError("project is required")
        for zone, instances in self.inventory.items():
            yield f"zones/{zone}", _FakeScopedList(list(instances))
        yield "zones/asia-east1-a", _FakeScopedList([])  # zones without instances are normal

    def stop(self, project: str, zone: str, instance: str):
        if instance in self.fail_on:
            raise RuntimeError("simulated API error (403 permission denied)")
        self.stop_calls.append((zone, instance))
        for inst in self.inventory.get(zone, []):
            if inst.name == instance:
                inst.status = "STOPPING"
        return _FakeOperation(f"operation-fake-stop-{instance}")


FAKE_INVENTORY = {
    "us-central1-a": [("provex-cpu", "RUNNING"), ("provex-old", "TERMINATED")],
    "us-central1-b": [("provex-gpu", "RUNNING")],
    "us-central1-c": [("provex-guardtest", "STAGING")],
    "europe-west4-a": [("stray-vm", "PROVISIONING"), ("parked", "SUSPENDED")],
}
EXPECTED_ACTIVE = {("us-central1-a", "provex-cpu"), ("us-central1-b", "provex-gpu"),
                   ("us-central1-c", "provex-guardtest"), ("europe-west4-a", "stray-vm")}


def fake_event(payload) -> _FakeCloudEvent:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    return _FakeCloudEvent({
        "message": {"data": base64.b64encode(raw).decode("ascii"),
                    "attributes": {"schemaVersion": "1.0", "billingAccountId": "fake", "budgetId": "fake"}},
        "subscription": "projects/fake-project/subscriptions/eventarc-fake",
    })


def budget_payload(ratio: float, budget: float = 280.0) -> dict:
    return {"budgetDisplayName": "provex-v2-budget", "alertThresholdExceeded": 0.9,
            "costAmount": round(budget * ratio, 2), "costIntervalStart": "2026-10-10T00:00:00Z",
            "budgetAmount": budget, "budgetAmountType": "SPECIFIED_AMOUNT", "currencyCode": "USD"}


def run_scenario(name: str, event, expected: set, kill_at: float, fail_on=()) -> bool:
    client = FakeInstancesClient(FAKE_INVENTORY, fail_on=fail_on)
    try:
        handle_event(event, client_factory=lambda: client, project_resolver=lambda: "fake-project", kill_at=kill_at)
    except Exception as exc:  # noqa: BLE001 - the handler must never raise on these inputs
        print(f"FAIL  {name}: handler raised {exc!r}")
        return False
    got = {call for call in client.stop_calls}
    ok = got == expected
    detail = "" if ok else f" (expected {sorted(expected) or 'nothing'})"
    print(f"{'PASS' if ok else 'FAIL'}  {name}: stopped {sorted(got) or 'nothing'}{detail}")
    return ok


def selftest(kill_at: float) -> bool:
    above = min(kill_at + 0.01, 10.0)
    below = max(kill_at - 0.45, 0.0)
    cases = [
        ("cost above KILL_AT stops every active VM", fake_event(budget_payload(above)), EXPECTED_ACTIVE, ()),
        ("cost exactly at KILL_AT stops every active VM",
         fake_event({"costAmount": kill_at, "budgetAmount": 1.0}), EXPECTED_ACTIVE, ()),
        ("cost below KILL_AT stops nothing", fake_event(budget_payload(below)), set(), ()),
        ("malformed (non-JSON) data is ignored", fake_event(b"not-json"), set(), ()),
        ("missing costAmount is ignored", fake_event({"budgetAmount": 280.0}), set(), ()),
        ("Pub/Sub message without data is ignored", _FakeCloudEvent({"message": {}}), set(), ()),
        ("zero budget with spend stops everything", fake_event({"costAmount": 1.0, "budgetAmount": 0}),
         EXPECTED_ACTIVE, ()),
        ("a failing stop() does not protect the other VMs", fake_event(budget_payload(above)),
         EXPECTED_ACTIVE - {("us-central1-a", "provex-cpu")}, ("provex-cpu",)),
    ]
    results = [run_scenario(name, event, expected, kill_at, fail_on) for name, event, expected, fail_on in cases]
    print(f"\n{sum(results)}/{len(results)} scenarios passed (KILL_AT={kill_at})")
    return all(results)


def cli_test(ratio: float, kill_at: float) -> bool:
    payload = budget_payload(ratio)
    expected = EXPECTED_ACTIVE if spend_ratio(payload) >= kill_at else set()
    print(f"fake budget notification: cost {payload['costAmount']} / budget {payload['budgetAmount']}"
          f" (ratio {ratio}), KILL_AT {kill_at}")
    return run_scenario(f"--test {ratio}", fake_event(payload), expected, kill_at)


def cli_dry_run(project: str | None, ratio: float, kill_at: float) -> bool:
    summary = handle_budget_payload(budget_payload(ratio), make_client(), project or resolve_project(),
                                    kill_at, dry_run=True)
    print(json.dumps(summary, indent=2, default=str))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PROVEX budget kill switch - local test harness")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", type=float, metavar="RATIO", help="fake client; cost = RATIO x budget")
    mode.add_argument("--selftest", action="store_true", help="every built-in scenario with the fake client")
    mode.add_argument("--dry-run", action="store_true", help="REAL project, read-only: list what would be stopped")
    parser.add_argument("--project", help="project id for --dry-run (default: GCP_PROJECT_ID / ADC)")
    parser.add_argument("--ratio", type=float, default=1.0, help="cost/budget ratio simulated by --dry-run")
    parser.add_argument("--kill-at", type=float, default=None, help="override KILL_AT (default: env or 0.95)")
    args = parser.parse_args(argv)
    kill_at = kill_threshold() if args.kill_at is None else args.kill_at
    if args.selftest:
        ok = selftest(kill_at)
    elif args.test is not None:
        ok = cli_test(args.test, kill_at)
    else:
        ok = cli_dry_run(args.project, args.ratio, kill_at)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
