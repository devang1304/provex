# PROVEX v2 on Google Cloud (`infra/gcp`)

Scripts that run PROVEX jobs on GCP **without wasting money**: a VM runs only while a job runs,
and it powers itself off when the job ends, even if the laptop that started it is asleep.

* Laptop side: bash scripts for **Git Bash on Windows** with the Windows gcloud CLI (Linux and
  macOS work too). gcloud is not on Git Bash's PATH; `lib.sh` prepends `GCLOUD_BIN_DIR` from
  `config.env`.
* VM side: Ubuntu 22.04 (CPU VM) and the Deep Learning VM image (GPU VM).
* All settings live in `config.env`. Personal overrides go in `config.local.env` (git-ignored).
  Any value can also be overridden for one command, e.g. `ZONE=us-central1-b ./create_gpu_vm.sh`.

| File | Purpose |
|---|---|
| `config.env` | project, zones, machine types, budget, guardrails, approximate rate table |
| `lib.sh` | shared helpers, sourced by every script |
| `bootstrap_project.sh` | project, billing link (closed/Free Trial check), APIs, service accounts, bucket, Pub/Sub topic, budget, kill-switch function |
| `request_quotas.sh` | GPU quota preferences via the Cloud Quotas API; `--status` only shows them |
| `create_cpu_vm.sh` | CPU dev/test VM (e2-standard-4) + 50 GB data disk at `/data`; the repo is at `/data/provex` |
| `create_gpu_vm.sh` | GPU VM: L4 Spot by default, or `--ondemand` / `--a100`, with zone fallback |
| `sync_code.sh` | push the local working tree (uncommitted edits included) to `/data/provex` on a VM |
| `run_job.sh` | one job: start the VM, sync, tmux `provex-job`, timeout, logs to GCS, ledger, self-shutdown |
| `ssh_vm.sh` | shell or one command on a VM (starts it if it is stopped) |
| `status.sh` | what runs and what it costs, credit days left. Exits 1 for an idle running VM or spend above 80 % |
| `down.sh` | `--stop` (default), `--delete-vms`, or `--all` (bucket and project too) |
| `test_guardrails.sh` | proves the 4 guardrails on a throw-away e2-micro |
| `killswitch/` | Cloud Function that stops every VM at 95 % of the budget (`python main.py --selftest`) |
| `vm/startup_*.sh`, `vm/provex-watchdog.sh` | VM boot setup and the idle watchdog (systemd timer) |
| `vm/job_runner.sh` | VM side of `run_job.sh` |
| `vm/setup_python.sh` | uv venv: torch 2.5.1 (cu124 or cpu), torch_scatter, torch_geometric 2.7.0, numpy<2, ... |
| `Makefile` | optional shortcuts (GNU make is not installed in Git Bash by default) |

## What you must do (once, on the laptop)

1. **Account and billing.** Done: project `provex-v2-dd1304`, billing upgraded from the Free
   Trial (`BILLING_UPGRADED=1`). A Free Trial account cannot use GPUs. Upgrading
   (Console > Billing > Overview > Upgrade) keeps the remaining credit.
2. `gcloud auth login`. You only need `gcloud auth application-default login` if you want to run
   `python killswitch/main.py --dry-run` against the real project.
3. `gcloud components install beta`, for `request_quotas.sh`.
4. Check `GCLOUD_BIN_DIR` in `config.env` (the Cloud SDK `bin` folder).
5. **First SSH.** The first `gcloud compute ssh` from Windows creates
   `%USERPROFILE%\.ssh\google_compute_engine.ppk` (PuTTY) and adds the public key to the project
   metadata. If PuTTY asks to cache a host key, answer `y`. `create_*_vm.sh` does this for you; if
   ssh ever hangs, run `./ssh_vm.sh cpu` once interactively.

## First run, in this order

```bash
cd /e/provex/repo/infra/gcp            # Git Bash
./bootstrap_project.sh                 # idempotent; ~5 min (function build). Re-run any time.
./request_quotas.sh --status           # filed GPU quota preferences + effective limits
./test_guardrails.sh                   # ~45-60 min on an e2-micro, < 0.10 USD: all 4 must PASS
./create_cpu_vm.sh                     # disk + VM, clone + git lfs pull + extract, then STOPS the VM
./run_job.sh cpu 1 venv-cpu-1 -- 'bash infra/gcp/vm/setup_python.sh'
./run_job.sh cpu 0.5 tests-cpu-1 --follow -- 'python -m pytest -q tests'
./status.sh
# when GPU work starts:
./create_gpu_vm.sh                     # Spot L4 (--ondemand if Spot keeps getting preempted)
./run_job.sh gpu 1 venv-gpu-1 -- 'bash infra/gcp/vm/setup_python.sh'
./run_job.sh gpu 2 tests-gpu-1 -- 'python -m pytest -q tests'
```

`run_job.sh` always pushes your working tree first, so the VM runs exactly what is on your disk.
GitHub only needs the repo for the first clone and the Git LFS `artifact.tar.gz`. The branch
`REPO_BRANCH=v2` falls back to the default branch until `v2` is pushed.

## Daily use

```bash
./run_job.sh cpu 2 explain-w3 -- 'python -m explanations.window_analysis'
./run_job.sh --status cpu              # active job + log tail
./run_job.sh --abort cpu               # stop it now; logs are uploaded, then the VM powers off
gcloud storage cat gs://provex-v2-dd1304-artifacts/jobs/explain-w3/job.log
./sync_code.sh cpu                     # push edits to a VM that is already running
./ssh_vm.sh cpu                        # interactive; run `tmux new -s provex-job` to keep it up
./status.sh                            # run this whenever in doubt; exit 1 = something burns money
./down.sh                              # stop everything now
```

Each job writes `~/jobs/<job>/{command.sh,meta.env,job.log,monitor.log,heartbeat,exit_code}` on
the VM. These are mirrored to `gs://<bucket>/jobs/<job>/` every 5 minutes and at the end, and one
row is appended to `gs://<bucket>/ledger/runs.csv` (job, vm, zone, machine type, provisioning
model, start, end, hours, billable hours, exit code, status, rate, estimated cost, commit).

## Cost model (approximate us-central1 list prices, see the rate table in `config.env`)

| Resource | Price | Notes |
|---|---|---|
| CPU VM e2-standard-4 | 0.134 USD/h | only while running |
| GPU VM g2-standard-8 + L4 | 0.35 USD/h Spot, 0.85 USD/h on-demand | Spot is the default |
| GPU VM a2-highgpu-1g + A100 | 1.20 USD/h Spot, 3.67 USD/h on-demand | `--a100`; 12 vCPU, so stop the CPU VM first (CPUS_ALL_REGIONS=12) |
| External IP | 0.005 USD/h | only while running |
| Data disk 50 GB pd-balanced | ~5 USD/month | **billed even when every VM is stopped** |
| Boot disks 50 GB (CPU) + 100 GB (GPU) | ~5 + ~10 USD/month | delete the GPU VM between GPU phases (`./down.sh --delete-vms` keeps the data disk) |
| Bucket | 0.02 USD/GB-month | logs are tiny; old versions expire |
| Kill-switch function, Pub/Sub, budget | ~0 | |

The data disk is 50 GB, not 500 GB, because Postgres is no longer needed. 500 GB of pd-balanced
would cost ~50 USD/month, about 150 USD of the credit, with every VM stopped. Disks can grow
(`gcloud compute disks resize`) but never shrink. Use 500 only with `INSTALL_POSTGRES=1`.

Example: 40 Spot L4 hours (14 USD) + 100 CPU hours (13.4 USD) + 3 months of disks (~60 USD,
or ~30 USD if the GPU VM is deleted between phases) is ~60-90 USD of the 280 USD budget.

## Guardrails (each one alone stops the spending)

1. **Job-end shutdown.** `vm/job_runner.sh` runs the command under `timeout <max_hours>h`. Whatever
   happens (success, failure, timeout, `--abort`, tmux killed), it uploads logs, writes the ledger
   row and runs `sudo shutdown -h now`. A guest shutdown puts a GCE VM in TERMINATED, which bills
   no compute.
2. **Idle watchdog.** systemd timer: first check 10 min after boot, then every 5 min. The VM is
   idle when there is no tmux session `provex-job`, GPU utilisation is below 5 % (or there is no
   GPU), and the 1-min load is below 0.5. After `IDLE_MINUTES` (15) of continuous idleness it
   powers off. It never powers off while `provex-job` exists. ssh sessions are ignored.
   Overrides: `/etc/provex-watchdog.conf`.
3. **GCE hard cap.** Before every start the scripts set
   `max-run-duration = max_hours + 20 min` (`INTERACTIVE_MAX_HOURS` = 4 h for ssh/sync starts)
   with termination action STOP. GCE enforces it even if the VM hangs.
4. **Budget kill switch.** The budget (280 USD, **gross cost: credits excluded**, period
   2026-10-10..2027-01-08, alerts at 25/50/75/90/100 %) publishes to Pub/Sub. The function stops
   every RUNNING/STAGING/PROVISIONING VM in all zones once cost reaches 95 % of the budget. Billing
   data lags by hours, so this is the last line of defence.
5. Other settings: `--no-restart-on-failure` everywhere; Spot uses termination action STOP, never
   DELETE. The create scripts stop the VM again after the first-boot setup. `status.sh` exits 1 when
   a VM runs without a job or the estimated spend passes 80 %.

Disks and the bucket are billed while VMs are stopped; no guardrail stops that, only `down.sh`
does.

## Teardown

```bash
./down.sh                 # stop all VMs (default)
./down.sh --delete-vms    # delete VMs + boot disks, keep the data disk and the bucket
./down.sh --all           # VMs, all disks, bucket (typed confirmation), budget + project (typed confirmation)
```

Deleting the project stops all billing right away. It can be undone for 30 days with
`gcloud projects undelete provex-v2-dd1304`.

## Windows / Git Bash notes

* gcloud on Windows prints CRLF, so captured output is passed through `tr -d '\r'` (`gcv` in `lib.sh`).
* MSYS rewrites POSIX-looking arguments for native programs. gcloud's launcher needs that, so it
  stays on, and `MSYS2_ARG_CONV_EXCL` exempts `--command=`, `--metadata=`, `--message=` and
  `gs://` so remote paths like `/data/provex` reach the VM unchanged. Local files are always
  passed to gcloud as relative paths from `infra/gcp`.
* `lib.sh` points gcloud's sh launcher at the SDK's bundled Python (`CLOUDSDK_PYTHON`) so it
  does not pick the WindowsApps `python3` stub or an unsupported Python.
* `gcloud compute ssh` uses PuTTY (plink/pscp) on Windows. Code sync pipes a tarball through
  ssh stdin (`SYNC_METHOD=pipe`) and falls back to `gcloud compute scp` automatically (or set
  `SYNC_METHOD=scp`).
* VM readiness is read from the serial console (`PROVEX-STARTUP-DONE epoch=...`), so no ssh is
  needed while a VM boots.
* E: is exFAT, which has no exec bits. Run scripts as `./x.sh` from Git Bash (shebang) or
  `bash x.sh`. When committing, mark them executable for Linux users with
  `git add --chmod=+x infra/gcp/*.sh infra/gcp/vm/*.sh`.

## Verify once the account is used (flags / names that change over time)

* DLVM image family `common-cu124-ubuntu-2204-py310`: `create_gpu_vm.sh` checks it and lists the
  alternatives. Any family with a CUDA 12 driver runs the cu124 torch wheels.
* Whether `--accelerator` is accepted next to g2/a2 machine types; otherwise set
  `GPU_ATTACH_ACCELERATOR_FLAG=0`.
* `gcloud billing budgets create --start-date/--end-date`. If it is refused, the script falls back
  to a monthly budget and says so.
* `gcloud compute instances set-scheduling --max-run-duration`. A failure only prints a warning.
* `gcloud storage cp --if-generation-match` and the `gs://...#generation` syntax, used by the
  ledger on the VM.
