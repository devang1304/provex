# PROVEX v2 - optional GCP shortcuts. Every target is ONE script call; GNU make is NOT part of
# Git for Windows, so you can simply run the scripts from Git Bash instead
# (or install make: `winget install ezwinports.make` / `choco install make`).
#
#   make -C infra/gcp help
#   make -C infra/gcp job VM=cpu JOB=tests-1 HOURS=0.5 CMD='python -m pytest -q tests'
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.RECIPEPREFIX := >
.DEFAULT_GOAL := help

VM ?= cpu
JOB ?=
HOURS ?= 1
CMD ?=
GPU_FLAGS ?=
DOWN_FLAGS ?= --stop
JOB_FLAGS ?=
export VM JOB HOURS CMD GPU_FLAGS DOWN_FLAGS JOB_FLAGS

.PHONY: help bootstrap quotas cpu-up gpu-up status down test-guardrails job sync ssh venv tests job-status abort killswitch-selftest

help:
> @echo "PROVEX v2 GCP targets (each one is a script in infra/gcp you can run directly):"
> @echo "  bootstrap        ./bootstrap_project.sh       project, APIs, SAs, bucket, budget, kill switch"
> @echo "  quotas           ./request_quotas.sh          GPU quota preferences + status"
> @echo "  cpu-up           ./create_cpu_vm.sh           CPU dev/test VM + data disk (stops after setup)"
> @echo "  gpu-up           ./create_gpu_vm.sh GPU_FLAGS GPU VM (GPU_FLAGS=--ondemand / --a100 / --spot)"
> @echo "  status           ./status.sh                  what runs, what it costs, credit left"
> @echo "  down             ./down.sh DOWN_FLAGS         --stop (default) | --delete-vms | --all"
> @echo "  test-guardrails  ./test_guardrails.sh         prove the 4 money guardrails (~1 h, < 0.10 USD)"
> @echo "  job              ./run_job.sh VM HOURS JOB -- CMD   (JOB_FLAGS=--follow)"
> @echo "  sync / ssh       ./sync_code.sh VM / ./ssh_vm.sh VM"
> @echo "  venv             build /data/provex/.venv on VM (vm/setup_python.sh, as a job)"
> @echo "  tests            python -m pytest -q tests on VM (as a job)"
> @echo "  job-status/abort ./run_job.sh --status|--abort VM"

bootstrap:
> bash ./bootstrap_project.sh

quotas:
> bash ./request_quotas.sh

cpu-up:
> bash ./create_cpu_vm.sh

gpu-up:
> bash ./create_gpu_vm.sh $$GPU_FLAGS

status:
> bash ./status.sh

down:
> bash ./down.sh $$DOWN_FLAGS

test-guardrails:
> bash ./test_guardrails.sh

job:
> @if [ -z "$$JOB" ] || [ -z "$$CMD" ]; then echo 'usage: make job VM=cpu|gpu JOB=<name> HOURS=<h> CMD="<command>" [JOB_FLAGS=--follow]'; exit 2; fi
> bash ./run_job.sh "$$VM" "$$HOURS" "$$JOB" $$JOB_FLAGS -- "$$CMD"

sync:
> bash ./sync_code.sh "$$VM"

ssh:
> bash ./ssh_vm.sh "$$VM"

venv:
> bash ./run_job.sh "$$VM" 1 "venv-$$VM-$$(date +%Y%m%d-%H%M%S)" $$JOB_FLAGS -- 'bash infra/gcp/vm/setup_python.sh'

tests:
> bash ./run_job.sh "$$VM" "$$HOURS" "tests-$$VM-$$(date +%Y%m%d-%H%M%S)" $$JOB_FLAGS -- 'python -m pytest -q tests'

job-status:
> bash ./run_job.sh --status "$$VM"

abort:
> bash ./run_job.sh --abort "$$VM"

killswitch-selftest:
> cd killswitch && python main.py --selftest
