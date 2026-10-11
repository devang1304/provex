# PROVEX v2. Every target runs from the repository root.
# The experiment targets need the CADETS E3 artifacts and trained KAIROS models; see README.md.

.PHONY: test tables reproduce clean

# unit and end-to-end tests on a synthetic model (CPU, a few minutes)
test:
	python -m pytest -q tests

# regenerate every number, table and case study of paper/provex_v2.tex from results/
tables:
	bash scripts/run_experiments.sh tables

# the full pipeline after KAIROS training (hours on a many-core CPU machine)
reproduce:
	bash scripts/run_experiments.sh detect
	bash scripts/run_experiments.sh parity
	bash scripts/run_experiments.sh gmfit
	for d in 4_6 4_11 4_12 4_13; do bash scripts/run_experiments.sh explain $$d; done
	bash scripts/run_experiments.sh e6 4_6
	bash scripts/run_experiments.sh e6 4_12
	bash scripts/run_experiments.sh e7 4_6
	bash scripts/run_experiments.sh tables

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete