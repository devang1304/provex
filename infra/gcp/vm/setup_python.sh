#!/usr/bin/env bash
# =============================================================================
# vm/setup_python.sh - build / refresh the PROVEX Python env on a VM with uv (idempotent).
# Run it as a job so it gets logs + auto-shutdown (cwd of a job is /data/provex):
#   ./run_job.sh cpu 1 venv-cpu-1 -- 'bash infra/gcp/vm/setup_python.sh'
#   ./run_job.sh gpu 1 venv-gpu-1 -- 'bash infra/gcp/vm/setup_python.sh'
#
# Usage: setup_python.sh [auto|cpu|cu124] [--recreate]
#   auto (default) = cu124 when nvidia-smi works, else cpu
# Result: /data/provex/.venv (python 3.10) with torch 2.5.1 (+cu124 wheels from
#   download.pytorch.org/whl/cu124, or +cpu), torch_scatter (data.pyg.org wheels for torch
#   2.5.0), torch_geometric 2.7.0, numpy<2, scipy, scikit-learn, pytz, xxhash, networkx, tqdm,
#   pytest, pyyaml. run_job.sh activates this venv for every job.
# Env: PROVEX_REPO_DIR (/data/provex), PROVEX_VENV (<repo>/.venv), PROVEX_PYTHON (3.10),
#      EXTRA_PACKAGES (space-separated extra requirements)
# =============================================================================
set -euo pipefail

VARIANT=auto RECREATE=0
for a in "$@"; do
  case $a in
    auto|cpu|cu124) VARIANT=$a ;;
    --recreate) RECREATE=1 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

REPO_DIR="${PROVEX_REPO_DIR:-/data/provex}"
VENV="${PROVEX_VENV:-${REPO_DIR}/.venv}"
PY_VER="${PROVEX_PYTHON:-3.10}"
TORCH_VERSION=2.5.1
PYG_TORCH=2.5.0                     # data.pyg.org publishes one wheel index per torch minor
export UV_LINK_MODE=copy            # uv cache (boot disk) and venv (/data) are different filesystems

if [[ $VARIANT == auto ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then VARIANT=cu124; else VARIANT=cpu; fi
fi
TORCH_INDEX="https://download.pytorch.org/whl/${VARIANT}"
PYG_LINKS="https://data.pyg.org/whl/torch-${PYG_TORCH}+${VARIANT}.html"

UV="$(command -v uv 2>/dev/null || true)"
if [[ -z $UV && -x /usr/local/bin/uv ]]; then UV=/usr/local/bin/uv; fi
if [[ -z $UV && -x $HOME/.local/bin/uv ]]; then UV=$HOME/.local/bin/uv; fi
if [[ -z $UV ]]; then
  echo "== uv not found - installing it into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
  UV="$HOME/.local/bin/uv"
fi
echo "== $("$UV" --version) | variant ${VARIANT} | venv ${VENV} | python ${PY_VER}"

mkdir -p "$(dirname "$VENV")"
if (( RECREATE )) && [[ -d $VENV ]]; then echo "== removing ${VENV}"; rm -rf "$VENV"; fi
if [[ ! -x $VENV/bin/python ]]; then "$UV" venv --python "$PY_VER" "$VENV"; fi
PY="$VENV/bin/python"

if [[ -f $VENV/.provex-variant && "$(cat "$VENV/.provex-variant")" != "$VARIANT" ]]; then
  echo "== venv was built for $(cat "$VENV/.provex-variant") - replacing torch with the ${VARIANT} build"
  "$UV" pip uninstall --python "$PY" torch torch-scatter || true
fi

echo "== torch ${TORCH_VERSION} (${VARIANT}) from ${TORCH_INDEX}"
"$UV" pip install --python "$PY" "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"

echo "== torch_scatter (prebuilt wheel) from ${PYG_LINKS}"
"$UV" pip install --python "$PY" torch-scatter --find-links "$PYG_LINKS" --only-binary :all:

echo "== torch_geometric + scientific stack"
# shellcheck disable=SC2086
"$UV" pip install --python "$PY" "torch_geometric==2.7.0" "numpy<2" scipy scikit-learn pytz xxhash \
  networkx tqdm pytest pyyaml ${EXTRA_PACKAGES:-}
echo "$VARIANT" > "$VENV/.provex-variant"

echo "== verification"
PROVEX_VARIANT="$VARIANT" "$PY" - <<'PYEOF'
import os
import sys

import numpy
import torch
import torch_geometric
import torch_scatter

print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "| torch_geometric", torch_geometric.__version__,
      "| torch_scatter", torch_scatter.__version__)
want_cuda = os.environ["PROVEX_VARIANT"] != "cpu"
if want_cuda:
    assert torch.cuda.is_available(), "CUDA wheels installed but torch.cuda.is_available() is False"
    print("GPU:", torch.cuda.get_device_name(0))
dev = "cuda" if want_cuda else "cpu"
out = torch_scatter.scatter_add(torch.ones(4, device=dev), torch.tensor([0, 0, 1, 1], device=dev))
assert out.tolist() == [2.0, 2.0], out
print("torch_scatter.scatter_add OK on", dev)
PYEOF
echo "== done. Activate with: source ${VENV}/bin/activate  (run_job.sh does it for every job)"
