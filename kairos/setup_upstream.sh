#!/usr/bin/env bash
# Build kairos/cadets_e3 = upstream KAIROS DARPA/CADETS_E3 at the pinned commit + our patches.
# Upstream has no license, so its code is fetched here instead of being committed to this repo.
set -euo pipefail
SHA=0e0b633beb46a1117c0a6d63be5d2481b59ac0dc
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "fetching ProvenanceAnalytics/kairos@${SHA}"
curl -fsSL "https://codeload.github.com/ProvenanceAnalytics/kairos/tar.gz/${SHA}" | tar -xz -C "$WORK"
SRC="$WORK/kairos-${SHA}/DARPA/CADETS_E3"
# integrity: blob SHA-1s recorded in UPSTREAM.md
grep -E '^\| `[A-Za-z_]+(\.py)?`.*DARPA/CADETS_E3/' "$HERE/UPSTREAM.md" | while IFS='|' read -r _ local _ _ blob _; do
  f="$(echo "$local" | tr -d ' `')"; want="$(echo "$blob" | tr -d ' `')"
  got="$(git hash-object "$SRC/$f")"
  [[ "$got" == "$want" ]] || { echo "integrity FAIL for $f: $got != $want"; exit 1; }
done
echo "upstream files verified"
( cd "$SRC" && for p in "$HERE"/patches/*.patch; do git apply --whitespace=nowarn "$p"; echo "applied $(basename "$p")"; done )
OUT="$HERE/cadets_e3"
mkdir -p "$OUT"
cp -a "$SRC/." "$OUT/"
chmod +x "$OUT/run_pipeline.sh" 2>/dev/null || true
echo "built $OUT"
