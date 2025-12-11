"""Generate Markdown reports for a given explanation JSON (GPT-powered)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

# pylint: disable=import-error
import config
from reporting import report_builder
# pylint: enable=import-error

# Add provex directory to sys.path to allow importing config and reporting
CAD_DIR = Path(__file__).resolve().parents[1]  # .../provex


DEFAULT_JSON = CAD_DIR / "artifact" / "explanations" / \
    "2018-04-06_11_00_00~2018-04-06_12_15_00_explanations.json"
DEFAULT_MAPPING_FALLBACK = CAD_DIR / "artifact" / \
    "explanations" / "node_mapping.json"
DEFAULT_MAPPING = Path(
    getattr(config, "NODE_MAPPING_JSON", DEFAULT_MAPPING_FALLBACK))


def _load_env(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key.strip(), value.strip())
    return values


def main() -> None:
    """Run the report generation process."""
    t0 = time.perf_counter()
    print("[report] Loading environment overrides (.env) …")
    for key, value in _load_env(CAD_DIR / ".env").items():
        os.environ.setdefault(key, value)

    json_path = Path(os.environ.get(
        "PROVEX_EXPLANATION_JSON", DEFAULT_JSON)).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Explanation JSON not found: {json_path}")

    mapping_path = Path(os.environ.get(
        "PROVEX_NODE_MAPPING_JSON", DEFAULT_MAPPING)).resolve()
    if not mapping_path.exists():
        mapping_path = None

    print(f"[report] Loading payload: {json_path}")
    t_load = time.perf_counter()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"[report] Payload loaded in {time.perf_counter() - t_load:.2f}s")

    output_dir = (CAD_DIR / "artifact" / "explanations").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_gpt = bool(os.environ.get("OPENAI_API_KEY"))
    existing_summary = data.get("gpt_summary")
    if existing_summary and not use_gpt:
        print(
            "[report] Reusing existing GPT summary from artifact (OPENAI_API_KEY not set).")
    elif use_gpt:
        print("[report] OPENAI_API_KEY detected; generating GPT narrative …")
    else:
        print("[report] No GPT available; generating report without AI narrative …")

    print("[report] Building Markdown report …")
    t_build = time.perf_counter()
    md_path, gpt_summary = report_builder.build_reports(
        data,
        output_dir,
        node_mapping_path=mapping_path,
        run_gpt=use_gpt,
        existing_summary=existing_summary,
    )
    print(
        f"[report] Report build completed in {time.perf_counter() - t_build:.2f}s")

    if not md_path:
        raise RuntimeError("Markdown report was not generated.")

    if gpt_summary:
        print("[report] Persisting GPT summary back to explanation JSON …")
        t_write = time.perf_counter()
        data["gpt_summary"] = gpt_summary
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(
            f"[report] Summary persisted in {time.perf_counter() - t_write:.2f}s")

    total = time.perf_counter() - t0
    print(f"[report] Markdown written to {md_path}")
    print(f"[report] Done in {total:.2f}s")


if __name__ == "__main__":
    main()
