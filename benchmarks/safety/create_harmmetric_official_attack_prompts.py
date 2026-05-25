"""Create solver-format prompt files from HarmMetric_Eval's official dataset.

This is not a fallback or synthetic corpus. It extracts the released
HarmMetric_Eval rows whose ``attack_method`` is one of:
PAP, PAIR, GPTFuzzer, DAN, AutoDAN, or GCG.

The output is intended as model-input data, so ``response`` and ``score`` are
left empty for your downstream solver/evaluator to fill.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "HarmMetric_Eval" / "data" / "dataset.jsonl"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "benchmarks" / "safety" / "attack_method_prompts" / "harmmetric_official"
)

ATTACK_METHODS = ("PAP", "PAIR", "GPTFuzzer", "DAN", "AutoDAN", "GCG")


def load_rows(dataset_path: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_prompts: dict[str, set[str]] = {method: set() for method in ATTACK_METHODS}

    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            attack_method = row.get("attack_method", "")
            if attack_method not in ATTACK_METHODS:
                continue

            harmful_prompt = row["harmful_prompt"]
            if harmful_prompt in seen_prompts[attack_method]:
                continue
            seen_prompts[attack_method].add(harmful_prompt)

            rows_by_method[attack_method].append(
                {
                    "harmful_prompt": harmful_prompt,
                    "source": row.get("source", ""),
                    "response": "",
                    "attack_method": attack_method,
                    "attack_dimension": attack_method,
                    "sample_index": 0,
                    "score": "",
                    "attack_cases_source": str(dataset_path),
                    "harmmetric_prompt_id": row.get("prompt_id"),
                    "harmmetric_response_id": row.get("response_id"),
                    "harmmetric_original_response_category": row.get("response_category", ""),
                }
            )

    return rows_by_method


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows_by_method = load_rows(args.dataset)
    combined: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}

    for method in ATTACK_METHODS:
        rows = rows_by_method.get(method, [])
        write_json(args.output_dir / f"safety_solver_{method.lower()}.json", rows)
        combined.extend(rows)
        summary[method] = {
            "rows": len(rows),
            "status": "written" if rows else "missing",
            "source": str(args.dataset),
        }

    write_json(args.output_dir / "safety_solver_harmmetric_official_attack_methods_all.json", combined)
    write_json(args.output_dir / "summary.json", summary)

    print(f"Wrote {len(combined)} combined prompt(s) -> {args.output_dir}")
    for method in ATTACK_METHODS:
        print(f"{method}: {summary[method]['status']} ({summary[method]['rows']} row(s))")


if __name__ == "__main__":
    main()
