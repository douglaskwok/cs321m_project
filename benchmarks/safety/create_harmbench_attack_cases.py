"""Create HarmBench behavior/test-case files for SafetySuite prompts.

This script handles the file-format bridge around HarmBench:

1. Convert ``safety_solver.json`` into a HarmBench ``behaviors.csv``.
2. Optionally flatten a HarmBench ``test_cases.json`` into our solver format.

It does not run HarmBench attacks itself; use ``HarmBench/generate_test_cases.py``
for that step.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUT = Path(__file__).parent / "safety_solver.json"
DEFAULT_BEHAVIORS = Path(__file__).parent / "safety_solver_behaviors.csv"


def read_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise TypeError(f"Expected {path} to contain a JSON list")
    return data


def write_behaviors(input_path: Path, output_path: Path) -> None:
    items = read_json_list(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["BehaviorID", "Behavior", "ContextString", "Tags"],
        )
        writer.writeheader()
        for idx, item in enumerate(items):
            prompt = str(item.get("harmful_prompt", "")).strip()
            if not prompt:
                continue
            writer.writerow(
                {
                    "BehaviorID": f"safety_{idx:06d}",
                    "Behavior": prompt,
                    "ContextString": "",
                    "Tags": item.get("source", ""),
                }
            )


def load_behavior_metadata(behaviors_path: Path) -> dict[str, dict]:
    with behaviors_path.open("r", encoding="utf-8", newline="") as fh:
        return {row["BehaviorID"]: row for row in csv.DictReader(fh)}


def flatten_test_cases(
    test_cases_path: Path,
    behaviors_path: Path,
    attack_method: str,
    output_path: Path,
) -> None:
    with test_cases_path.open("r", encoding="utf-8") as fh:
        test_cases = json.load(fh)
    if not isinstance(test_cases, dict):
        raise TypeError(f"Expected {test_cases_path} to contain a JSON object")

    behavior_metadata = load_behavior_metadata(behaviors_path)
    rows = []
    for behavior_id, cases in test_cases.items():
        metadata = behavior_metadata.get(behavior_id, {})
        if not isinstance(cases, list):
            cases = [cases]
        for case_idx, test_case in enumerate(cases):
            rows.append(
                {
                    "item_id": f"{behavior_id}__{attack_method}__{case_idx}",
                    "behavior_id": behavior_id,
                    "source": metadata.get("Tags", ""),
                    "base_prompt": metadata.get("Behavior", ""),
                    "attack_method": attack_method,
                    "test_case_index": case_idx,
                    "harmful_prompt": test_case,
                    "response": "",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    behaviors_parser = subparsers.add_parser("behaviors")
    behaviors_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    behaviors_parser.add_argument("--output", type=Path, default=DEFAULT_BEHAVIORS)

    flatten_parser = subparsers.add_parser("flatten")
    flatten_parser.add_argument("--test-cases", type=Path, required=True)
    flatten_parser.add_argument("--behaviors", type=Path, default=DEFAULT_BEHAVIORS)
    flatten_parser.add_argument("--attack-method", required=True)
    flatten_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "behaviors":
        write_behaviors(args.input, args.output)
        print(f"Wrote behaviors -> {args.output}")
    elif args.command == "flatten":
        flatten_test_cases(
            args.test_cases,
            args.behaviors,
            args.attack_method,
            args.output,
        )
        print(f"Wrote flattened attack cases -> {args.output}")


if __name__ == "__main__":
    main()
