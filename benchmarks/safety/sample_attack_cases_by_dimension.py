"""Sample attack-case prompts into solver JSON files.

This is the same lightweight solver shape as
``safety_solver_humanjailbreaks_sampled.json``:

    harmful_prompt, source, response, sample_index

The script samples one generated attack prompt per behavior for every attack-case
directory under ``benchmarks/safety/attack_cases``. The behavior ``Tags`` column
is kept in the ``source`` field, but benchmark tags are combined into one JSON
per attack method by default. Pass ``--split-by-dimension`` to write one JSON per
tag instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ATTACK_CASES_ROOT = Path(__file__).parent / "attack_cases"
DEFAULT_BEHAVIORS = Path(__file__).parent / "safety_solver_behaviors.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "attack_cases_sampled_by_dimension"
DEFAULT_EXCLUDE_DIMENSIONS = ""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def load_behavior_dimensions(
    behaviors_path: Path,
    dimension_field: str,
) -> dict[str, str]:
    with behaviors_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "BehaviorID" not in reader.fieldnames:
            raise ValueError(f"{behaviors_path} must include a BehaviorID column")
        if dimension_field not in reader.fieldnames:
            fields = ", ".join(reader.fieldnames)
            raise ValueError(
                f"{behaviors_path} has no {dimension_field!r} column; "
                f"available columns: {fields}"
            )
        return {
            row["BehaviorID"]: row.get(dimension_field, "").strip() or "unknown"
            for row in reader
        }


def parse_dimension_list(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def flatten_prompt_candidates(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        prompts: list[str] = []
        for item in value:
            prompts.extend(flatten_prompt_candidates(item))
        return prompts
    if value is None:
        return []
    return [str(value)]


def load_test_cases(test_cases_path: Path) -> dict[str, list[str]]:
    with test_cases_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Expected {test_cases_path} to contain a JSON object")

    return {
        behavior_id: flatten_prompt_candidates(cases)
        for behavior_id, cases in data.items()
    }


def discover_attack_case_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if (root / "test_cases.json").is_file():
        return [root / "test_cases.json"]
    return sorted(path / "test_cases.json" for path in root.iterdir() if (path / "test_cases.json").is_file())


def sample_attack_cases(
    test_cases_path: Path,
    behavior_dimensions: dict[str, str],
    output_dir: Path,
    seed: int,
    include_dimensions: set[str],
    exclude_dimensions: set[str],
    split_by_dimension: bool,
) -> dict[str, int]:
    rng = random.Random(seed)
    attack_name = test_cases_path.parent.name
    test_cases = load_test_cases(test_cases_path)
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for behavior_id in sorted(test_cases):
        candidates = test_cases[behavior_id]
        if not candidates:
            continue

        sample_index = rng.randrange(len(candidates))
        dimension = behavior_dimensions.get(behavior_id, "unknown")
        normalized_dimension = dimension.lower()
        if include_dimensions and normalized_dimension not in include_dimensions:
            continue
        if normalized_dimension in exclude_dimensions:
            continue

        group = dimension if split_by_dimension else "combined"
        rows_by_group[group].append(
            {
                "harmful_prompt": candidates[sample_index],
                "source": dimension,
                "response": "",
                "sample_index": sample_index,
            }
        )

    attack_output_dir = output_dir / slugify(attack_name)
    attack_output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for group, rows in sorted(rows_by_group.items()):
        if split_by_dimension:
            output_name = f"safety_solver_{slugify(attack_name)}_{slugify(group)}_sampled.json"
        else:
            output_name = f"safety_solver_{slugify(attack_name)}_sampled.json"
        output_path = attack_output_dir / output_name
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        counts[str(output_path)] = len(rows)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample one attack prompt per behavior into solver JSON files."
    )
    parser.add_argument(
        "--attack-cases-root",
        type=Path,
        default=DEFAULT_ATTACK_CASES_ROOT,
        help="Attack-case root, attack-case directory, or test_cases.json file.",
    )
    parser.add_argument("--behaviors", type=Path, default=DEFAULT_BEHAVIORS)
    parser.add_argument(
        "--dimension-field",
        default="Tags",
        help="Column in the behaviors CSV to use for per-dimension output files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--split-by-dimension",
        action="store_true",
        help="Write one output JSON per dimension instead of one combined JSON per attack method.",
    )
    parser.add_argument(
        "--include-dimensions",
        default="",
        help="Comma-separated dimensions to include. Empty means include all except exclusions.",
    )
    parser.add_argument(
        "--exclude-dimensions",
        default=DEFAULT_EXCLUDE_DIMENSIONS,
        help="Comma-separated dimensions to skip. Defaults to JailbreakBench.",
    )
    args = parser.parse_args()

    behavior_dimensions = load_behavior_dimensions(args.behaviors, args.dimension_field)
    include_dimensions = parse_dimension_list(args.include_dimensions)
    exclude_dimensions = parse_dimension_list(args.exclude_dimensions)
    test_case_files = discover_attack_case_files(args.attack_cases_root)
    if not test_case_files:
        raise FileNotFoundError(f"No test_cases.json files found under {args.attack_cases_root}")

    total_files = 0
    total_rows = 0
    for test_cases_path in test_case_files:
        counts = sample_attack_cases(
            test_cases_path=test_cases_path,
            behavior_dimensions=behavior_dimensions,
            output_dir=args.output_dir,
            seed=args.seed,
            include_dimensions=include_dimensions,
            exclude_dimensions=exclude_dimensions,
            split_by_dimension=args.split_by_dimension,
        )
        total_files += len(counts)
        total_rows += sum(counts.values())
        for output_path, row_count in counts.items():
            print(f"Wrote {row_count} sampled prompt(s) -> {output_path}")

    print(f"Wrote {total_files} file(s), {total_rows} sampled prompt(s) total.")


if __name__ == "__main__":
    main()
