"""Combine final_solver *_all_models_scored.json files into one JSON list.

This script preserves each row and adds:

    attack_family: derived from the filename, e.g. "autodan"
    scored_source_file: the source scored JSON filename
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_JSON = DEFAULT_INPUT_DIR / "all_attacks_all_models_scored.json"
DEFAULT_SUMMARY_CSV = DEFAULT_INPUT_DIR / "all_attacks_all_models_scored_summary.csv"
ATTACK_ORDER = [
    "direct",
    "humanjailbreaks",
    "dan",
    "pap",
    "pair",
    "gcg",
    "autodan",
]


def attack_family_from_path(path: Path) -> str:
    return path.name.split("_all_models_scored.json", 1)[0]


def discover_scored_files(input_dir: Path) -> list[Path]:
    files = [
        path
        for path in input_dir.glob("*_all_models_scored.json")
        if path.name != DEFAULT_OUTPUT_JSON.name
    ]
    attack_rank = {attack: idx for idx, attack in enumerate(ATTACK_ORDER)}
    return sorted(
        files,
        key=lambda path: (attack_rank.get(attack_family_from_path(path), 10_000), path.name),
    )


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise TypeError(f"Expected {path} to contain a JSON list, got {type(data).__name__}")
    if not all(isinstance(row, dict) for row in data):
        raise TypeError(f"Expected every row in {path} to be a JSON object")
    return data


def combine(files: list[Path]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    combined: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for path in files:
        rows = read_json_list(path)
        attack_family = attack_family_from_path(path)
        for row in rows:
            combined.append(
                {
                    "attack_family": attack_family,
                    "scored_source_file": path.name,
                    **row,
                }
            )

        summary_rows.append(
            {
                "attack_family": attack_family,
                "scored_source_file": path.name,
                "n_rows": len(rows),
                "n_scored": sum(row.get("score") is not None for row in rows),
                "n_errors": sum(bool(str(row.get("error", "")).strip()) for row in rows),
            }
        )

    return combined, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine final_solver scored JSON files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    args = parser.parse_args()

    files = discover_scored_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No *_all_models_scored.json files found in {args.input_dir}")

    combined, summary = combine(files)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    summary.to_csv(args.summary_csv, index=False)

    print(f"Combined {len(files)} files into {len(combined)} rows -> {args.output_json}")
    print(f"Wrote summary -> {args.summary_csv}")


if __name__ == "__main__":
    main()
