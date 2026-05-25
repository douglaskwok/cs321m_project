"""Combine per-model safety result JSONs into final_solver files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SAFETY_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SAFETY_DIR / "results"
DEFAULT_OUTPUT_DIR = SAFETY_DIR / "final_solver"

RESULT_PATTERNS = {
    "autodan": "safety_solver_autodan_vicuna_7b_v1_5_sampled_*.json",
    "dan": "safety_solver_dan_*.json",
    "gcg": "safety_solver_gcg_vicuna_7b_v1_5_sampled_*.json",
    "pair": "safety_solver_pair_vicuna_7b_v1_5_sampled_*.json",
    "pap": "safety_solver_pap_top5_sampled_*.json",
}

DIRECT_RESULT_FILES = [
    "safety_solver_qwenqwen3508b.json",
    "safety_solver_qwenqwen352b.json",
    "safety_solver_qwenqwen354b.json",
    "safety_solver_qwenqwen359b.json",
    "safety_solver_qwenqwen3527b.json",
    "safety_solver_ministral3b_bf16_2512.json",
    "safety_solver_ministral8b_bf16_2512.json",
    "safety_solver_ministral14b_bf16_2512.json",
    "safety_solver_claudehaiku4520251001.json",
    "safety_solver_claudesonnet46.json",
]


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise TypeError(f"Expected {path} to contain a JSON list")
    return data


def result_files(results_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in results_dir.glob(pattern)
        if path.is_file() and not path.name.endswith("_responses.jsonl")
    )


def normalize_rows(result_file: Path) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    rows = read_json_list(result_file)
    for idx, row in enumerate(rows):
        item = dict(row)
        item.setdefault("score", "")
        item.setdefault("input_index", idx)
        item["result_file"] = result_file.name
        combined.append(item)
    return combined


def combine_pattern(results_dir: Path, name: str, pattern: str) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for result_file in result_files(results_dir, pattern):
        combined.extend(normalize_rows(result_file))
    if not combined:
        raise FileNotFoundError(f"No result files found for {name!r} with pattern {pattern!r}")
    return combined


def combine_direct(results_dir: Path) -> list[dict[str, Any]]:
    missing = [name for name in DIRECT_RESULT_FILES if not (results_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing direct result file(s): {missing}")

    combined: list[dict[str, Any]] = []
    for file_name in DIRECT_RESULT_FILES:
        combined.extend(normalize_rows(results_dir / file_name))
    return combined


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def parse_outputs(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return ["direct", *RESULT_PATTERNS]
    outputs = [part.strip().lower() for part in value.split(",") if part.strip()]
    known_outputs = {"direct", *RESULT_PATTERNS}
    unknown = [output for output in outputs if output not in known_outputs]
    if unknown:
        known = ", ".join(sorted(known_outputs))
        raise ValueError(f"Unknown output(s): {unknown}. Known outputs: {known}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-model safety result JSONs into final_solver outputs."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--outputs",
        default="all",
        help="Comma-separated output names, or 'all'. Known: direct,autodan,dan,gcg,pair,pap.",
    )
    args = parser.parse_args()

    for output in parse_outputs(args.outputs):
        if output == "direct":
            rows = combine_direct(args.results_dir)
        else:
            rows = combine_pattern(args.results_dir, output, RESULT_PATTERNS[output])
        output_path = args.output_dir / f"{output}_all_models.json"
        write_json(output_path, rows)
        print(f"Wrote {len(rows)} row(s) -> {output_path}")


if __name__ == "__main__":
    main()
