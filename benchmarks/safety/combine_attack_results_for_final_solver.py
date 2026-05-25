"""Combine per-model safety attack result JSONs into final_solver files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SAFETY_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = SAFETY_DIR / "results"
DEFAULT_OUTPUT_DIR = SAFETY_DIR / "final_solver"

ATTACK_PATTERNS = {
    "autodan": "safety_solver_autodan_vicuna_7b_v1_5_sampled_*.json",
    "gcg": "safety_solver_gcg_vicuna_7b_v1_5_sampled_*.json",
    "pair": "safety_solver_pair_vicuna_7b_v1_5_sampled_*.json",
    "pap": "safety_solver_pap_top5_sampled_*.json",
}


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


def combine_attack(results_dir: Path, attack: str, pattern: str) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for result_file in result_files(results_dir, pattern):
        rows = read_json_list(result_file)
        for idx, row in enumerate(rows):
            item = dict(row)
            item.setdefault("score", "")
            item.setdefault("input_index", idx)
            item["result_file"] = result_file.name
            combined.append(item)
    if not combined:
        raise FileNotFoundError(f"No result files found for {attack!r} with pattern {pattern!r}")
    return combined


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def parse_attacks(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ATTACK_PATTERNS)
    attacks = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [attack for attack in attacks if attack not in ATTACK_PATTERNS]
    if unknown:
        known = ", ".join(ATTACK_PATTERNS)
        raise ValueError(f"Unknown attack(s): {unknown}. Known attacks: {known}")
    return attacks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-model safety attack result JSONs into final_solver outputs."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--attacks",
        default="all",
        help="Comma-separated attack names, or 'all'. Known: autodan,gcg,pair,pap.",
    )
    args = parser.parse_args()

    for attack in parse_attacks(args.attacks):
        rows = combine_attack(args.results_dir, attack, ATTACK_PATTERNS[attack])
        output_path = args.output_dir / f"{attack}_all_models.json"
        write_json(output_path, rows)
        print(f"Wrote {len(rows)} row(s) -> {output_path}")


if __name__ == "__main__":
    main()
