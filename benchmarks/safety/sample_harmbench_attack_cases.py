"""Sample one HarmBench attack case per behavior into solver JSON format."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_ATTACK_CASES = Path(__file__).parent / "attack_cases" / "HumanJailbreaks"
DEFAULT_BEHAVIORS = Path(__file__).parent / "safety_solver_behaviors.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "safety_solver_humanjailbreaks_sampled.json"


def load_behavior_sources(behaviors_path: Path) -> dict[str, str]:
    with behaviors_path.open("r", encoding="utf-8", newline="") as fh:
        return {row["BehaviorID"]: row.get("Tags", "") for row in csv.DictReader(fh)}


def load_test_cases(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a JSON object")

    normalized = {}
    for behavior_id, cases in data.items():
        if isinstance(cases, list):
            normalized[behavior_id] = [str(case) for case in cases]
        else:
            normalized[behavior_id] = [str(cases)]
    return normalized


def sample_cases(
    test_cases_path: Path,
    behaviors_path: Path,
    output_path: Path,
    seed: int,
) -> None:
    sources = load_behavior_sources(behaviors_path)
    test_cases = load_test_cases(test_cases_path)
    rng = random.Random(seed)

    rows = []
    for behavior_id in sorted(test_cases):
        cases = test_cases[behavior_id]
        if not cases:
            continue
        sample_index = rng.randrange(len(cases))
        rows.append(
            {
                "harmful_prompt": cases[sample_index],
                "source": sources.get(behavior_id, ""),
                "response": "",
                "sample_index": sample_index,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} sampled attack prompt(s) -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases", type=Path, default=DEFAULT_ATTACK_CASES)
    parser.add_argument("--behaviors", type=Path, default=DEFAULT_BEHAVIORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    sample_cases(args.test_cases, args.behaviors, args.output, args.seed)


if __name__ == "__main__":
    main()
