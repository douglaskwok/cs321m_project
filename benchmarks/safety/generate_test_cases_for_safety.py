"""Generate lightweight HarmBench-style test cases for SafetySuite prompts.

This is a focused replacement for HarmBench/generate_test_cases.py for the
methods that do not require loading attack/target models.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARMBENCH_ROOT = PROJECT_ROOT / "HarmBench"
DEFAULT_BEHAVIORS = Path(__file__).parent / "safety_solver_behaviors.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "attack_cases"


def load_behaviors(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def behavior_text(row: dict[str, str]) -> str:
    behavior = row.get("Behavior", "")
    context = row.get("ContextString", "")
    if context:
        return f"{context}\n\n---\n\n{behavior}"
    return behavior


def load_human_jailbreaks() -> list[str]:
    jailbreaks_path = HARMBENCH_ROOT / "baselines" / "human_jailbreaks" / "jailbreaks.py"
    if not jailbreaks_path.exists():
        raise FileNotFoundError(
            f"Could not find {jailbreaks_path}. Clone HarmBench at {HARMBENCH_ROOT}."
        )

    spec = importlib.util.spec_from_file_location("harmbench_jailbreaks", jailbreaks_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {jailbreaks_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.JAILBREAKS)


def generate_direct(behaviors: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        row["BehaviorID"]: [behavior_text(row)]
        for row in behaviors
        if row.get("BehaviorID") and row.get("Behavior")
    }


def generate_human(
    behaviors: list[dict[str, str]],
    random_subset: int,
    seed: int,
) -> dict[str, list[str]]:
    jailbreaks = load_human_jailbreaks()
    rng = random.Random(seed)
    test_cases: dict[str, list[str]] = {}

    for row in behaviors:
        behavior_id = row.get("BehaviorID")
        if not behavior_id or not row.get("Behavior"):
            continue

        shuffled = list(jailbreaks)
        rng.shuffle(shuffled)
        selected = shuffled if random_subset < 0 else shuffled[:random_subset]
        prompt = behavior_text(row)
        test_cases[behavior_id] = [f"{jailbreak}\n\n{prompt}" for jailbreak in selected]

    return test_cases


def write_outputs(
    test_cases: dict[str, list[str]],
    output_dir: Path,
    method_config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_cases.json").open("w", encoding="utf-8") as fh:
        json.dump(test_cases, fh, ensure_ascii=False, indent=2)
    with (output_dir / "logs.json").open("w", encoding="utf-8") as fh:
        json.dump({}, fh, ensure_ascii=False, indent=2)
    with (output_dir / "method_config.json").open("w", encoding="utf-8") as fh:
        json.dump(method_config, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["direct", "human"],
        required=True,
        help="Lightweight test-case generator to run.",
    )
    parser.add_argument("--behaviors", type=Path, default=DEFAULT_BEHAVIORS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-subset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_name = "DirectRequest" if args.method == "direct" else "HumanJailbreaks"
        output_dir = DEFAULT_OUTPUT_DIR / output_name

    behaviors = load_behaviors(args.behaviors)
    if args.method == "direct":
        test_cases = generate_direct(behaviors)
        method_config = {"method": "DirectRequest"}
    else:
        test_cases = generate_human(behaviors, args.random_subset, args.seed)
        method_config = {
            "method": "HumanJailbreaks",
            "random_subset": args.random_subset,
            "seed": args.seed,
        }

    write_outputs(test_cases, output_dir, method_config)
    print(f"Wrote {len(test_cases)} behavior(s) -> {output_dir / 'test_cases.json'}")


if __name__ == "__main__":
    main()
