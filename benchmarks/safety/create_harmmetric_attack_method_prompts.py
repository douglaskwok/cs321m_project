"""Build solver-format prompt files for HarmMetric_Eval attack methods.

HarmMetric_Eval labels six attack methods in its benchmark data:
PAP, PAIR, GPTFuzzer, DAN, AutoDAN, and GCG.

For HarmBench-generated methods, this script reads real ``test_cases.json`` files
from ``benchmarks/safety/attack_cases`` and samples one prompt per behavior.
It does not invent missing PAP/GCG/etc. outputs. Missing attack-case files are
reported in ``missing_attack_methods.json``.

The DAN method is the exception: HarmBench does not expose a standalone ``DAN``
optimizer method, so this script builds DAN prompts from the DAN template in
HarmBench's human jailbreak template list and labels the source explicitly.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAFETY_DIR = Path(__file__).parent
HARMBENCH_ROOT = PROJECT_ROOT / "HarmBench"

DEFAULT_ATTACK_CASES_DIR = SAFETY_DIR / "attack_cases"
DEFAULT_BEHAVIORS = SAFETY_DIR / "safety_solver_behaviors.csv"
DEFAULT_OUTPUT_DIR = SAFETY_DIR / "attack_method_prompts"

ATTACK_METHODS = {
    "PAP": ["PAP-top5"],
    "PAIR": ["PAIR-vicuna_7b_v1_5"],
    "GPTFuzzer": ["GPTFuzzer-vicuna_7b_v1_5", "GPTFuzz-vicuna_7b_v1_5"],
    "DAN": [],
    "AutoDAN": ["AutoDAN-vicuna_7b_v1_5"],
    "GCG": ["GCG-vicuna_7b_v1_5"],
}


def load_behaviors(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def behavior_text(row: dict[str, str]) -> str:
    behavior = row.get("Behavior", "")
    context = row.get("ContextString", "")
    if context:
        return f"{context}\n\n---\n\n{behavior}"
    return behavior


def load_test_cases(path: Path) -> dict[str, list[str]]:
    if path.is_dir():
        path = path / "test_cases.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a JSON object")

    normalized: dict[str, list[str]] = {}
    for behavior_id, cases in data.items():
        if isinstance(cases, list):
            normalized[behavior_id] = [str(case) for case in cases]
        else:
            normalized[behavior_id] = [str(cases)]
    return normalized


def resolve_attack_cases_path(base_dir: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = base_dir / candidate
        if path.is_file():
            return path
        if (path / "test_cases.json").is_file():
            return path
    return None


def load_dan_template() -> str:
    jailbreaks_path = HARMBENCH_ROOT / "baselines" / "human_jailbreaks" / "jailbreaks.py"
    spec = importlib.util.spec_from_file_location("harmbench_jailbreaks", jailbreaks_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {jailbreaks_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    jailbreaks = list(module.JAILBREAKS)
    for template in jailbreaks:
        lowered = template.lower()
        if "pretend to be dan" in lowered and "do anything now" in lowered:
            return str(template)
    for template in jailbreaks:
        if "dan" in template.lower():
            return str(template)
    raise ValueError("Could not find a DAN template in HarmBench human jailbreaks")


def base_index_from_behavior_id(behavior_id: str) -> int | None:
    try:
        return int(behavior_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def make_row(
    *,
    harmful_prompt: str,
    attack_method: str,
    row: dict[str, str],
    sample_index: int,
    attack_cases_source: str,
) -> dict[str, Any]:
    behavior_id = row["BehaviorID"]
    return {
        "harmful_prompt": harmful_prompt,
        "source": row.get("Tags", ""),
        "response": "",
        "attack_method": attack_method,
        "attack_dimension": attack_method,
        "behavior_id": behavior_id,
        "base_prompt": behavior_text(row),
        "base_source": row.get("Tags", ""),
        "base_index": base_index_from_behavior_id(behavior_id),
        "sample_index": sample_index,
        "score": "",
        "attack_cases_source": attack_cases_source,
    }


def build_rows_from_test_cases(
    *,
    attack_method: str,
    behaviors: list[dict[str, str]],
    test_cases: dict[str, list[str]],
    rng: random.Random,
    attack_cases_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in behaviors:
        behavior_id = row.get("BehaviorID", "")
        cases = test_cases.get(behavior_id, [])
        if not cases:
            continue
        sample_index = rng.randrange(len(cases))
        rows.append(
            make_row(
                harmful_prompt=cases[sample_index],
                attack_method=attack_method,
                row=row,
                sample_index=sample_index,
                attack_cases_source=attack_cases_source,
            )
        )
    return rows


def build_dan_rows(behaviors: list[dict[str, str]]) -> list[dict[str, Any]]:
    dan_template = load_dan_template()
    source = "HarmBench/baselines/human_jailbreaks/jailbreaks.py:DAN_template"
    rows: list[dict[str, Any]] = []
    for row in behaviors:
        prompt = f"{dan_template}\n\n{behavior_text(row)}"
        rows.append(
            make_row(
                harmful_prompt=prompt,
                attack_method="DAN",
                row=row,
                sample_index=0,
                attack_cases_source=source,
            )
        )
    return rows


def write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-cases-dir", type=Path, default=DEFAULT_ATTACK_CASES_DIR)
    parser.add_argument("--behaviors", type=Path, default=DEFAULT_BEHAVIORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    behaviors = load_behaviors(args.behaviors)
    rng = random.Random(args.seed)

    combined: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}

    for attack_method, candidates in ATTACK_METHODS.items():
        if attack_method == "DAN":
            rows = build_dan_rows(behaviors)
            source = "HarmBench DAN human-jailbreak template"
        else:
            path = resolve_attack_cases_path(args.attack_cases_dir, candidates)
            if path is None:
                missing[attack_method] = candidates
                summary[attack_method] = {"rows": 0, "status": "missing"}
                continue
            test_cases = load_test_cases(path)
            source = str(path)
            rows = build_rows_from_test_cases(
                attack_method=attack_method,
                behaviors=behaviors,
                test_cases=test_cases,
                rng=rng,
                attack_cases_source=source,
            )

        write_json(args.output_dir / f"safety_solver_{attack_method.lower()}.json", rows)
        combined.extend(rows)
        summary[attack_method] = {
            "rows": len(rows),
            "status": "written",
            "source": source,
        }

    write_json(args.output_dir / "safety_solver_harmmetric_attack_methods_all.json", combined)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "missing_attack_methods.json", missing)

    print(f"Wrote {len(combined)} combined prompt(s) -> {args.output_dir}")
    for attack_method in ATTACK_METHODS:
        info = summary.get(attack_method, {"rows": 0, "status": "missing"})
        print(f"{attack_method}: {info['status']} ({info['rows']} row(s))")
    if missing:
        print(f"Missing generated attack cases for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
