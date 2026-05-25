"""Merge HarmBench per-behavior attack outputs without importing HarmBench classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a JSON object")
    return data


def merge_parts(save_dir: Path, filename: str) -> dict[str, Any]:
    parts_dir = save_dir / "test_cases_individual_behaviors"
    if not parts_dir.is_dir():
        raise FileNotFoundError(f"Missing {parts_dir}")

    merged: dict[str, Any] = {}
    for behavior_dir in sorted(parts_dir.iterdir()):
        if not behavior_dir.is_dir():
            continue
        path = behavior_dir / filename
        if not path.is_file():
            continue
        part = load_json(path)
        duplicate_ids = set(merged).intersection(part)
        if duplicate_ids:
            duplicate = sorted(duplicate_ids)[0]
            raise ValueError(f"Duplicate behavior ID {duplicate} in {path}")
        merged.update(part)
    return merged


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=Path)
    args = parser.parse_args()

    test_cases = merge_parts(args.save_dir, "test_cases.json")
    if not test_cases:
        raise ValueError(f"No test cases found under {args.save_dir}")
    write_json(args.save_dir / "test_cases.json", test_cases)
    print(f"Wrote {len(test_cases)} behavior(s) -> {args.save_dir / 'test_cases.json'}")

    logs = merge_parts(args.save_dir, "logs.json")
    if logs:
        write_json(args.save_dir / "logs.json", logs)
        print(f"Wrote {len(logs)} behavior log(s) -> {args.save_dir / 'logs.json'}")


if __name__ == "__main__":
    main()
