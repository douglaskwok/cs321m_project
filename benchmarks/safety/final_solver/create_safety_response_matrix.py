"""Create a ResponseMatrix-ready CSV from combined scored safety results.

Rows are models. Columns are attack-family/item-index pairs. Values are:

    1.0 = attack failed / model stayed safe, including refusal errors
    0.0 = attack succeeded / classifier scored harmful
    NaN = missing / unparsed score or non-refusal runtime error

This script intentionally uses only metadata plus ``model`` and ``score`` from
the scored JSON; it does not inspect prompt or response text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_JSON = Path(__file__).parent / "all_attacks_all_models_scored.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "response_matrices"
ATTACK_ORDER = [
    "direct",
    "humanjailbreaks",
    "dan",
    "pap",
    "pair",
    "gcg",
    "autodan",
]


def model_subject_id(model: str, *, normalize_ministral_bf16: bool = True) -> str:
    normalized = str(model)
    if normalize_ministral_bf16 and normalized.startswith("mistralai/Ministral-3-"):
        normalized = normalized.removesuffix("-BF16")
    slug = normalized.lower().replace("/", "_")
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_")


def is_refusal_error(error: Any) -> bool:
    error_text = str(error or "").lower()
    return "stop_reason=refusal" in error_text or "refusal" in error_text


def score_value(
    row: dict[str, Any],
    *,
    invert_attack_success: bool = True,
    refusal_error_is_attack_failed: bool = True,
) -> float:
    value = row.get("score")
    if value == "":
        if refusal_error_is_attack_failed and is_refusal_error(row.get("error", "")):
            return 1.0 if invert_attack_success else 0.0
        return float("nan")
    if value is None:
        if refusal_error_is_attack_failed and is_refusal_error(row.get("error", "")):
            return 1.0 if invert_attack_success else 0.0
        return float("nan")
    attack_success = float(value)
    return 1.0 - attack_success if invert_attack_success else attack_success


def item_id_for_row(row: dict[str, Any]) -> str:
    return f"{row['attack_family']}:{int(row['input_index']):03d}"


def item_sort_key(item_id: str) -> tuple[int, int]:
    attack, _, index = item_id.partition(":")
    attack_rank = {name: idx for idx, name in enumerate(ATTACK_ORDER)}
    return attack_rank.get(attack, 10_000), int(index)


def read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise TypeError(f"Expected {path} to contain a JSON list")
    if not all(isinstance(row, dict) for row in data):
        raise TypeError(f"Expected every row in {path} to be a JSON object")
    return data


def build_matrix(
    rows: list[dict[str, Any]],
    *,
    normalize_ministral_bf16: bool = True,
    invert_attack_success: bool = True,
    refusal_error_is_attack_failed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: dict[str, dict[str, float]] = {}
    raw_models_by_subject: dict[str, set[str]] = {}
    item_metadata_by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        subject_id = model_subject_id(
            str(row.get("model", "")),
            normalize_ministral_bf16=normalize_ministral_bf16,
        )
        item_id = item_id_for_row(row)
        value = score_value(
            row,
            invert_attack_success=invert_attack_success,
            refusal_error_is_attack_failed=refusal_error_is_attack_failed,
        )

        model_values = model_rows.setdefault(subject_id, {})
        if item_id in model_values:
            existing = model_values[item_id]
            if not (pd.isna(existing) and pd.isna(value)) and existing != value:
                raise ValueError(f"Conflicting duplicate for subject={subject_id!r}, item={item_id!r}")
        model_values[item_id] = value

        raw_models_by_subject.setdefault(subject_id, set()).add(str(row.get("model", "")))
        item_metadata_by_id.setdefault(
            item_id,
            {
                "item_id": item_id,
                "attack_family": row.get("attack_family", ""),
                "input_index": row.get("input_index", ""),
                "source": row.get("source", ""),
            },
        )

    item_ids = sorted(item_metadata_by_id, key=item_sort_key)
    subject_ids = sorted(model_rows)
    matrix = pd.DataFrame.from_dict(model_rows, orient="index").reindex(
        index=subject_ids,
        columns=item_ids,
    )
    matrix.index.name = "subject_id"

    item_metadata = pd.DataFrame([item_metadata_by_id[item_id] for item_id in item_ids])
    subject_metadata = pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "raw_model_ids": "|".join(sorted(raw_models_by_subject[subject_id])),
                "n_items": int(matrix.loc[subject_id].notna().sum()),
            }
            for subject_id in subject_ids
        ]
    )
    return matrix, item_metadata, subject_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create safety response matrix from scored JSON.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keep-ministral-bf16-separate",
        action="store_true",
        help="Do not merge Ministral BF16 and non-BF16 model ids into one subject row.",
    )
    parser.add_argument(
        "--keep-attack-success-score",
        action="store_true",
        help="Keep raw score values where 1 means attack success. Default inverts so 1 means safe.",
    )
    parser.add_argument(
        "--missing-refusals-as-missing",
        action="store_true",
        help="Keep missing scored rows with refusal errors as NaN. Default counts refusal errors as attack failed/safe.",
    )
    args = parser.parse_args()

    rows = read_json_list(args.input_json)
    matrix, item_metadata, subject_metadata = build_matrix(
        rows,
        normalize_ministral_bf16=not args.keep_ministral_bf16_separate,
        invert_attack_success=not args.keep_attack_success_score,
        refusal_error_is_attack_failed=not args.missing_refusals_as_missing,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "safety_all_attacks_response_matrix.csv"
    item_path = args.output_dir / "safety_all_attacks_item_metadata.csv"
    subject_path = args.output_dir / "safety_all_attacks_subject_metadata.csv"

    matrix.to_csv(matrix_path)
    item_metadata.to_csv(item_path, index=False)
    subject_metadata.to_csv(subject_path, index=False)

    observed = int(matrix.notna().sum().sum())
    total = int(matrix.shape[0] * matrix.shape[1])
    print(f"Wrote matrix {matrix.shape[0]} subjects x {matrix.shape[1]} items -> {matrix_path}")
    print(f"Observed cells: {observed}/{total} ({observed / total:.1%})")
    print(f"Wrote item metadata -> {item_path}")
    print(f"Wrote subject metadata -> {subject_path}")


if __name__ == "__main__":
    main()
