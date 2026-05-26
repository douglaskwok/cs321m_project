"""Create a response matrix from KUDGE challenge outputs.

Rows are solver models, derived from ``kudge_*_responses.jsonl`` file names in
``results/kudge_challenge_easy_hard``. Columns are KUDGE item ids. Values are
the binary ``correct`` field.

The converter intentionally ignores raw model response text and keeps only
small item metadata needed for analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path(__file__).parent / "results" / "kudge_challenge_easy_hard"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "response_matrices"
FILE_PREFIX = "kudge_"
FILE_SUFFIX = "_responses.jsonl"


def subject_id_from_file(path: Path) -> str:
    stem = path.name.removeprefix(FILE_PREFIX).removesuffix(FILE_SUFFIX)
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def item_sort_key(item_id: str) -> tuple[int, str]:
    try:
        return int(item_id), item_id
    except ValueError:
        return 10**9, item_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def build_matrix(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: dict[str, dict[str, float]] = {}
    subject_metadata_rows: list[dict[str, Any]] = []
    item_metadata_by_id: dict[str, dict[str, Any]] = {}

    for path in paths:
        subject_id = subject_id_from_file(path)
        rows = read_jsonl(path)
        values: dict[str, float] = {}
        subsets: set[str] = set()

        for row in rows:
            if "id" not in row:
                raise ValueError(f"{path} has a row without id")
            item_id = str(row["id"])
            if item_id in values:
                raise ValueError(f"{path} has duplicate id={item_id!r}")

            correct = row.get("correct")
            values[item_id] = float(correct) if correct is not None else float("nan")
            subsets.add(str(row.get("subset", "")))

            item_metadata_by_id.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "subset": row.get("subset", ""),
                    "gold": row.get("gold", ""),
                },
            )

        model_rows[subject_id] = values
        subject_metadata_rows.append(
            {
                "subject_id": subject_id,
                "source_file": path.name,
                "n_items": len(values),
                "accuracy": pd.Series(values).mean(skipna=True),
                "subsets": "|".join(sorted(subsets)),
            }
        )

    item_ids = sorted(item_metadata_by_id, key=item_sort_key)
    subject_ids = sorted(model_rows)
    matrix = pd.DataFrame.from_dict(model_rows, orient="index").reindex(
        index=subject_ids,
        columns=item_ids,
    )
    matrix.index.name = "subject_id"

    subject_metadata = pd.DataFrame(subject_metadata_rows).sort_values("subject_id")
    item_metadata = pd.DataFrame([item_metadata_by_id[item_id] for item_id in item_ids])
    return matrix, subject_metadata, item_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create KUDGE challenge response matrix.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(f"{FILE_PREFIX}*{FILE_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"No {FILE_PREFIX}*{FILE_SUFFIX} files found in {args.input_dir}")

    matrix, subject_metadata, item_metadata = build_matrix(paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "kudge_challenge_easy_hard_response_matrix.csv"
    subject_path = args.output_dir / "kudge_challenge_easy_hard_subject_metadata.csv"
    item_path = args.output_dir / "kudge_challenge_easy_hard_item_metadata.csv"

    matrix.to_csv(matrix_path)
    subject_metadata.to_csv(subject_path, index=False)
    item_metadata.to_csv(item_path, index=False)

    observed = int(matrix.notna().sum().sum())
    total = int(matrix.shape[0] * matrix.shape[1])
    print(f"Wrote matrix {matrix.shape[0]} subjects x {matrix.shape[1]} items -> {matrix_path}")
    print(f"Observed cells: {observed}/{total} ({observed / total:.1%})")
    print(f"Wrote subject metadata -> {subject_path}")
    print(f"Wrote item metadata -> {item_path}")


if __name__ == "__main__":
    main()
