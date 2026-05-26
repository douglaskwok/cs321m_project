"""Create a ResponseMatrix-ready CSV from MMLU-Pro solver JSONL outputs.

Rows are solver models. Columns are MMLU-Pro ``original_id`` item ids.
Values are:

    1.0 = correct
    0.0 = incorrect
    NaN = unparsed / missing correctness

The CSV can be loaded in a notebook with:

    df = pd.read_csv("benchmarks/mmlu/response_matrices/mmlu_pro_solver_response_matrix.csv", index_col=0)
    rm = ResponseMatrix.from_dataframe(df)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


MMLU_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = MMLU_DIR / "response_matrices"
DEFAULT_INPUT_DIRS = [
    MMLU_DIR / "qwen35_solving_outputs",
    MMLU_DIR / "anthropic_solving_outputs",
]


def model_id_from_filename(path: Path) -> str:
    name = path.name
    if name.startswith("qwen35_08b_"):
        return "Qwen/Qwen3.5-0.8B"
    return path.stem.split("_solving_", 1)[0]


def model_slug(model_id: str) -> str:
    slug = model_id.lower().replace("/", "_")
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def discover_solver_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_dir in input_dirs:
        files.extend(sorted(input_dir.resolve().glob("*_solving_gpt_mmlu_pro*.jsonl")))
    return [
        path
        for path in files
        if "limit" not in path.name and "256tok" not in path.name
    ]


def correctness_value(value: Any) -> float:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return float("nan")


def build_matrix(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: dict[str, dict[str, float]] = {}
    model_metadata: list[dict[str, str]] = []
    item_metadata_by_id: dict[str, dict[str, Any]] = {}

    for path in files:
        rows = read_jsonl(path)
        if not rows:
            continue

        first_model_id = rows[0].get("model_id") or model_id_from_filename(path)
        subject_id = model_slug(str(first_model_id))
        if subject_id in model_rows:
            raise ValueError(f"Duplicate subject id {subject_id!r} from {path}")

        values: dict[str, float] = {}
        for row in rows:
            item_id = str(row["original_id"])
            values[item_id] = correctness_value(row.get("correct"))
            item_metadata_by_id.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "pair_id": row.get("pair_id", ""),
                    "source": row.get("source", ""),
                    "split": row.get("split", ""),
                    "gold_letter": row.get("gold_letter", ""),
                },
            )

        model_rows[subject_id] = values
        model_metadata.append(
            {
                "subject_id": subject_id,
                "model_id": str(first_model_id),
                "result_file": str(path.resolve().relative_to(MMLU_DIR)),
                "n_rows": str(len(rows)),
                "n_scored": str(sum(row.get("correct") is not None for row in rows)),
            }
        )

    item_ids = sorted(item_metadata_by_id, key=lambda value: int(value) if value.isdigit() else value)
    matrix = pd.DataFrame.from_dict(model_rows, orient="index").reindex(columns=item_ids)
    matrix.index.name = "subject_id"

    item_metadata = pd.DataFrame(
        [item_metadata_by_id[item_id] for item_id in item_ids]
    )
    subject_metadata = pd.DataFrame(model_metadata)
    return matrix, item_metadata, subject_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an MMLU-Pro solver response matrix from JSONL outputs."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUT_DIRS,
        help="Directories containing *_solving_gpt_mmlu_pro.jsonl files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    files = discover_solver_files(args.input_dirs)
    if not files:
        raise FileNotFoundError("No full *_solving_gpt_mmlu_pro.jsonl files found")

    matrix, item_metadata, subject_metadata = build_matrix(files)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = args.output_dir / "mmlu_pro_solver_response_matrix.csv"
    item_path = args.output_dir / "mmlu_pro_solver_item_metadata.csv"
    subject_path = args.output_dir / "mmlu_pro_solver_subject_metadata.csv"

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
