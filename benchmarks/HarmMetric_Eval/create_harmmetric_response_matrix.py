"""Create a response matrix from final HarmMetric eval prompt-score CSVs.

Rows are HarmJudge models, derived from file names. Columns are the regular
117 prompt ids. Values are ``overall_effectiveness_score``.

This script intentionally reads only score/metadata columns needed for the
matrix and does not copy prompt or response text into metadata.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


HARMMETRIC_DIR = Path(__file__).parent
DEFAULT_INPUT_DIR = HARMMETRIC_DIR / "Final files to use"
DEFAULT_OUTPUT_DIR = HARMMETRIC_DIR / "response_matrices"
PROMPT_SCORE_PATTERN = "*-HarmJudge-safety_solver_prompt_scores*.csv"
REQUIRED_COLUMNS = {"prompt_id", "overall_effectiveness_score", "source"}


def subject_id_from_file(path: Path) -> str:
    model_name = model_name_from_file(path)
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def model_name_from_file(path: Path) -> str:
    return re.sub(
        r"-HarmJudge-safety_solver_prompt_scores(?:_with_missing)?\.csv$",
        "",
        path.name,
    )


def prompt_sort_key(prompt_id: Any) -> tuple[int, str]:
    text = str(prompt_id)
    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def find_prompt_score_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob(PROMPT_SCORE_PATTERN))


def read_score_file(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = REQUIRED_COLUMNS - set(header.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return pd.read_csv(path, usecols=sorted(REQUIRED_COLUMNS))


def regular_prompt_ids_by_intersection(frames_by_path: dict[Path, pd.DataFrame]) -> list[str]:
    prompt_sets = [
        set(df["prompt_id"].astype(str))
        for df in frames_by_path.values()
    ]
    if not prompt_sets:
        return []
    prompt_ids = set.intersection(*prompt_sets)
    return sorted(prompt_ids, key=prompt_sort_key)


def build_matrix(
    frames_by_path: dict[Path, pd.DataFrame],
    *,
    regular_prompt_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regular_id_set = set(regular_prompt_ids)
    rows: dict[str, dict[str, float]] = {}
    subject_rows: list[dict[str, Any]] = []
    item_metadata_by_id: dict[str, dict[str, Any]] = {}

    for path, df in frames_by_path.items():
        model_name = model_name_from_file(path)
        subject_id = subject_id_from_file(path)

        working = df.copy()
        working["prompt_id"] = working["prompt_id"].astype(str)
        working = working[working["prompt_id"].isin(regular_id_set)]

        if working["prompt_id"].duplicated().any():
            duplicated = sorted(working.loc[working["prompt_id"].duplicated(), "prompt_id"].unique())
            raise ValueError(f"{path} has duplicate prompt_id values: {duplicated[:5]}")

        rows[subject_id] = (
            working.set_index("prompt_id")["overall_effectiveness_score"]
            .astype(float)
            .reindex(regular_prompt_ids)
            .to_dict()
        )

        subject_rows.append(
            {
                "subject_id": subject_id,
                "model_name": model_name,
                "source_file": path.name,
                "n_prompts_observed": int(working["prompt_id"].isin(regular_id_set).sum()),
            }
        )

        for _, row in working.iterrows():
            prompt_id = str(row["prompt_id"])
            item_metadata_by_id.setdefault(
                prompt_id,
                {
                    "item_id": prompt_id,
                    "prompt_id": prompt_id,
                    "source": row.get("source", ""),
                },
            )

    subject_ids = sorted(rows)
    matrix = pd.DataFrame.from_dict(rows, orient="index").reindex(
        index=subject_ids,
        columns=regular_prompt_ids,
    )
    matrix.index.name = "subject_id"

    subject_metadata = pd.DataFrame(subject_rows).sort_values("subject_id")
    item_metadata = pd.DataFrame(
        [
            item_metadata_by_id.get(
                prompt_id,
                {"item_id": prompt_id, "prompt_id": prompt_id, "source": ""},
            )
            for prompt_id in regular_prompt_ids
        ]
    )
    return matrix, subject_metadata, item_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a HarmMetric eval response matrix from final prompt-score CSVs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    files = find_prompt_score_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No {PROMPT_SCORE_PATTERN} files found in {args.input_dir}")

    frames_by_path = {path: read_score_file(path) for path in files}
    regular_prompt_ids = regular_prompt_ids_by_intersection(frames_by_path)
    if len(regular_prompt_ids) != 117:
        print(f"Warning: expected 117 shared regular prompts, found {len(regular_prompt_ids)}")

    matrix, subject_metadata, item_metadata = build_matrix(
        frames_by_path,
        regular_prompt_ids=regular_prompt_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "harmmetric_eval_response_matrix.csv"
    subject_path = args.output_dir / "harmmetric_eval_subject_metadata.csv"
    item_path = args.output_dir / "harmmetric_eval_item_metadata.csv"

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
