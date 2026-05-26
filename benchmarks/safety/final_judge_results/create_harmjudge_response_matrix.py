"""Create a response matrix from HarmJudge safety-solver prompt scores.

Rows are HarmJudge models, derived from the file names. Columns are the 117
regular prompt ids. Values are ``overall_effectiveness_score``.

This script intentionally uses only score and metadata columns; it does not
copy prompt text into the response-matrix metadata.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "response_matrices"
PROMPT_SCORE_SUFFIX = "-HarmJudge-safety_solver_prompt_scores.csv"
REQUIRED_COLUMNS = {"prompt_id", "overall_effectiveness_score", "source"}


def subject_id_from_file(path: Path) -> str:
    model_name = path.name.removesuffix(PROMPT_SCORE_SUFFIX)
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def model_name_from_file(path: Path) -> str:
    return path.name.removesuffix(PROMPT_SCORE_SUFFIX)


def prompt_sort_key(prompt_id: Any) -> tuple[int, str]:
    text = str(prompt_id)
    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def find_prompt_score_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob(f"*{PROMPT_SCORE_SUFFIX}"))


def read_score_file(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = REQUIRED_COLUMNS - set(header.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return pd.read_csv(path, usecols=sorted(REQUIRED_COLUMNS))


def regular_prompt_ids_by_union(frames_by_path: dict[Path, pd.DataFrame]) -> list[str]:
    """Use the union from full 117-prompt files as the regular prompt universe."""
    prompt_ids: set[str] = set()
    full_files = 0

    for path, df in frames_by_path.items():
        model_name = model_name_from_file(path).lower()
        if "qwen3.5-0.8b" in model_name:
            continue

        unique_ids = set(df["prompt_id"].astype(str))
        if len(unique_ids) == 117:
            full_files += 1
            prompt_ids.update(unique_ids)

    if not prompt_ids:
        for df in frames_by_path.values():
            prompt_ids.update(df["prompt_id"].astype(str))

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
        if "qwen3.5-0.8b" in model_name.lower():
            working = working[working["prompt_id"].isin(regular_id_set)]

        if working["prompt_id"].duplicated().any():
            duplicated = sorted(working.loc[working["prompt_id"].duplicated(), "prompt_id"].unique())
            raise ValueError(f"{path} has duplicate prompt_id values: {duplicated[:5]}")

        values = (
            working.set_index("prompt_id")["overall_effectiveness_score"]
            .astype(float)
            .reindex(regular_prompt_ids)
            .to_dict()
        )
        rows[subject_id] = values

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
            if prompt_id not in regular_id_set:
                continue
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

    item_metadata = pd.DataFrame(
        [
            item_metadata_by_id.get(
                prompt_id,
                {"item_id": prompt_id, "prompt_id": prompt_id, "source": ""},
            )
            for prompt_id in regular_prompt_ids
        ]
    )
    subject_metadata = pd.DataFrame(subject_rows).sort_values("subject_id")
    return matrix, subject_metadata, item_metadata


def write_filtered_prompt_scores(
    *,
    files: list[Path],
    regular_prompt_ids: list[str],
    output_dir: Path,
) -> None:
    regular_id_set = set(regular_prompt_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        df = pd.read_csv(path)
        if "prompt_id" not in df.columns:
            raise ValueError(f"{path} is missing prompt_id")

        working = df.copy()
        working["prompt_id"] = working["prompt_id"].astype(str)
        filtered = working[working["prompt_id"].isin(regular_id_set)].copy()
        filtered["_prompt_sort_key"] = filtered["prompt_id"].map(
            {prompt_id: idx for idx, prompt_id in enumerate(regular_prompt_ids)}
        )
        filtered = (
            filtered.sort_values("_prompt_sort_key")
            .drop(columns=["_prompt_sort_key"])
            .reset_index(drop=True)
        )
        filtered.to_csv(output_dir / path.name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a response matrix from HarmJudge safety-solver prompt scores."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    files = find_prompt_score_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No *{PROMPT_SCORE_SUFFIX} files found in {args.input_dir}")

    frames_by_path = {path: read_score_file(path) for path in files}
    regular_prompt_ids = regular_prompt_ids_by_union(frames_by_path)
    if len(regular_prompt_ids) != 117:
        print(f"Warning: expected 117 regular prompts, found {len(regular_prompt_ids)}")

    qwen08_prompt_file = any("qwen3.5-0.8b" in model_name_from_file(path).lower() for path in files)
    qwen08_aggregate = args.input_dir / "Qwen3.5-0.8B-HarmJudge.csv"
    if not qwen08_prompt_file and qwen08_aggregate.exists():
        print("Warning: Qwen3.5-0.8B has only an aggregate CSV, so it is not included.")

    matrix, subject_metadata, item_metadata = build_matrix(
        frames_by_path,
        regular_prompt_ids=regular_prompt_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "harmjudge_safety_solver_response_matrix.csv"
    subject_path = args.output_dir / "harmjudge_safety_solver_subject_metadata.csv"
    item_path = args.output_dir / "harmjudge_safety_solver_item_metadata.csv"
    filtered_dir = args.output_dir / "filtered_prompt_scores"

    matrix.to_csv(matrix_path)
    subject_metadata.to_csv(subject_path, index=False)
    item_metadata.to_csv(item_path, index=False)
    write_filtered_prompt_scores(
        files=files,
        regular_prompt_ids=regular_prompt_ids,
        output_dir=filtered_dir,
    )

    observed = int(matrix.notna().sum().sum())
    total = int(matrix.shape[0] * matrix.shape[1])
    print(f"Wrote matrix {matrix.shape[0]} subjects x {matrix.shape[1]} items -> {matrix_path}")
    print(f"Observed cells: {observed}/{total} ({observed / total:.1%})")
    print(f"Wrote subject metadata -> {subject_path}")
    print(f"Wrote item metadata -> {item_path}")
    print(f"Wrote filtered prompt-score CSVs -> {filtered_dir}")


if __name__ == "__main__":
    main()
