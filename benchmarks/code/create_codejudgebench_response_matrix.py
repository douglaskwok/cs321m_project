"""Create CodeJudgeBench pairwise response matrices from final JSONL results.

The input files are taken from:

    benchmarks/code/results/final files to use/

The output follows the same convention as the other benchmark matrices:
``subject_id`` rows, item columns, plus separate item/subject metadata files.
Unparseable judgments stored as ``correct = -1`` are written as missing values.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "results" / "final files to use"
OUT_DIR = ROOT / "response_matrices"


@dataclass(frozen=True)
class ModelSpec:
    subject_id: str
    model: str
    response_file: Path


MODEL_SPECS = [
    ModelSpec(
        "claude_haiku_4_5",
        "Claude Haiku 4.5",
        FINAL_DIR / "codejudgebench_pairwise_claudehaiku4520251001_oneorder_seed123_verdict_tok16_responses.jsonl",
    ),
    ModelSpec(
        "mistral_3b",
        "Mistral 3B",
        FINAL_DIR / "codejudgebench_pairwise_ministral3b_oneorder_seed123_verdict_tok64_noprefill_responses.jsonl",
    ),
    ModelSpec(
        "mistral_8b",
        "Mistral 8B",
        FINAL_DIR / "codejudgebench_pairwise_ministral8b_oneorder_seed123_verdict_tok64_noprefill_responses.jsonl",
    ),
    ModelSpec(
        "mistral_14b",
        "Mistral 14B",
        FINAL_DIR / "codejudgebench_pairwise_ministral14b_oneorder_seed123_verdict_tok64_noprefill_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_0_8b",
        "Qwen3.5 0.8B",
        FINAL_DIR / "codejudgebench_pairwise_qwen3508b_oneorder_seed123_verdict_tok16_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_2b",
        "Qwen3.5 2B",
        FINAL_DIR / "codejudgebench_pairwise_qwen352b_oneorder_seed123_verdict_tok16_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_4b",
        "Qwen3.5 4B",
        FINAL_DIR / "codejudgebench_pairwise_qwen354b_oneorder_seed123_verdict_tok16_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_9b",
        "Qwen3.5 9B",
        FINAL_DIR / "codejudgebench_pairwise_qwen359b_oneorder_seed123_verdict_tok16_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_27b",
        "Qwen3.5 27B",
        FINAL_DIR / "codejudgebench_pairwise_qwen3527b_oneorder_seed123_verdict_tok16_responses.jsonl",
    ),
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_scores(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    scores: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_rows(path):
        item_id = str(row["item_id"])
        correct = row.get("correct")
        scores[item_id] = "" if correct == -1 or correct is None else int(correct)
        metadata[item_id] = {
            "item_id": item_id,
            "pair_id": row.get("pair_id", ""),
            "question_id": row.get("question_id", ""),
            "split": row.get("split", ""),
            "ordering": row.get("ordering", ""),
            "difficulty": row.get("difficulty", ""),
            "gold": row.get("gold", ""),
        }
    return scores, metadata


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    missing_files = [spec.response_file for spec in MODEL_SPECS if not spec.response_file.exists()]
    if missing_files:
        formatted = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing final CodeJudgeBench files:\n{formatted}")

    reference_rows = read_rows(MODEL_SPECS[0].response_file)
    item_ids = [str(row["item_id"]) for row in reference_rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Duplicate item_id values found in reference file")

    item_metadata = {
        str(row["item_id"]): {
            "item_id": str(row["item_id"]),
            "pair_id": row.get("pair_id", ""),
            "question_id": row.get("question_id", ""),
            "split": row.get("split", ""),
            "ordering": row.get("ordering", ""),
            "difficulty": row.get("difficulty", ""),
            "gold": row.get("gold", ""),
        }
        for row in reference_rows
    }

    matrix_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        scores, metadata = read_scores(spec.response_file)
        if set(scores) != set(item_ids):
            missing = len(set(item_ids) - set(scores))
            extra = len(set(scores) - set(item_ids))
            raise ValueError(
                f"{spec.response_file.name} item mismatch: missing={missing}, extra={extra}"
            )
        if [str(row["item_id"]) for row in read_rows(spec.response_file)] != item_ids:
            raise ValueError(f"{spec.response_file.name} has a different item order")

        values = [scores[item_id] for item_id in item_ids]
        valid = [value for value in values if value != ""]
        n_missing = len(values) - len(valid)
        mean_correct = sum(valid) / len(valid) if valid else ""

        matrix_row = {"subject_id": spec.subject_id}
        for item_id in item_ids:
            matrix_row[item_id] = scores[item_id]
            meta = metadata[item_id]
            long_rows.append(
                {
                    "subject_id": spec.subject_id,
                    "model": spec.model,
                    "item_id": item_id,
                    "correct": scores[item_id],
                    "pair_id": meta.get("pair_id", ""),
                    "question_id": meta.get("question_id", ""),
                    "split": meta.get("split", ""),
                    "ordering": meta.get("ordering", ""),
                    "difficulty": meta.get("difficulty", ""),
                    "gold": meta.get("gold", ""),
                }
            )
        matrix_rows.append(matrix_row)

        subject_rows.append(
            {
                "subject_id": spec.subject_id,
                "model_id": spec.subject_id,
                "model": spec.model,
                "response_file": str(spec.response_file.relative_to(ROOT.parent.parent)),
                "n_rows": len(values),
                "n_scored": len(valid),
                "n_missing": n_missing,
                "mean_correct": mean_correct,
            }
        )

    item_rows = []
    for idx, item_id in enumerate(item_ids):
        row = dict(item_metadata[item_id])
        row["selected_order"] = idx
        item_rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = OUT_DIR / "codejudgebench_pairwise_response_matrix.csv"
    item_path = OUT_DIR / "codejudgebench_pairwise_item_metadata.csv"
    subject_path = OUT_DIR / "codejudgebench_pairwise_subject_metadata.csv"
    long_jsonl_path = OUT_DIR / "codejudgebench_pairwise.jsonl"

    write_csv(matrix_path, matrix_rows, ["subject_id", *item_ids])
    write_csv(
        item_path,
        item_rows,
        [
            "item_id",
            "selected_order",
            "pair_id",
            "question_id",
            "split",
            "ordering",
            "difficulty",
            "gold",
        ],
    )
    write_csv(
        subject_path,
        subject_rows,
        [
            "subject_id",
            "model_id",
            "model",
            "response_file",
            "n_rows",
            "n_scored",
            "n_missing",
            "mean_correct",
        ],
    )
    with long_jsonl_path.open("w", encoding="utf-8") as f:
        for row in long_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"models: {len(MODEL_SPECS)}")
    print(f"items: {len(item_ids)}")
    print(f"matrix: {matrix_path}")
    print(f"item_metadata: {item_path}")
    print(f"subject_metadata: {subject_path}")
    print(f"long_jsonl: {long_jsonl_path}")
    for row in subject_rows:
        print(
            f"{row['subject_id']:<20} scored={row['n_scored']:<4} "
            f"missing={row['n_missing']:<3} mean_correct={row['mean_correct']}"
        )


if __name__ == "__main__":
    main()
