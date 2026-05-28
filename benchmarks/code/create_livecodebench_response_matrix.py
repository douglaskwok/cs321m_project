"""Create LiveCodeBench response matrices from per-model JSONL results.

Outputs are filtered to the 676 CodeJudgeBench-overlapping LiveCodeBench items
listed in ``benchmarks/code/codegen_selected_pairs.json``.  Qwen 27B uses the
``codegen_nothink`` result file; all other models use their regular
``*_responses.jsonl`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SELECTED_PAIRS_FILE = ROOT / "codegen_selected_pairs.json"
OUT_DIR = ROOT / "response_matrices"

DATASET_ID = "livecodebench/code_generation_lite"
VERSION_TAG = "release_v6"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    response_file: Path


MODEL_SPECS = [
    ModelSpec(
        "claude_haiku_4_5",
        "Claude Haiku 4.5",
        RESULTS_DIR / "livecodebench_claudehaiku4520251001_responses.jsonl",
    ),
    ModelSpec(
        "mistral_3b",
        "Mistral 3B",
        RESULTS_DIR / "livecodebench_mistral3b_responses.jsonl",
    ),
    ModelSpec(
        "mistral_8b",
        "Mistral 8B",
        RESULTS_DIR / "livecodebench_mistral8b_responses.jsonl",
    ),
    ModelSpec(
        "mistral_14b",
        "Mistral 14B",
        RESULTS_DIR / "livecodebench_mistral14b_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_0_8b",
        "Qwen3.5 0.8B",
        RESULTS_DIR / "livecodebench_qwen3508b_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_2b",
        "Qwen3.5 2B",
        RESULTS_DIR / "livecodebench_qwen352b_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_4b",
        "Qwen3.5 4B",
        RESULTS_DIR / "livecodebench_qwen354b_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_9b",
        "Qwen3.5 9B",
        RESULTS_DIR / "livecodebench_qwen359b_responses.jsonl",
    ),
    ModelSpec(
        "qwen3_5_27b",
        "Qwen3.5 27B",
        RESULTS_DIR / "livecodebench_qwen3527b_responses_codegen_nothink.jsonl",
    ),
]


_FORMAT_WITH_STARTER = (
    "### Question:\n{question}\n\n"
    "### Format: You will use the following starter code to write the solution "
    "to the problem and enclose your code within delimiters.\n"
    "```python\n{starter_code}\n```\n\n"
    "### Answer: (use the provided format with backticks)\n"
)

_FORMAT_STDIN = (
    "### Question:\n{question}\n\n"
    "### Format: Read the inputs from stdin solve the problem and write the "
    "answer to stdout (do not directly test on the sample inputs). Enclose your "
    "code within delimiters as follows. Ensure that when the python program "
    "runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
    "```python\n# YOUR CODE HERE\n```\n\n"
    "### Answer: (use the provided format with backticks)\n"
)


def build_solver_prompt(question: str, starter_code: str) -> str:
    question = (question or "").strip()
    starter_code = (starter_code or "").strip()
    if starter_code:
        return _FORMAT_WITH_STARTER.format(
            question=question,
            starter_code=starter_code,
        )
    return _FORMAT_STDIN.format(question=question)


def read_selected_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        selected_pairs = json.load(f)
    allowed_ids = [str(row["question_id"]) for row in selected_pairs]
    return sorted(set(allowed_ids))


def read_jsonl_scores(path: Path, selected: set[str]) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    scores: dict[str, int] = {}
    metadata: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return scores, metadata
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("id", ""))
            if item_id not in selected:
                continue
            correct = row.get("correct")
            scores[item_id] = int(correct) if correct is not None else None
            metadata[item_id] = {
                "difficulty": row.get("difficulty", ""),
                "platform": row.get("platform", ""),
            }
    return scores, metadata


def load_livecodebench_metadata(selected: set[str], include_prompt: bool = True) -> dict[str, dict[str, Any]]:
    """Load item metadata from the HF dataset, if available."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"Could not import datasets; prompt metadata skipped: {exc}")
        return {}

    try:
        ds = load_dataset(
            DATASET_ID,
            version_tag=VERSION_TAG,
            split="test",
            trust_remote_code=True,
        )
    except TypeError:
        ds = load_dataset(DATASET_ID, version_tag=VERSION_TAG, split="test")
    except Exception as exc:
        print(f"Could not load {DATASET_ID}; prompt metadata skipped: {exc}")
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    for row in ds:
        item_id = str(row.get("question_id", ""))
        if item_id not in selected:
            continue
        question = row.get("question_content", "")
        starter_code = row.get("starter_code", "")
        metadata[item_id] = {
            "item_id": item_id,
            "difficulty": str(row.get("difficulty", "")),
            "platform": str(row.get("platform", "")),
            "question_content": question if include_prompt else "",
            "starter_code": starter_code if include_prompt else "",
            "prompt": build_solver_prompt(question, starter_code) if include_prompt else "",
        }
    return metadata


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--fetch-prompts",
        action="store_true",
        help="Try to load LiveCodeBench from Hugging Face for prompt text.",
    )
    args = parser.parse_args()

    selected_ids = read_selected_ids(SELECTED_PAIRS_FILE)
    selected = set(selected_ids)
    if len(selected_ids) != 676:
        raise ValueError(f"Expected 676 selected IDs, found {len(selected_ids)}")

    item_metadata = load_livecodebench_metadata(selected) if args.fetch_prompts else {}

    matrix_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    fallback_item_metadata: dict[str, dict[str, Any]] = {}

    for spec in MODEL_SPECS:
        scores, response_metadata = read_jsonl_scores(spec.response_file, selected)
        fallback_item_metadata.update(response_metadata)
        subject_rows.append(
            {
                "subject_id": spec.model_id,
                "model_id": spec.model_id,
                "model": spec.display_name,
                "response_file": str(spec.response_file.relative_to(ROOT.parent.parent)),
                "n_selected_scored": len(scores),
                "mean_correct": (
                    sum(v for v in scores.values() if v is not None) / len(scores)
                    if scores
                    else ""
                ),
            }
        )
        row = {"subject_id": spec.model_id}
        for item_id in selected_ids:
            row[item_id] = scores.get(item_id, "")
            meta = item_metadata.get(item_id) or fallback_item_metadata.get(item_id, {})
            long_rows.append(
                {
                    "subject_id": spec.model_id,
                    "model_id": spec.model_id,
                    "model": spec.display_name,
                    "item_id": item_id,
                    "correct": scores.get(item_id, None),
                    "difficulty": meta.get("difficulty", ""),
                    "platform": meta.get("platform", ""),
                    "prompt": meta.get("prompt", ""),
                }
            )
        matrix_rows.append(row)

    item_rows = []
    for idx, item_id in enumerate(selected_ids):
        meta = item_metadata.get(item_id) or fallback_item_metadata.get(item_id, {})
        item_rows.append(
            {
                "item_id": item_id,
                "selected_order": idx,
                "difficulty": meta.get("difficulty", ""),
                "platform": meta.get("platform", ""),
                "question_content": meta.get("question_content", ""),
                "starter_code": meta.get("starter_code", ""),
                "prompt": meta.get("prompt", ""),
            }
        )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = out_dir / "livecodebench_response_matrix.csv"
    item_path = out_dir / "livecodebench_item_metadata.csv"
    subject_path = out_dir / "livecodebench_subject_metadata.csv"
    long_jsonl_path = out_dir / "livecodebench.jsonl"

    write_csv(matrix_path, matrix_rows, ["subject_id", *selected_ids])
    write_csv(
        item_path,
        item_rows,
        [
            "item_id",
            "selected_order",
            "difficulty",
            "platform",
            "question_content",
            "starter_code",
            "prompt",
        ],
    )
    write_csv(
        subject_path,
        subject_rows,
        ["subject_id", "model_id", "model", "response_file", "n_selected_scored", "mean_correct"],
    )
    with long_jsonl_path.open("w", encoding="utf-8") as f:
        for row in long_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"selected_items: {len(selected_ids)}")
    print(f"models: {len(MODEL_SPECS)}")
    print(f"matrix: {matrix_path}")
    print(f"item_metadata: {item_path}")
    print(f"subject_metadata: {subject_path}")
    print(f"long_jsonl: {long_jsonl_path}")
    for row in subject_rows:
        print(
            f"{row['model_id']:<20} scored={row['n_selected_scored']:<4} "
            f"mean_correct={row['mean_correct']}"
        )


if __name__ == "__main__":
    main()
