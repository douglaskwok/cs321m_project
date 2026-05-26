"""Build the 676-pair codegen evaluation subset from CodeJudgeBench.

Loads the codegen config of mattymchen/codejudgebench, finds all 676 unique
question_ids, and selects one row per question_id stratified evenly across
the claude, qwen, and gemini model families (~225–226 each).

Run locally (no Modal):

    pip install datasets
    python benchmarks/code/select_codegen_subset.py

Output: benchmarks/code/results/codegen_selected_pairs.json
  A JSON array of 676 objects, each with keys:
    question_id   – the LCB question identifier
    split         – dataset split the row came from
    model         – exact model string from the row
    model_family  – one of "claude", "qwen", "gemini"
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DATASET_ID = "mattymchen/codejudgebench"
CONFIG = "codegen"
TARGET_FAMILIES = ["claude", "qwen", "gemini"]
EXPECTED_UNIQUE_IDS = 676
OUTPUT_FILE = Path(__file__).parent / "results" / "codegen_selected_pairs.json"


def detect_family(model_str: str) -> str | None:
    s = model_str.lower()
    for f in TARGET_FAMILIES:
        if f in s:
            return f
    return None


def find_model_field(row: dict) -> str:
    for candidate in ["model", "generator", "model_name", "llm", "source"]:
        val = row.get(candidate)
        if isinstance(val, str) and detect_family(val) is not None:
            return candidate
    for k, v in row.items():
        if isinstance(v, str) and detect_family(v) is not None:
            return k
    return "model"


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, CONFIG, trust_remote_code=True)
    print(f"Splits: {list(ds.keys())}")

    first_split = next(iter(ds.values()))
    print(f"Columns: {first_split.column_names}")
    model_field = find_model_field(dict(first_split[0]))
    print(f"Using model field: {model_field!r}")

    by_question: dict[str, list[dict]] = defaultdict(list)
    for split_name, split_ds in ds.items():
        for row in split_ds:
            qid = str(row["question_id"])
            model_str = str(row.get(model_field, ""))
            by_question[qid].append(
                {
                    "question_id": qid,
                    "split": split_name,
                    "model": model_str,
                    "model_family": detect_family(model_str),
                }
            )

    n_unique = len(by_question)
    print(f"Unique question_ids: {n_unique}")
    if n_unique != EXPECTED_UNIQUE_IDS:
        print(f"WARNING: expected {EXPECTED_UNIQUE_IDS}, got {n_unique}")

    sorted_qids = sorted(by_question.keys())
    selected: list[dict] = []
    family_counts: dict[str, int] = defaultdict(int)

    for i, qid in enumerate(sorted_qids):
        rows = by_question[qid]
        target_family = TARGET_FAMILIES[i % len(TARGET_FAMILIES)]

        # Prefer the target family; fall back to any target family; then any row.
        row = next((r for r in rows if r["model_family"] == target_family), None)
        if row is None:
            for f in TARGET_FAMILIES:
                row = next((r for r in rows if r["model_family"] == f), None)
                if row:
                    break
        if row is None:
            row = rows[0]

        selected.append(row)
        family_counts[row["model_family"] or "unknown"] += 1

    print(f"Selected {len(selected)} pairs")
    print(f"Family distribution: {dict(sorted(family_counts.items()))}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
