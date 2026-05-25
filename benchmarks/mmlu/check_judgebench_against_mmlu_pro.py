import csv
import json
import re
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "judgebench_mmlu_pro_answer_key_checked.csv"
OUT_JSONL = ROOT / "judgebench_mmlu_pro_answer_key_checked.jsonl"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_judgebench_suffix(question: str) -> str:
    question = question.replace(
        "If you cannot determine the correct multiple-choice answer, take your best guess.",
        "",
    )
    question = re.sub(
        r"Once you have your answer, please duplicate that letter five times in a single string\."
        r"\s*For example, if the answer is K, then write KKKKK\."
        r"\s*Let's think step by step\.?",
        "",
        question,
        flags=re.IGNORECASE,
    )
    question = re.sub(
        r"Once you have your answer, please duplicate that letter five times in a single string\."
        r"\s*Let's think step by step\.?",
        "",
        question,
        flags=re.IGNORECASE,
    )
    return normalize_text(question)


def format_mmlu_question(row: dict) -> str:
    options = "\n".join(
        f"({chr(65 + i)}) {option}" for i, option in enumerate(row["options"])
    )
    return normalize_text(f"{row['question']}\n{options}")


def extract_repeated_letter(text: str) -> str | None:
    matches = re.findall(r"\b([A-J])\1{2,}\b", text.upper())
    return matches[-1] if matches else None


def load_judgebench_rows() -> list[dict]:
    rows = []
    for split in ["gpt", "claude"]:
        for row in load_dataset("ScalerLab/JudgeBench", split=split):
            row = dict(row)
            if str(row["source"]).startswith("mmlu-pro"):
                row["split"] = split
                rows.append(row)
    return rows


def main() -> None:
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    mmlu_by_id = {int(row["question_id"]): dict(row) for row in mmlu}
    mmlu_by_text = {format_mmlu_question(dict(row)): dict(row) for row in mmlu}

    checked = []
    misses = []
    for row in load_judgebench_rows():
        original_id = int(row["original_id"])
        mmlu_row = mmlu_by_id.get(original_id)
        match_method = "question_id"

        stripped = strip_judgebench_suffix(row["question"])
        if mmlu_row is None:
            mmlu_row = mmlu_by_text.get(stripped)
            match_method = "formatted_question_text" if mmlu_row else "none"

        if mmlu_row is None:
            misses.append(row)
            continue

        mmlu_formatted = format_mmlu_question(mmlu_row)
        text_matches = stripped == mmlu_formatted
        answer = mmlu_row["answer"]
        winner = row["response_A"] if row["label"] == "A>B" else row["response_B"]
        winner_letter = extract_repeated_letter(winner)

        checked.append(
            {
                "split": row["split"],
                "pair_id": row["pair_id"],
                "judgebench_original_id": original_id,
                "mmlu_question_id": int(mmlu_row["question_id"]),
                "source": row["source"],
                "mmlu_category": mmlu_row["category"],
                "match_method": match_method,
                "question_text_matches_mmlu": text_matches,
                "mmlu_answer": answer,
                "mmlu_answer_index": int(mmlu_row["answer_index"]),
                "judgebench_label": row["label"],
                "winner_final_letter": winner_letter,
                "winner_matches_mmlu_answer": winner_letter == answer,
                "question": row["question"],
                "mmlu_question": mmlu_row["question"],
                "mmlu_options": json.dumps(mmlu_row["options"], ensure_ascii=False),
            }
        )

    if misses:
        raise RuntimeError(f"Could not match {len(misses)} JudgeBench MMLU-Pro rows")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in checked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(checked[0].keys()))
        writer.writeheader()
        writer.writerows(checked)

    print(f"checked_rows={len(checked)}")
    print(f"unique_pair_ids={len(set(r['pair_id'] for r in checked))}")
    print(f"unique_mmlu_question_ids={len(set(r['mmlu_question_id'] for r in checked))}")
    print(
        "question_text_matches_mmlu="
        f"{sum(r['question_text_matches_mmlu'] for r in checked)}/{len(checked)}"
    )
    print(
        "winner_matches_mmlu_answer="
        f"{sum(r['winner_matches_mmlu_answer'] for r in checked)}/{len(checked)}"
    )
    bad = [r for r in checked if not r["winner_matches_mmlu_answer"]]
    if bad:
        print("mismatched winners:")
        for row in bad[:10]:
            print(
                row["split"],
                row["judgebench_original_id"],
                row["mmlu_answer"],
                row["winner_final_letter"],
                row["source"],
            )
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
