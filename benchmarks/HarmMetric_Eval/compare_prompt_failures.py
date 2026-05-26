import csv
import json
from pathlib import Path


CATEGORY_NAMES = {
    1: "harmful_1",
    2: "harmful_2",
    3: "harmful_3",
    4: "harmful_4",
    5: "refusal",
    6: "prevention",
    7: "redirection",
    8: "irrelevant_1",
    9: "irrelevant_2",
    10: "irrelevant_3",
    11: "prompt_repetition",
    12: "useless_affirmation",
}


def pid(response_id):
    return response_id // 100


def cid(response_id):
    return response_id % 100


def read_prompt_scores(path):
    with path.open("r", encoding="utf-8") as f:
        return {int(row["prompt_id"]): row for row in csv.DictReader(f)}


def read_item_scores(path):
    scores = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            scores.setdefault(pid(row["response_id"]), {})[cid(row["response_id"])] = row["score"]
    return scores


def main():
    root = Path(__file__).resolve().parent
    p4 = read_prompt_scores(root / "results/Qwen3.5-4B-HarmJudge-safety_solver_prompt_scores.csv")
    p9 = read_prompt_scores(root / "results/Qwen3.5-9B-HarmJudge-safety_solver_prompt_scores.csv")
    i4 = read_item_scores(root / "temp_results/Qwen3.5-4B-HarmJudge-safety_solver.jsonl")
    i9 = read_item_scores(root / "temp_results/Qwen3.5-9B-HarmJudge-safety_solver.jsonl")

    rows = []
    for prompt_id in sorted(p4):
        if float(p4[prompt_id]["overall_effectiveness_score"]) != 1.0:
            continue
        if float(p9[prompt_id]["overall_effectiveness_score"]) != 0.0:
            continue

        row = {
            "prompt_id": prompt_id,
            "source": p4[prompt_id]["source"],
            "harmful_prompt": p4[prompt_id]["harmful_prompt"],
            "4B_overall": p4[prompt_id]["overall_effectiveness_score"],
            "9B_overall": p9[prompt_id]["overall_effectiveness_score"],
            "4B_min_harmful": p4[prompt_id]["min_harmful_score"],
            "9B_min_harmful": p9[prompt_id]["min_harmful_score"],
            "4B_max_safe": p4[prompt_id]["max_safe_score"],
            "9B_max_safe": p9[prompt_id]["max_safe_score"],
            "4B_max_irrelevant_or_useless": p4[prompt_id]["max_irrelevant_or_useless_score"],
            "9B_max_irrelevant_or_useless": p9[prompt_id]["max_irrelevant_or_useless_score"],
        }
        for cat_id, name in CATEGORY_NAMES.items():
            row[f"4B_{name}"] = i4[prompt_id].get(cat_id)
            row[f"9B_{name}"] = i9[prompt_id].get(cat_id)
        rows.append(row)

    out = root / "results/Qwen3.5-4B_vs_9B_full_to_zero_diagnostics.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} diagnostics to {out}")


if __name__ == "__main__":
    main()
