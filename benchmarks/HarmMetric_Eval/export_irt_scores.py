import argparse
import csv
import json
from pathlib import Path


def load_dataset(dataset_path):
    rows_by_response_id = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows_by_response_id[row["response_id"]] = row
    return rows_by_response_id


def export_irt_scores(metric_stem, dataset_path, temp_dir, out_path):
    rows_by_response_id = load_dataset(dataset_path)
    score_path = temp_dir / f"{metric_stem}.jsonl"

    output_rows = []
    with score_path.open("r", encoding="utf-8") as f:
        for line in f:
            score_row = json.loads(line)
            response_id = score_row["response_id"]
            dataset_row = rows_by_response_id[response_id]
            output_rows.append(
                {
                    "model_metric": metric_stem,
                    "prompt_id": dataset_row["prompt_id"],
                    "response_id": response_id,
                    "response_category_id": response_id % 100,
                    "response_category": dataset_row["response_category"],
                    "attack_method": dataset_row.get("attack_method", ""),
                    "source": dataset_row.get("source", ""),
                    "score": score_row["score"],
                    "score_0_1": score_row["score"] / 2,
                    "harmful_prompt": dataset_row["harmful_prompt"],
                    "response": dataset_row["response"],
                }
            )

    output_rows.sort(key=lambda row: (row["prompt_id"], row["response_category_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    return len(output_rows)


def main():
    parser = argparse.ArgumentParser(description="Export HarmMetric per-response scores for IRT analysis")
    parser.add_argument("--metric-stem", required=True, help="Filename stem in temp_results, without .jsonl")
    parser.add_argument("--dataset", default="data/dataset.jsonl")
    parser.add_argument("--temp-dir", default="temp_results")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metric_stem = args.metric_stem
    out = args.out or f"results/{metric_stem}_irt_items.csv"
    count = export_irt_scores(
        metric_stem=metric_stem,
        dataset_path=Path(args.dataset),
        temp_dir=Path(args.temp_dir),
        out_path=Path(out),
    )
    print(f"Wrote {count} item scores to {Path(out).resolve()}")


if __name__ == "__main__":
    main()
