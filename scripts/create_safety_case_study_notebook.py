import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "IRT" / "paper_artifacts" / "safety" / "case_studies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOLVER_MATRIX = ROOT / "benchmarks" / "safety" / "final_solver" / "response_matrices" / "safety_all_attacks_response_matrix.csv"
SOLVER_META = ROOT / "benchmarks" / "safety" / "final_solver" / "response_matrices" / "safety_all_attacks_item_metadata.csv"
JUDGE_MATRIX = ROOT / "benchmarks" / "HarmMetric_Eval" / "response_matrices" / "harmmetric_eval_response_matrix.csv"
PROMPT_LOOKUP = ROOT / "K-Factor" / "results" / "safety_solver_judge_comparison" / "safety_input_index_to_harmmetric_prompt_id.csv"

TARGETS = [
    {
        "case_type": "mistral14_judge_success_solver_failure",
        "model": "Mistral-14B",
        "solver_id": "mistralai_ministral_3_14b_instruct_2512",
        "judge_id": "ministral_3_14b_instruct_2512_bf16",
        "direction": "judge_success_solver_failure",
    },
    {
        "case_type": "qwen2_solver_success_judge_failure",
        "model": "Qwen3.5-2B",
        "solver_id": "qwen_qwen3_5_2b",
        "judge_id": "qwen3_5_2b",
        "direction": "solver_success_judge_failure",
    },
]
N_SELECTED = 10


def read_matrix(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        item_ids = header[1:]
        matrix = {}
        for row in reader:
            subject_id = row[0]
            matrix[subject_id] = dict(zip(item_ids, row[1:]))
        return matrix


def read_dict_by_key(path, key):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row[key]: row for row in reader}


def read_prompt_lookup(path):
    lookup = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[(row["input_index"], row["source"])] = row["prompt_id"]
    return lookup


def as_float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_score(value):
    if value is None:
        return ""
    return f"{value:.3g}"


def select_stratified_by_attack(cases, n_selected):
    """Prefer attack-family diversity over arbitrary alphabetical ordering."""
    by_attack = defaultdict(list)
    for case in cases:
        by_attack[case["attack_family"]].append(case)

    selected = []
    attack_families = sorted(by_attack)
    while len(selected) < n_selected:
        added_this_round = False
        for attack_family in attack_families:
            bucket = by_attack[attack_family]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            added_this_round = True
            if len(selected) >= n_selected:
                break
        if not added_this_round:
            break
    return selected


solver = read_matrix(SOLVER_MATRIX)
judge = read_matrix(JUDGE_MATRIX)
solver_meta = read_dict_by_key(SOLVER_META, "item_id")
prompt_lookup = read_prompt_lookup(PROMPT_LOOKUP)

all_cases = []
summary_rows = []

for target in TARGETS:
    cases = []
    solver_row = solver[target["solver_id"]]
    judge_row = judge[target["judge_id"]]

    for item_id, raw_solver_score in solver_row.items():
        solver_score = as_float(raw_solver_score)
        meta = solver_meta[item_id]
        prompt_id = prompt_lookup.get((meta["input_index"], meta["source"]))
        if prompt_id is None:
            continue
        judge_score = as_float(judge_row.get(prompt_id))
        if judge_score is None:
            continue

        if target["direction"] == "judge_success_solver_failure":
            keep = solver_score == 0.0 and judge_score >= 0.75
            interpretation = "solver failed to stay safe; judge scored the HarmMetric prompt well"
        else:
            keep = solver_score == 1.0 and judge_score <= 0.25
            interpretation = "solver stayed safe; judge scored the HarmMetric prompt poorly"

        if keep:
            cases.append(
                {
                    "case_type": target["case_type"],
                    "model": target["model"],
                    "prompt_id": prompt_id,
                    "item_id": item_id,
                    "attack_family": meta["attack_family"],
                    "input_index": meta["input_index"],
                    "source": meta["source"],
                    "solver_score": solver_score,
                    "judge_score": judge_score,
                    "interpretation": interpretation,
                }
            )

    if target["direction"] == "judge_success_solver_failure":
        cases.sort(key=lambda r: (-r["judge_score"], r["source"], r["attack_family"], int(r["prompt_id"])))
    else:
        cases.sort(key=lambda r: (r["judge_score"], r["source"], r["attack_family"], int(r["prompt_id"])))

    selected = select_stratified_by_attack(cases, N_SELECTED)
    all_cases.extend(selected)
    summary_rows.append(
        {
            "case_type": target["case_type"],
            "model": target["model"],
            "n_available": len(cases),
            "n_selected": len(selected),
            "selection_rule": "judge_score >= 0.75 and solver_score == 0"
            if target["direction"] == "judge_success_solver_failure"
            else "solver_score == 1 and judge_score <= 0.25",
        }
    )

    out_path = OUT_DIR / f"safety_{target['case_type']}_metadata_only.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()) if selected else [])
        if selected:
            writer.writeheader()
            writer.writerows(selected)

combined_path = OUT_DIR / "safety_mistral_qwen2_solver_judge_case_study_examples_metadata_only.csv"
summary_path = OUT_DIR / "safety_mistral_qwen2_solver_judge_case_study_summary.csv"

with combined_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = list(all_cases[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_cases)

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)


def md_table(rows, columns):
    if not rows:
        return "_No rows found._"
    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = format_score(value)
            vals.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


summary_md = md_table(summary_rows, ["model", "case_type", "n_available", "n_selected", "selection_rule"])

sections = [
    "# Safety Case Study: Weak Solvers Can Be Good Judges, and Good Solvers Can Be Weak Judges\n\n"
    "This notebook is intentionally **metadata-only**: it does not display raw harmful prompts or model responses.\n\n"
    "Conventions:\n\n"
    "- `solver_score = 1` means the solver stayed safe / the attack failed.\n"
    "- `solver_score = 0` means the solver failed to stay safe / the attack succeeded.\n"
    "- `judge_score` is the HarmMetric Eval prompt-level score for the judge model; higher is better.\n",
    "## Selection Summary\n\n" + summary_md,
]

for target in TARGETS:
    rows = [row for row in all_cases if row["case_type"] == target["case_type"]]
    sections.append(
        f"## {target['model']}: {target['case_type']}\n\n"
        + md_table(
            rows,
            [
                "model",
                "prompt_id",
                "item_id",
                "attack_family",
                "input_index",
                "source",
                "solver_score",
                "judge_score",
                "interpretation",
            ],
        )
    )

sections.append(
    "## Files Written\n\n"
    + md_table(
        [
            {"artifact": "combined metadata", "path": str(combined_path.relative_to(ROOT))},
            {"artifact": "summary", "path": str(summary_path.relative_to(ROOT))},
            {"artifact": "notebook", "path": "IRT/paper_artifacts/safety/case_studies/safety_mistral_qwen2_case_study_examples.ipynb"},
        ],
        ["artifact", "path"],
    )
)

optional_prompt_cell = f'''# Optional: reveal prompt text in the notebook UI.
# Default is False so this notebook stays metadata-only unless you opt in.
SHOW_PROMPTS = False

import csv
import json
from pathlib import Path
from IPython.display import display

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "K-Factor").exists() and (candidate / "IRT").exists():
            return candidate
    raise FileNotFoundError("Could not find repo root containing K-Factor/ and IRT/.")

repo_root = find_repo_root(Path.cwd())
case_path = repo_root / "{combined_path.relative_to(ROOT)}"
lookup_path = repo_root / "K-Factor/results/safety_solver_judge_comparison/safety_input_index_to_harmmetric_prompt_id.csv"
raw_solver_path = repo_root / "benchmarks/safety/final_solver/all_attacks_all_models_scored.json"
judge_score_files = {{
    "Mistral-14B": repo_root / "benchmarks/HarmMetric_Eval/Final files to use/Ministral-3-14B-Instruct-2512-BF16-HarmJudge-safety_solver_prompt_scores.csv",
    "Qwen3.5-2B": repo_root / "benchmarks/HarmMetric_Eval/Final files to use/Qwen3.5-2B-HarmJudge-safety_solver_prompt_scores.csv",
}}
raw_model_candidates = {{
    "Mistral-14B": [
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "mistralai/Ministral-3-14B-Instruct-2512",
    ],
    "Qwen3.5-2B": ["Qwen/Qwen3.5-2B"],
}}

if SHOW_PROMPTS:
    with case_path.open(newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    prompt_lookup = {{}}
    with lookup_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt_lookup[(row["input_index"], row["source"])] = row.get("original_prompt", "")

    with raw_solver_path.open(encoding="utf-8") as f:
        raw_solver_rows = json.load(f)
    response_lookup = {{}}
    for row in raw_solver_rows:
        response_lookup[
            (
                row.get("model", ""),
                str(row.get("attack_family", "")),
                str(row.get("input_index", "")),
                str(row.get("source", "")),
            )
        ] = row.get("response", "")

    judge_prompt_scores = {{}}
    for model, path in judge_score_files.items():
        model_scores = {{}}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                model_scores[row["prompt_id"]] = row
        judge_prompt_scores[model] = model_scores

    rows = []
    for row in cases:
        enriched = dict(row)
        enriched["prompt"] = prompt_lookup.get((row["input_index"], row["source"]), "")
        solver_response = ""
        for raw_model in raw_model_candidates.get(row["model"], []):
            solver_response = response_lookup.get(
                (raw_model, row["attack_family"], row["input_index"], row["source"]),
                "",
            )
            if solver_response:
                break
        enriched["solver_response"] = solver_response
        judge_row = judge_prompt_scores.get(row["model"], {{}}).get(row["prompt_id"], {{}})
        enriched["judge_overall_effectiveness_score"] = judge_row.get("overall_effectiveness_score", row.get("judge_score", ""))
        enriched["judge_component_scores"] = (
            f"refusal={{judge_row.get('refusal', '')}}; "
            f"prevention={{judge_row.get('prevention', '')}}; "
            f"redirection={{judge_row.get('redirection', '')}}; "
            f"irrelevant={{judge_row.get('irrelevant_response', '')}}; "
            f"prompt_repetition={{judge_row.get('prompt_repetition', '')}}; "
            f"useless_affirmation={{judge_row.get('useless_affirmation', '')}}"
        )
        enriched["judge_score_bounds"] = (
            f"min_harmful={{judge_row.get('min_harmful_score', '')}}; "
            f"max_safe={{judge_row.get('max_safe_score', '')}}; "
            f"max_irrelevant_or_useless={{judge_row.get('max_irrelevant_or_useless_score', '')}}"
        )
        rows.append(enriched)

    try:
        import pandas as pd
        display(pd.DataFrame(rows))
    except Exception:
        for row in rows:
            print("\\n" + "=" * 80)
            for key in [
                "case_type",
                "model",
                "prompt_id",
                "item_id",
                "attack_family",
                "source",
                "solver_score",
                "judge_score",
                "prompt",
                "solver_response",
                "judge_overall_effectiveness_score",
                "judge_component_scores",
                "judge_score_bounds",
            ]:
                print(f"{{key}}: {{row.get(key, '')}}")
else:
    print("Set SHOW_PROMPTS = True and rerun this cell to display prompt text.")
'''

download_cell = '''# Download the spreadsheet from inside Jupyter.
from pathlib import Path
from IPython.display import FileLink, display

xlsx_path = Path("safety_mistral_qwen2_solver_judge_case_study_examples_with_prompts.xlsx")
csv_path = Path("safety_mistral_qwen2_solver_judge_case_study_examples_metadata_only.csv")

if xlsx_path.exists():
    display(FileLink(str(xlsx_path), result_html_prefix="Download Excel: "))
else:
    print(f"Excel file not found at {xlsx_path}.")

if csv_path.exists():
    display(FileLink(str(csv_path), result_html_prefix="Download CSV: "))
'''

nb = {
    "cells": (
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": section.splitlines(keepends=True),
            }
            for section in sections[:-1]
        ]
        + [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Optional Prompt Inspection\n",
                    "\n",
                    "The cell below can reveal prompt text for the selected metadata rows. It is off by default.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": optional_prompt_cell.splitlines(keepends=True),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Download Spreadsheet\n",
                    "\n",
                    "Run the cell below to get clickable download links from Jupyter.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": download_cell.splitlines(keepends=True),
            },
        ]
        + [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": sections[-1].splitlines(keepends=True),
            }
        ]
    ),
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

notebook_path = OUT_DIR / "safety_mistral_qwen2_case_study_examples.ipynb"
notebook_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")

print(notebook_path.relative_to(ROOT))
print(combined_path.relative_to(ROOT))
print(summary_path.relative_to(ROOT))
