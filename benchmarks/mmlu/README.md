# benchmarks/mmlu/

MMLU-Pro multiple-choice knowledge (solver) and JudgeBench-MMLU pairwise judge benchmarks.

## Scripts

| Script | Purpose |
|---|---|
| `run_qwen35_solving_modal.py` | Runs Qwen3.5 models on MMLU-Pro/JudgeBench solving task via Modal |
| `run_qwen35_judging_modal.py` | Runs Qwen3.5 models as pairwise judges on MMLU-Pro items via Modal |
| `run_anthropic_judgebench.py` | Runs Anthropic Claude models on the JudgeBench judging task |
| `create_solver_response_matrix.py` | Aggregates per-model solving results into a response matrix CSV |
| `create_judging_response_matrix.py` | Aggregates per-model judging results into a response matrix CSV |
| `check_judgebench_against_mmlu_pro.py` | Validates JudgeBench question alignment with MMLU-Pro |

## Data files

- `judgebench_mmlu_pro_questions_common_43.csv` — 43 questions common to both JudgeBench and MMLU-Pro
- `judgebench_mmlu_pro_questions_unique.csv/.jsonl` — questions unique to JudgeBench
- `judgebench_mmlu_pro_answer_key_checked.csv/.jsonl` — verified answer keys
- `judgebench_mmlu_pro_knowledge_by_split.jsonl` — knowledge-domain split metadata
- `final files to use/` — curated final inputs used in the paper runs

## Usage

```bash
modal run benchmarks/mmlu/run_qwen35_solving_modal.py
modal run benchmarks/mmlu/run_qwen35_judging_modal.py

python benchmarks/mmlu/create_solver_response_matrix.py
python benchmarks/mmlu/create_judging_response_matrix.py
```

## Outputs

- `response_matrices/mmlu_pro_solver_response_matrix.csv` — binary solver matrix (models × questions)
- `response_matrices/mmlu_pro_judging_response_matrix.csv` — binary judge matrix (models × question pairs)
- `response_matrices/*_item_metadata.csv` / `*_subject_metadata.csv`

## Data source

- MMLU-Pro: `TIGER-Lab/MMLU-Pro` (HuggingFace)
- JudgeBench: `ScalerLab/JudgeBench` (HuggingFace)
