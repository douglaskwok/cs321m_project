# K-Factor/

Logistic factor model (K-Factor) fitting, evaluation, and comparison of solver vs. judge item difficulty.

`LogisticFM` with K=1 and K=2 factors is fitted per domain. Item factors are rotated via varimax for interpretability. Model selection uses in-sample reconstruction loss (lower is better).

## Notebooks and scripts

| File                                           | Purpose                                                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `kfactor.ipynb`                                | Main K-Factor fitting notebook for binary response matrices; set `KFACTOR_MATRIX` to select domain |
| `kfactor_continuous.ipynb`                     | K-Factor notebook for continuous response scores (HarmMetric judge)                                |
| `run_all_kfactor_notebooks.py`                 | Batch-executes `kfactor.ipynb` and `kfactor_continuous.ipynb` for all domains                      |
| `compare_code_solver_judge_difficulty.ipynb`   | Compares solver vs. judge item difficulty for Coding                                               |
| `compare_mmlu_solver_judge_difficulty.ipynb`   | Compares solver vs. judge item difficulty for MMLU                                                 |
| `compare_kudge_solver_judge_difficulty.ipynb`  | Compares solver vs. judge item difficulty for KUDGE                                                |
| `compare_safety_solver_judge_difficulty.ipynb` | Compares solver vs. judge item difficulty for Safety                                               |
| `extract_kudge_case_studies.ipynb`             | Extracts case-study items from KUDGE K-Factor results                                              |
| `extract_mmlu_case_studies.ipynb`              | Extracts case-study items from MMLU K-Factor results                                               |
| `extract_safety_case_studies.ipynb`            | Extracts case-study items from Safety K-Factor results                                             |

## Usage

```bash
# Run all domains in batch
python K-Factor/run_all_kfactor_notebooks.py

# Run a single domain interactively:
# Open kfactor.ipynb and set KFACTOR_MATRIX to one of:
#   mmlu_solver, mmlu_judging, safety_all_attacks,
#   code_solver, code_judge, kudge_challenge, kudge_judge

# Generate comparison figures for each domain from the repository root
jupyter nbconvert --to notebook --execute K-Factor/compare_code_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_mmlu_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_safety_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_kudge_solver_judge_difficulty.ipynb
```

## Outputs

- `results/` and `results_continuous/` — raw K-Factor fit summaries, item loadings, and intermediate comparison outputs
- `charts_and_tables/` — curated paper-ready figures, tables, chart data, and model-selection summaries (see `charts_and_tables/README.md`)

## Selected K values (used in paper)

| Domain | Solver K | Judge K        |
| ------ | -------- | -------------- |
| MMLU   | 2        | 2              |
| KUDGE  | 2        | 2              |
| Coding | 2        | 2              |
| Safety | 2        | 2 (HarmMetric) |
