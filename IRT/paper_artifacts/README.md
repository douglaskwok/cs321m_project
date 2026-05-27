# IRT Paper Artifacts

This folder collects the IRT outputs that are useful for paper writing.

## Model Selection

- `model_selection/all_irt_model_selection_compact.csv`: combined compact table across MMLU, Kudge, and Safety solver/judge runs. This is the main table for comparing AUC and AIC.
- `model_selection/selected_fit_by_auc_and_aic.csv`: one-row summary per domain/role with the best fit by AUC and the best fit by AIC.
- `model_selection/*_model_selection_compact.csv`: per-run compact tables.
- `model_selection/full/`: copied full heldout evaluation summaries, including JSON backups.

Compact model-selection tables include:

- `heldout_auc_mean`, `heldout_auc_sd`
- `aic`
- `heldout_ece_mean`, `heldout_ece_sd`
- `heldout_log_likelihood_mean`, `heldout_log_likelihood_sd`
- `bic`
- `best_by_auc`, `best_by_aic`

## Charts

- `charts/irt_all_solver_judge_scatter_se.png`
- `charts/irt_all_solver_judge_scatter_se.pdf`
- `charts/irt_mmlu_solver_judge_scatter_se.*`
- `charts/irt_kudge_solver_judge_scatter_se.*`
- `charts/irt_safety_solver_judge_scatter_se.*`

These are solver-vs-judge IRT ability scatter plots with standard-error bars.

## Data

- `data/irt_*_solver_judge_scatter_data.csv`: plotted scatter data after model-name normalization.
- `data/*_solver_judge_capability_table.csv`: copied exported solver/judge capability tables.

## Manifest

- `manifest.json`: machine-readable list of copied/generated artifacts and source paths.
