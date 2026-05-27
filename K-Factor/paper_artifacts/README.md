# K-factor Paper Artifacts

Compact bundle for paper writing.

Included for each domain (`kudge`, `mmlu`, `safety`):

- `figures/`: paper-facing K-factor comparison charts (`.png` and `.pdf`).
- `model_selection/`: K-factor fit summaries and selected-k summaries.
- `chart_data/`: compact CSV/JSON files used directly in charts.

Intentionally omitted:

- raw paired-item tables
- prompt-heavy item metadata
- full per-item score dictionaries

Part 1 numbers:

- `*_difficulty_bin_summary.csv`: aggregate mean judge score by solver difficulty bin.
- `*_judge_model_bin_summary_mean.csv`: mean and SD across judge models by solver difficulty bin.
- `part1_judge_model_bin_means_and_sd_all_domains.csv`: combined version across domains.

Part 2 numbers:

- `*_solver_judge_difficulty_scatter.csv`: values used in raw solver-vs-judge difficulty scatter plots.
- `*_solver_judge_difficulty_percentiles.csv`: values used in percentile-vs-percentile plots.

Note: safety artifacts reflect the currently saved `K-Factor/results/safety_solver_judge_comparison` outputs. If HarmMetric Eval K-factor was rerun after adding a model, rerun `compare_safety_solver_judge_difficulty.ipynb` and recreate this bundle.
