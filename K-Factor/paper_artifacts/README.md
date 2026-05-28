# K-factor Paper Artifacts

Compact bundle for paper writing.

Included for each domain (`kudge`, `mmlu`, `safety`, `code`):

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

Model selection:

- Current paper-facing artifacts use lower held-out loss for K selection.
- Selected K values are summarized in `selected_k_summary_all_domains.csv`.
- Current selections: KUDGE solver/judge K=1, MMLU solver/judge K=1, safety solver K=1 and HarmMetric judge K=2, code solver/judge K=1.

Note: older K=2 artifacts and case-study files may still be present for traceability, but the current paper-facing figures and chart data use the selected lower-loss prefixes (`kudge_k1`, `mmlu_k1`, `safety_k1`, `code_solverk1_judgek1`).
