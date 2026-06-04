# K-factor Charts and Tables

Curated bundle of paper-ready K-factor figures, tables, chart data, model-selection summaries, and compact case-study outputs.

Included for each domain (`kudge`, `mmlu`, `safety`, `code`):

- `figures/`: paper-facing K-factor comparison charts (`.png` and `.pdf`).
- `model_selection/`: K-factor fit summaries and selected-k summaries.
- `chart_data/`: compact CSV/JSON files used directly in charts.
- `case_studies/`: compact domain-specific case-study exports where available.

At the artifact root:

- `combined_figures/`: all-domain composite charts for the paper.
- `tables/`: paper table sources in CSV and LaTeX.
- `manifest.json`: machine-readable index of selected K values and included paths.

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

Combined figures:

- `combined_figures/all_domains_judge_model_bin_summary.{png,pdf}`: all-domain version of the judge-performance-by-bin chart.
- `combined_figures/all_domains_solver_judge_difficulty_scatter.{png,pdf}`: all-domain version of the raw solver-vs-judge difficulty scatter chart.
- `combined_figures/all_domains_k2_judge_model_bin_summary.{png,pdf}`: descriptive K=2 all-domain version.
- `combined_figures/all_domains_k2_solver_judge_difficulty_scatter.{png,pdf}`: descriptive K=2 raw solver-vs-judge difficulty scatter.

Model selection:

- Current paper-facing artifacts use lower in-sample loss for K selection.
- Selected K values are summarized in `selected_k_summary_all_domains.csv`.
- Current selections: KUDGE solver/judge K=2, MMLU solver/judge K=2, safety solver K=2 and HarmMetric judge K=2, code solver/judge K=2.
- In-sample reconstruction summaries are also provided as `*_kfactor_fit_summary_insample.csv` under each domain's `model_selection/` folder, plus `kfactor_fit_summary_insample_all_domains.csv` and `selected_k_by_insample_loss_all_domains.csv` at the artifact root.
- Descriptive K=2 chart/data artifacts are included for all four domains for exploratory item-structure analysis.

Path conventions:

- Paths in `manifest.json` are relative to `K-Factor/charts_and_tables/`.
- `source_file` columns in table CSVs use repository-root-relative paths.
