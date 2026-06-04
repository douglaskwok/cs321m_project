# IRT/

Item Response Theory (IRT) fitting, evaluation, and figure generation.

IRT models infer latent *ability* for each model and latent *difficulty* (plus discrimination and guessing for 2PL/3PL) for each benchmark item from the binary response matrix. Six model variants are fitted: 1PL, 2PL, and 3PL, each with joint MLE (JMLE) and item-marginal MMLE (EM + Gauss-Hermite quadrature). Model selection uses held-out AUC, AIC, and BIC.

## Scripts

| Script | Purpose |
|---|---|
| `irt.py` | Core IRT fitting script — fits all 6 variants, runs held-out AUC evaluation, exports capability and model-selection tables |
| `run_irt_modal.py` | Dispatches `irt.py` to a Modal cloud GPU |
| `irt_continuous.py` | IRT variant for continuous (non-binary) response scores |
| `run_irt_continuous_modal.py` | Dispatches `irt_continuous.py` to Modal |
| `irt_reduced_mmle.py` | Reduced MMLE variant with a simplified E-step |
| `run_irt_modal_reduced_mmle.py` | Dispatches `irt_reduced_mmle.py` to Modal |
| `plot_solver_judge_irt_scatter.py` | Generates solver-vs-judge ability scatter plots with SE bars for all four domains |
| `correlate_rankings.py` | Computes Spearman rank correlation between solver and judge ability rankings |

## Notebooks

| Notebook | Purpose |
|---|---|
| `irt.ipynb` | Interactive IRT exploration: load results, inspect fits, plot ability estimates |
| `irt_continuous.ipynb` | Interactive exploration for continuous IRT results |
| `view_irt_results.ipynb` | Quick viewer for IRT result CSVs |
| `view_irt_continuous_results.ipynb` | Quick viewer for continuous IRT results |
| `extract_safety_case_studies.ipynb` | Extracts case-study items from the safety IRT results |

## Ability tables (paper-ready)

| File | Domain |
|---|---|
| `IRT - MMLU.csv` | MMLU-Pro solver + JudgeBench judge |
| `IRT - Coding.csv` | LiveCodeBench solver + CodeJudgeBench judge |
| `IRT - Kudge.csv` | KUDGE challenge solver + KUDGE pairwise judge |
| `IRT - Safety.csv` | HarmBench safety solver + HarmMetric judge |

Each CSV contains side-by-side solver/judge ability and standard error columns, used for rank correlation and scatter plots.

## Usage

```bash
# Run IRT on Modal (GPU-accelerated, recommended)
modal run IRT/run_irt_modal.py --matrix solver          --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix judging         --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix safety          --seed 123 --heldout-repeats 5 --gpu H100
modal run IRT/run_irt_modal.py --matrix code_solver     --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix code_judge      --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix kudge_challenge --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix kudge_judge     --seed 123 --heldout-repeats 5 --gpu A10G

# Quick smoke test (CPU, 1 repeat)
python IRT/irt.py --matrix kudge_challenge --seed 123 --heldout-repeats 1 --device cpu

# Generate scatter plots from pre-computed ability tables
python IRT/plot_solver_judge_irt_scatter.py

# Compute rank correlations
python IRT/correlate_rankings.py "IRT/IRT - MMLU.csv"
python IRT/correlate_rankings.py "IRT/IRT - Coding.csv"
python IRT/correlate_rankings.py "IRT/IRT - Kudge.csv"
python IRT/correlate_rankings.py "IRT/IRT - Safety.csv"
```

## Outputs

- `results_modal/<matrix>/capability_scores.csv` — per-model ability estimates and SEs for each fitted model variant
- `results_modal/<matrix>/heldout_eval_summary.csv` — held-out AUC, log-likelihood, Brier score, ECE across random splits
- `results_modal/<matrix>/information_criteria.csv` — AIC/BIC per model variant
- `figures/` — scatter plots (PDF + PNG) for all four domains
- `paper_artifacts/` — curated paper-ready tables, charts, and case studies (see `paper_artifacts/README.md`)
