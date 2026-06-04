# Solver vs. Judge: Measuring LLM Capability with IRT and K-Factor Models

This project investigates whether a large language model that performs well as a _solver_ also performs well as a _judge_ (and vice versa), across four benchmark domains: **MMLU** (multiple-choice knowledge), **LiveCodeBench** (competitive programming), **Safety/HarmBench** (adversarial safety), and **KUDGE** (Korean pairwise preference judging). We measure model capability using two psychometric frameworks: **Item Response Theory (IRT)** and **logistic factor models (K-Factor)**, implemented via the `torch_measure` package (see below).

---

## Repository Structure

```
cs321m_project/
├── benchmarks/            # Benchmark runners, collected outputs, and response matrices
│   ├── llm_client.py      # Shared helper: routes queries to OpenAI, Anthropic, or HF Transformers
│   ├── code/              # LiveCodeBench solver + CodeJudgeBench judge
│   │   ├── solving_outputs/
│   │   ├── judging_outputs/
│   │   └── response_matrices/
│   ├── mmlu/              # MMLU-Pro solver + JudgeBench-MMLU judge
│   │   ├── solving_outputs/
│   │   ├── judging_outputs/
│   │   └── response_matrices/
│   ├── safety/            # HarmBench safety solver + judge outputs
│   │   ├── solver_outputs/
│   │   │   ├── attack_results/
│   │   │   └── final/
│   │   └── judge_outputs/
│   ├── kudge/             # KUDGE Korean preference benchmark
│   └── HarmMetric_Eval/   # HarmMetric judge evaluation pipeline and matrices
├── IRT/                   # IRT analysis: fitting, evaluation, figures, charts, and tables
│   ├── irt.py             # Core IRT fitting script (1PL/2PL/3PL, MLE + item-marginal MMLE)
│   ├── run_irt_modal.py   # Runs irt.py on Modal cloud GPUs
│   ├── irt.ipynb          # Interactive IRT exploration notebook
│   ├── plot_solver_judge_irt_scatter.py  # Generates solver-vs-judge scatter figures
│   ├── correlate_rankings.py             # Spearman rank correlation between solver/judge abilities
│   ├── IRT - *.csv        # Per-domain solver/judge ability tables (paper-ready)
│   ├── results_modal/     # IRT fit outputs from Modal runs
│   ├── results_continuous_modal/          # Continuous IRT fit outputs
│   ├── figures/           # Generated scatter plots (PDF + PNG)
│   └── charts_and_tables/ # Final tables, charts, data, and case studies for the paper
├── K-Factor/              # K-Factor (logistic FM) analysis: fitting, evaluation, charts, and tables
│   ├── kfactor.ipynb      # Main K-Factor notebook (binary response matrices)
│   ├── kfactor_continuous.ipynb          # K-Factor for continuous scores (HarmMetric)
│   ├── run_all_kfactor_notebooks.py      # Batch-executes kfactor.ipynb for all domains
│   ├── compare_*_solver_judge_difficulty.ipynb  # Per-domain solver-vs-judge difficulty comparisons
│   ├── results/           # K-Factor fit outputs
│   ├── results_continuous/ # Continuous K-Factor fit outputs
│   └── charts_and_tables/ # Final tables, figures, data, and case studies for the paper
├── HarmBench/             # Optional upstream checkout for regenerating HarmBench attacks (see setup)
├── HarmMetric_Eval/       # Optional upstream checkout for HarmMetric source-data inspection (see setup)
├── scripts/               # Utility scripts for generating case study notebooks/CSVs
│   ├── create_safety_case_study_notebook.py
│   └── export_safety_case_study_xlsx.py
├── docs/                  # Documentation source
├── src/                   # torch_measure package source (IRT, factor models, metrics, viz)
├── tests/                 # Unit tests
├── tutorials/             # Example notebooks
├── pyproject.toml         # Package definition and dependencies
├── requirements.txt       # Flexible dependency bounds for local setup
├── requirements-lock.txt  # Exact package versions from the reproduction environment
└── .env.example           # Template for local-only environment variables
```

---

## Environment Setup

### Prerequisites

- Python 3.10–3.12
- A CUDA-capable GPU is recommended for IRT fitting; CPU works but is slower
- [Modal](https://modal.com) account (free tier sufficient) for cloud benchmark collection
- API keys for OpenAI and/or Anthropic (for API-based model evaluations)

### 1. Install the package and dependencies

```bash
# Clone the repo
git clone <repo-url>
cd cs321m_project

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install torch_measure and all dependencies
pip install -e .

# Install additional runtime dependencies
pip install modal tqdm jupyter nbconvert ipykernel
```

For exact reproduction of the checked-in analysis environment, install from the
lock file instead:

```bash
pip install -r requirements-lock.txt
pip install -e .
```

The regular `requirements.txt` is a flexible dependency list for development
and fresh setups; `requirements-lock.txt` records exact installed package
versions.

### 2. Add optional upstream checkouts for full safety regeneration

The checked-in response matrices are enough to reproduce the IRT/K-Factor
analysis. If you want to regenerate HarmBench attacks from scratch, fork or
clone the upstream HarmBench repository so it appears at the repo root as
`cs321m_project/HarmBench/`; the safety attack-generation Modal scripts mount
that directory into the cloud container.

```bash
# From cs321m_project/
git clone https://github.com/centerforaisafety/HarmBench.git HarmBench
```

The adapted HarmMetric pipeline used by this project is tracked under
`benchmarks/HarmMetric_Eval/`. A separate top-level `HarmMetric_Eval/` checkout
is only needed if you want to inspect or rerun upstream HarmMetric source-data
processing notebooks outside the tracked benchmark pipeline.

```bash
# Optional; only for upstream HarmMetric source-data inspection
git clone https://github.com/Qusgo/HarmMetric_Eval.git HarmMetric_Eval
```

After these optional checkouts, the relevant external-resource layout is:

```text
cs321m_project/
├── HarmBench/                 # Upstream HarmBench checkout
├── HarmMetric_Eval/           # Optional upstream HarmMetric checkout
└── benchmarks/HarmMetric_Eval/ # Tracked adapted HarmMetric evaluation pipeline
```

### 3. Configure credentials

Local scripts that call Anthropic directly read `ANTHROPIC_API_KEY` from `.env`.
Copy `.env.example` to `.env` only if you plan to run those local scripts:

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY
```

OpenAI, Anthropic, and HuggingFace credentials used by Modal benchmark jobs are
configured as Modal secrets, not in `.env`. Register them once before running
Modal collection scripts:

```bash
modal secret create openai-secret    OPENAI_API_KEY=sk-...
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
modal secret create hf-secret        HF_TOKEN=hf_...   # for gated HF models
```

### 4. Register a Jupyter kernel (for K-Factor notebooks)

```bash
python -m ipykernel install --user --name cs321m-project --display-name "cs321m-project"
```

---

## Reproducing Results

The pipeline has five stages. Pre-computed response matrices are already
checked in under each benchmark's matrix directory, so you can skip to Stage 3
to reproduce IRT and K-Factor results directly. See `benchmarks/README.md` for
the exact paths, including nested KUDGE and safety matrix directories.

### Stage 1: Collect benchmark responses (cloud; skip if using pre-computed matrices)

Each script dispatches model queries in parallel via Modal and writes results to `benchmarks/<domain>/`.

**Coding — LiveCodeBench (solver):**

```bash
modal run benchmarks/code/livecodebench.py --model gpt-4o-mini
modal run benchmarks/code/livecodebench.py --model claude-haiku-4-5-20251001
```

**Coding — CodeJudgeBench (judge):**

```bash
modal run benchmarks/code/codejudgebench_pairwise.py --model gpt-4o-mini
```

**MMLU (solver + judge):**

```bash
modal run benchmarks/mmlu/run_qwen35_solving_modal.py
modal run benchmarks/mmlu/run_qwen35_judging_modal.py
```

**Safety (solver):**

```bash
modal run benchmarks/safety/run_hf_safety.py
modal run benchmarks/safety/run_claude_safety.py
```

**KUDGE (solver + judge):**

```bash
modal run benchmarks/kudge/kudge.py
modal run benchmarks/kudge/kudge_pairwise.py
```

### Stage 2: Build response matrices (skip if using pre-computed matrices)

```bash
python benchmarks/code/create_livecodebench_response_matrix.py
python benchmarks/code/create_codejudgebench_response_matrix.py
python benchmarks/mmlu/create_solver_response_matrix.py
python benchmarks/mmlu/create_judging_response_matrix.py
python benchmarks/kudge/create_challenge_response_matrix.py
python benchmarks/kudge/create_judge_response_matrix.py

# Safety solver: combine per-attack results into a common JSON
python benchmarks/safety/combine_attack_results_for_final_solver.py

# Safety judge: build the HarmMetric response matrix
python benchmarks/HarmMetric_Eval/create_harmmetric_response_matrix.py
```

> **Note:** The safety solver matrix requires prior attack generation (Stage 1). The pre-computed matrix in `benchmarks/safety/solver_outputs/final/response_matrices/` and the HarmMetric matrix in `benchmarks/HarmMetric_Eval/response_matrices/` can be used directly.

### Stage 3: Run IRT analysis

Fits 1PL, 2PL, and 3PL models (joint MLE + item-marginal MMLE) with held-out AUC across 5 random splits.

**On Modal (GPU-accelerated, recommended):**

```bash
# <matrix> is one of: solver, judging, safety, kudge_challenge, kudge_judge, code_solver, code_judge
modal run IRT/run_irt_modal.py --matrix solver        --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix judging       --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix safety        --seed 123 --heldout-repeats 5 --gpu H100
modal run IRT/run_irt_modal.py --matrix code_solver   --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix code_judge    --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix kudge_challenge --seed 123 --heldout-repeats 5 --gpu A10G
modal run IRT/run_irt_modal.py --matrix kudge_judge   --seed 123 --heldout-repeats 5 --gpu A10G
```

Results land in `IRT/results_modal/<matrix>/`.

**Locally (CPU fallback):**

```bash
python IRT/irt.py --matrix solver --seed 123 --heldout-repeats 5
```

### Stage 4: Run K-Factor analysis

Fits LogisticFM with K=1 and K=2; paper-facing K selection uses lower in-sample reconstruction loss.

**Batch (all domains):**

```bash
cd K-Factor
python run_all_kfactor_notebooks.py
```

**Single domain interactively:** Open `K-Factor/kfactor.ipynb`, set `KFACTOR_MATRIX` to one of:
`mmlu_solver`, `mmlu_judging`, `safety_all_attacks`, `code_solver`, `code_judge`, `kudge_challenge`, `kudge_judge`

### Stage 5: Generate figures and paper artifacts

```bash
# IRT solver-vs-judge scatter plots → IRT/figures/
python IRT/plot_solver_judge_irt_scatter.py

# Rank correlation tables
python IRT/correlate_rankings.py "IRT/IRT - MMLU.csv"
python IRT/correlate_rankings.py "IRT/IRT - Coding.csv"
python IRT/correlate_rankings.py "IRT/IRT - Kudge.csv"
python IRT/correlate_rankings.py "IRT/IRT - Safety.csv"

# K-Factor per-domain comparison notebooks → K-Factor/charts_and_tables/
jupyter nbconvert --to notebook --execute K-Factor/compare_code_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_mmlu_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_safety_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_kudge_solver_judge_difficulty.ipynb

# Safety case study (metadata only — no raw harmful prompts)
python scripts/create_safety_case_study_notebook.py
```

---

## Which Scripts Produce Which Results

| Paper result                                | Script / notebook                                  | Output location                                                  |
| ------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------- |
| IRT model selection (AIC/BIC/heldout AUC)   | `IRT/irt.py`                                       | `IRT/results_modal/<matrix>/heldout_eval_summary.csv`            |
| Model ability rankings                      | `IRT/irt.py`                                       | `IRT/results_modal/<matrix>/capability_scores.csv`               |
| Solver vs. judge scatter (IRT)              | `IRT/plot_solver_judge_irt_scatter.py`             | `IRT/figures/`                                                   |
| K-Factor model selection                    | `K-Factor/kfactor.ipynb`                           | `K-Factor/charts_and_tables/*_kfactor_fit_summary_insample*.csv` |
| Solver vs. judge item difficulty (K-Factor) | `K-Factor/compare_*_solver_judge_difficulty.ipynb` | `K-Factor/charts_and_tables/`                                    |
| Safety case studies                         | `scripts/create_safety_case_study_notebook.py`     | `IRT/charts_and_tables/safety/case_studies/`                     |
| Rank correlation tables                     | `IRT/correlate_rankings.py`                        | stdout / optional `--output` CSV                                 |

Pre-computed final outputs used in the paper are in `IRT/charts_and_tables/` and `K-Factor/charts_and_tables/`.

---

## Expected Runtime and Computational Requirements

| Stage                                          | Hardware                              | Approximate time |
| ---------------------------------------------- | ------------------------------------- | ---------------- |
| Benchmark collection (all models, all domains) | Modal cloud (≤20 parallel containers) | 4–8 hours total  |
| IRT fitting — one matrix, 5 held-out repeats   | A10G GPU                              | 10–30 min        |
| IRT fitting — safety matrix                    | H100 GPU                              | 30–60 min        |
| K-Factor — all domains                         | CPU                                   | 30–90 min        |
| Figure generation                              | CPU                                   | < 5 min          |

**Quick smoke-test** (no GPU needed):

```bash
python IRT/irt.py --matrix kudge_challenge --seed 123 --heldout-repeats 1 --device cpu
```

---

## Datasets

All benchmark data loads on-demand from public HuggingFace datasets — no manual downloads required.

| Benchmark                  | HuggingFace dataset                                               |
| -------------------------- | ----------------------------------------------------------------- |
| MMLU-Pro                   | `TIGER-Lab/MMLU-Pro`                                              |
| LiveCodeBench v6           | `livecodebench/code_generation_lite` (`release_v6`)               |
| KUDGE                      | `amphora/kudge-challenge` (Korean-Easy and Korean-Hard splits)    |
| Safety / HarmBench attacks | Collected via `benchmarks/safety/` using public HarmBench prompts |

Pre-computed response matrices are checked in under the benchmark matrix
directories listed in `benchmarks/README.md`, so Stages 1–2 can be skipped
entirely for reproducing the analysis.

---

## Reproducibility Notes

- All stochastic components use fixed seeds. The default is `--seed 123` for IRT and `seed=123` in K-Factor notebooks. Seeds are applied to Python `random`, NumPy, and PyTorch via `IRT/irt.py:set_seed`.
- `torch.manual_seed` is called before every individual model fit in K-Factor.
- `requirements-lock.txt` records exact package versions from the reproduction environment; `requirements.txt` keeps flexible bounds for development installs.
- Modal container images pin their own versions inside each script's `pip_install(...)` call for cloud reproducibility.

---

## Code Attribution

- **`src/`** — this repository is a fork of the [`torch_measure`](https://github.com/anthropics/torch_measure) package (MIT License). The core IRT and factor model implementations in `src/` are from the upstream library; this project adds benchmark collection, response matrix construction, and analysis notebooks on top of it.
- **`benchmarks/HarmMetric_Eval/`** — adapted from the [HarmMetric Eval](https://huggingface.co/datasets/anonymous-review-anonymous/HarmMetric_Eval) repository (_HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment_). We added Modal-based cloud collection scripts (`modal_claude_harmmetric.py`, `modal_qwen35_harmmetric.py`) and the response matrix construction script (`create_harmmetric_response_matrix.py`).
- All other code in `benchmarks/`, `IRT/`, `K-Factor/`, and `scripts/` is original work for this project.

---

---

# torch_measure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-5865F2.svg)](https://discord.gg/F6xbEwvvhb)

**PyTorch-native toolkit for predictive evaluation of AI systems.**

Benchmark scores increasingly gate deployment decisions but rarely predict how a model will behave in production. `torch_measure` treats evaluation itself as a predictive modeling problem: latent-variable models infer a system's capability directly from sparse benchmark observations and predict its performance on unseen tasks. Built on PyTorch, with GPU-accelerated IRT, factor models, amortized inference, adaptive testing, and tabular baselines.

## Installation

With **pip**:

```bash
pip install torch_measure
```

With **[uv](https://docs.astral.sh/uv/)** (faster; drop-in replacement for pip):

```bash
uv pip install torch_measure        # into the active environment
uv add torch_measure                # into a uv-managed project
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details, or drop by our [Discord](https://discord.gg/F6xbEwvvhb) to chat.
