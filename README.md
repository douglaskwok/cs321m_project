# Solver vs. Judge: Measuring LLM Capability with IRT and K-Factor Models

This project investigates whether a large language model that performs well as a *solver* also performs well as a *judge* (and vice versa), across four benchmark domains: **MMLU** (multiple-choice knowledge), **LiveCodeBench** (competitive programming), **Safety/HarmBench** (adversarial safety), and **KUDGE** (Korean pairwise preference judging). We measure model capability using two psychometric frameworks: **Item Response Theory (IRT)** and **logistic factor models (K-Factor)**, implemented via the `torch_measure` package (see below).

---

## Repository Structure

```
cs321m_project/
├── benchmarks/            # Scripts that run models on benchmarks and build response matrices
│   ├── llm_client.py      # Shared helper: routes queries to OpenAI, Anthropic, or HF Transformers
│   ├── code/              # LiveCodeBench (coding solver) + CodeJudgeBench (coding judge)
│   ├── mmlu/              # MMLU-Pro (solver + judge)
│   ├── safety/            # HarmBench safety (solver) + HarmMetric (judge)
│   ├── kudge/             # KUDGE Korean preference benchmark (solver + judge)
│   └── HarmMetric_Eval/   # HarmMetric judge evaluation pipeline
├── IRT/                   # IRT analysis: fitting, evaluation, figures, paper artifacts
│   ├── irt.py             # Core IRT fitting script (1PL/2PL/3PL, MLE + item-marginal MMLE)
│   ├── run_irt_modal.py   # Runs irt.py on Modal cloud GPUs
│   ├── irt.ipynb          # Interactive IRT exploration notebook
│   ├── plot_solver_judge_irt_scatter.py  # Generates solver-vs-judge scatter figures
│   ├── correlate_rankings.py             # Spearman rank correlation between solver/judge abilities
│   ├── IRT - *.csv        # Per-domain solver/judge ability tables (paper-ready)
│   ├── results_modal/     # IRT fit outputs from Modal runs
│   ├── figures/           # Generated scatter plots (PDF + PNG)
│   └── paper_artifacts/   # Final tables, charts, and case studies for the paper
├── K-Factor/              # K-Factor (logistic FM) analysis: fitting, evaluation, paper artifacts
│   ├── kfactor.ipynb      # Main K-Factor notebook (binary response matrices)
│   ├── kfactor_continuous.ipynb          # K-Factor for continuous scores (HarmMetric)
│   ├── run_all_kfactor_notebooks.py      # Batch-executes kfactor.ipynb for all domains
│   ├── compare_*_solver_judge_difficulty.ipynb  # Per-domain solver-vs-judge difficulty comparisons
│   ├── results/           # K-Factor fit outputs
│   └── paper_artifacts/   # Final tables, figures, and case studies for the paper
├── scripts/               # Utility scripts for generating case study notebooks/CSVs
│   ├── create_safety_case_study_notebook.py
│   └── export_safety_case_study_xlsx.py
├── src/                   # torch_measure package source (IRT, factor models, metrics, viz)
├── trash/                 # Non-essential files (logs, scratch outputs)
├── pyproject.toml         # Package definition and dependencies
├── requirements.txt       # Pinned environment for reproducing results
└── .env.example           # Template for API keys
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

Or install from the pinned requirements file for exact reproducibility:

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

For Modal-based benchmark runs, register the secrets once:

```bash
modal secret create openai-secret    OPENAI_API_KEY=sk-...
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
modal secret create hf-secret        HF_TOKEN=hf_...   # for gated HF models
```

### 3. Register a Jupyter kernel (for K-Factor notebooks)

```bash
python -m ipykernel install --user --name cs321m-project --display-name "cs321m-project"
```

---

## Reproducing Results

The pipeline has five stages. Pre-computed response matrices are already checked in under `benchmarks/*/response_matrices/`, so you can skip to Stage 3 to reproduce IRT and K-Factor results directly.

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
```

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

Fits LogisticFM with K=1 and K=2 and evaluates held-out AUC.

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

# K-Factor per-domain comparison notebooks → K-Factor/paper_artifacts/
jupyter nbconvert --to notebook --execute K-Factor/compare_code_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_mmlu_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_safety_solver_judge_difficulty.ipynb
jupyter nbconvert --to notebook --execute K-Factor/compare_kudge_solver_judge_difficulty.ipynb

# Safety case study (metadata only — no raw harmful prompts)
python scripts/create_safety_case_study_notebook.py
```

---

## Which Scripts Produce Which Results

| Paper result | Script / notebook | Output location |
|---|---|---|
| IRT model selection (AIC/BIC/heldout AUC) | `IRT/irt.py` | `IRT/results_modal/<matrix>/heldout_eval_summary.csv` |
| Model ability rankings | `IRT/irt.py` | `IRT/results_modal/<matrix>/capability_scores.csv` |
| Solver vs. judge scatter (IRT) | `IRT/plot_solver_judge_irt_scatter.py` | `IRT/figures/` |
| K-Factor model selection | `K-Factor/kfactor.ipynb` | `K-Factor/results/<domain>/*_kfactor_fit_summary.csv` |
| Solver vs. judge item difficulty (K-Factor) | `K-Factor/compare_*_solver_judge_difficulty.ipynb` | `K-Factor/paper_artifacts/` |
| Safety case studies | `scripts/create_safety_case_study_notebook.py` | `IRT/paper_artifacts/safety/case_studies/` |
| Rank correlation tables | `IRT/correlate_rankings.py` | stdout / optional `--output` CSV |

Pre-computed final outputs used in the paper are in `IRT/paper_artifacts/` and `K-Factor/paper_artifacts/`.

---

## Expected Runtime and Computational Requirements

| Stage | Hardware | Approximate time |
|---|---|---|
| Benchmark collection (all models, all domains) | Modal cloud (≤20 parallel containers) | 4–8 hours total |
| IRT fitting — one matrix, 5 held-out repeats | A10G GPU | 10–30 min |
| IRT fitting — safety matrix | H100 GPU | 30–60 min |
| K-Factor — all domains | CPU | 30–90 min |
| Figure generation | CPU | < 5 min |

**Quick smoke-test** (no GPU needed):
```bash
python IRT/irt.py --matrix kudge_challenge --seed 123 --heldout-repeats 1 --device cpu
```

---

## Datasets

All benchmark data loads on-demand from public HuggingFace datasets — no manual downloads required.

| Benchmark | HuggingFace dataset |
|---|---|
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` |
| LiveCodeBench v6 | `livecodebench/code_generation_lite` (`release_v6`) |
| KUDGE | `amphora/kudge-challenge` (Korean-Easy and Korean-Hard splits) |
| Safety / HarmBench attacks | Collected via `benchmarks/safety/` using public HarmBench prompts |

Pre-computed response matrices are checked in under `benchmarks/*/response_matrices/`, so Stages 1–2 can be skipped entirely for reproducing the analysis.

---

## Reproducibility Notes

- All stochastic components use fixed seeds. The default is `--seed 123` for IRT and `seed=123` in K-Factor notebooks. Seeds are applied to Python `random`, NumPy, and PyTorch via `IRT/irt.py:set_seed`.
- `torch.manual_seed` is called before every individual model fit in K-Factor.
- `requirements.txt` pins the package versions used during the final paper runs.
- Modal container images pin their own versions inside each script's `pip_install(...)` call for cloud reproducibility.

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
