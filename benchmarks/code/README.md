# benchmarks/code/

LiveCodeBench coding solver and CodeJudgeBench pairwise code judge benchmarks.

## Scripts

| Script | Purpose |
|---|---|
| `livecodebench.py` | Runs a model on LiveCodeBench v6; evaluates pass@1 on competitive programming problems |
| `codejudgebench_pairwise.py` | Runs a model as a pairwise judge on the CodeJudgeBench codegen subset |
| `codejudgebench_pairwise_faithful.py` | Variant of pairwise judge with faithfulness-focused prompt |
| `select_codegen_subset.py` | Selects a stratified 676-pair subset from CodeJudgeBench for judging |
| `create_livecodebench_response_matrix.py` | Aggregates per-model `.npz` results into a combined response matrix CSV |
| `create_codejudgebench_response_matrix.py` | Aggregates per-model judge results into a combined response matrix CSV |
| `run_codejudgebench_verdict_models.py` | Runs verdict models on CodeJudgeBench solutions |
| `judgebench_investigation.py` | Exploratory analysis of CodeJudgeBench data |
| `pairwise.py` | General pairwise utilities shared across judge scripts |

## Usage

```bash
# Solver: run one model on LiveCodeBench
modal run benchmarks/code/livecodebench.py --model gpt-4o-mini
modal run benchmarks/code/livecodebench.py --model claude-haiku-4-5-20251001

# Judge: first build the pair subset, then run judge models
python benchmarks/code/select_codegen_subset.py
modal run benchmarks/code/codejudgebench_pairwise.py --model gpt-4o-mini

# Build response matrices from collected results
python benchmarks/code/create_livecodebench_response_matrix.py
python benchmarks/code/create_codejudgebench_response_matrix.py
```

## Outputs

- `response_matrices/livecodebench_response_matrix.csv` — binary solver matrix (models × problems)
- `response_matrices/codejudgebench_pairwise_response_matrix.csv` — binary judge matrix (models × item pairs)
- `response_matrices/*_item_metadata.csv` — item difficulty, platform, etc.
- `response_matrices/*_subject_metadata.csv` — model names and slugs
- `codegen_selected_pairs.json` — the 676-pair stratified subset used for judging

## Data source

- LiveCodeBench v6: `livecodebench/code_generation_lite` (HuggingFace, `release_v6`)
- CodeJudgeBench: `mattymchen/codejudgebench` (HuggingFace)
