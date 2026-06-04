# benchmarks/kudge/

KUDGE Korean pairwise preference benchmark (Son et al., 2024), covering both challenge-style solving and LLM-as-judge evaluation.

- **Challenge task (solver)**: models answer Korean-language questions from the KUDGE challenge dataset (Korean-Easy and Korean-Hard splits). Correctness is determined against gold labels.
- **Judge task**: models act as pairwise preference judges, choosing the better of two candidate responses to a Korean instruction.

## Scripts

| Script | Purpose |
|---|---|
| `kudge.py` | Runs models on the KUDGE challenge (solver) via Modal |
| `kudge_pairwise.py` | Runs models as pairwise judges on KUDGE preference pairs via Modal |
| `create_challenge_response_matrix.py` | Aggregates per-model challenge results into a response matrix CSV |
| `create_judge_response_matrix.py` | Aggregates per-model judge results into a response matrix CSV |
| `run_overnight.py` | Convenience script to run multiple models sequentially overnight |
| `explore_kudge_data.ipynb` | Exploratory notebook for KUDGE dataset inspection |

## Data files

- `kudge_korean_hard_groundtruth_labels.json` — gold labels for the Korean-Hard split

## Usage

```bash
modal run benchmarks/kudge/kudge.py --model gpt-4o-mini
modal run benchmarks/kudge/kudge_pairwise.py --model gpt-4o-mini

python benchmarks/kudge/create_challenge_response_matrix.py
python benchmarks/kudge/create_judge_response_matrix.py
```

## Outputs

Per-model results are saved as `.npz` files under `results/kudge_challenge_easy_hard/` and `results/kudge_pairwise/`. The `create_*_response_matrix.py` scripts aggregate these into combined CSVs.

## Data source

KUDGE: `amphora/kudge-challenge` (HuggingFace, Korean-Easy and Korean-Hard splits)
