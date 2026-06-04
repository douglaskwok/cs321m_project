# benchmarks/

Response data collection and response matrix construction for four benchmark domains.

## Shared utilities

- `llm_client.py` — routes queries to OpenAI, Anthropic, or a locally-loaded HuggingFace model. Used by all benchmark scripts.

## Subdirectories

| Folder | Domain | Task |
|---|---|---|
| `code/` | LiveCodeBench (solver) + CodeJudgeBench (judge) | Competitive programming |
| `mmlu/` | MMLU-Pro (solver) + JudgeBench-MMLU (judge) | Multiple-choice knowledge |
| `safety/` | HarmBench adversarial attacks (solver) | Safety / jailbreak resistance |
| `kudge/` | KUDGE Korean preference benchmark (challenge + judge) | Pairwise preference |
| `HarmMetric_Eval/` | HarmMetric judge evaluation pipeline | Safety judgment |

## Pre-computed response matrices

Pre-built response matrix CSVs are checked in for each domain (rows = models,
columns = benchmark items). Most live in a direct `response_matrices/` folder,
while KUDGE and safety keep matrices under their output directories:

- `code/response_matrices/`
- `mmlu/response_matrices/`
- `kudge/solver_outputs/response_matrices/`
- `kudge/judging_outputs/response_matrices/`
- `safety/solver_outputs/final/response_matrices/`
- `safety/judge_outputs/response_matrices/`
- `HarmMetric_Eval/response_matrices/`

You can skip data collection and proceed directly to IRT or K-Factor analysis
using these files.

## Running data collection

All collection scripts dispatch work to Modal cloud containers. These jobs read
API credentials from Modal secrets rather than from the local `.env` file. Run
this one-time secret setup before launching Modal benchmark collection:

```bash
modal secret create openai-secret    OPENAI_API_KEY=sk-...
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
modal secret create hf-secret        HF_TOKEN=hf_...
```

See each subdirectory's README for domain-specific instructions.
