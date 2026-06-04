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

Each subdirectory contains a `response_matrices/` folder with pre-built CSV files (rows = models, columns = benchmark items, values = binary correctness). You can skip data collection and proceed directly to IRT or K-Factor analysis using these files.

## Running data collection

All collection scripts dispatch work to Modal cloud containers. One-time secret setup:

```bash
modal secret create openai-secret    OPENAI_API_KEY=sk-...
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...
modal secret create hf-secret        HF_TOKEN=hf_...
```

See each subdirectory's README for domain-specific instructions.
