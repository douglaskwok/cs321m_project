# benchmarks/safety/

HarmBench adversarial safety solver benchmark and response matrix pipeline.

Models are tested as *solvers* (targets of adversarial jailbreak attacks). A model "succeeds" when it refuses to comply with a harmful prompt — the HarmBench classifier determines compliance. Judge evaluation for safety is handled by the separate `HarmMetric_Eval/` pipeline.

## Scripts

| Script | Purpose |
|---|---|
| `run_hf_safety.py` | Runs HuggingFace Transformers models on safety prompts via Modal |
| `run_claude_safety.py` | Runs Anthropic Claude models on safety prompts via Modal |
| `run_harmbench_attack_generation.py` | Generates adversarial attacks using HarmBench attack methods |
| `run_harmbench_attack_generation_heavy.py` | Heavier attack generation (more methods / higher budget) |
| `run_harmbench_classifier.py` | Runs the HarmBench classifier to label model responses as harmful/refused |
| `run_harmbench_vllm_image.py` | Runs vLLM image for the attack/classify pipeline |
| `generate_test_cases_for_safety.py` | Generates test cases from behavior specifications |
| `create_harmbench_attack_cases.py` | Assembles attack cases from raw attack outputs |
| `merge_harmbench_attack_cases.py` | Merges attack cases across attack methods |
| `sample_attack_cases_by_dimension.py` | Samples attack cases stratified by harm dimension |
| `sample_harmbench_attack_cases.py` | Samples a fixed-size subset of attack cases |
| `combine_attack_results_for_final_solver.py` | Combines results across attack methods into the final solver matrix |
| `create_harmmetric_attack_method_prompts.py` | Creates HarmMetric-formatted prompts for the judge pipeline |
| `create_harmmetric_official_attack_prompts.py` | Creates prompts from official HarmBench attack scenarios |

## Data files

- `safety_solver.json` — curated safety behaviors used as solver prompts
- `safety_solver_behaviors.csv` — behavior metadata (harm category, attack method, etc.)
- `safety_solver_*_sampled.json` — per-method sampled attack cases (AutoDAN, GCG, PAP, PAIR, human jailbreaks)
- `attack_cases/` — raw attack case files organized by method
- `attack_cases_sampled_by_dimension/` — stratified subsets by harm dimension
- `attack_method_prompts/` — formatted HarmMetric input prompts
- `data_processing/` — intermediate processing scripts and outputs
- `final_solver/` — final combined results and response matrices used in the paper

## Usage

```bash
# Run models on safety prompts
modal run benchmarks/safety/run_hf_safety.py --model qwen3.5-0.8b
modal run benchmarks/safety/run_claude_safety.py --model claude-haiku-4-5-20251001

# Build the final solver response matrix from collected results
python benchmarks/safety/combine_attack_results_for_final_solver.py
```

## Outputs

- `final_solver/response_matrices/safety_all_attacks_response_matrix.csv` — binary solver matrix (models × attack instances)
- `final_solver/response_matrices/safety_all_attacks_item_metadata.csv` — item metadata (attack method, harm category, etc.)

## Data source

HarmBench: adversarial attacks sourced from public HarmBench behaviors and attack implementations.
