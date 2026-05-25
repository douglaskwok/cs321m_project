# HarmBench HF Overlap Summary

Compared local official HarmBench CSVs against Hugging Face datasets:

- `walledai/HarmBench`: `standard`, `contextual`, `copyright`
- `AlignmentResearch/HarmBench`: `default`, `pos` (`neg` has no train rows)

## Overall

| official_source | hf_source | official_rows | official_unique_normalized_prompts | hf_rows | hf_unique_normalized_prompts | exact_prompt_overlap | normalized_prompt_overlap | official_coverage_by_normalized_prompt | hf_coverage_by_normalized_prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_text_all | walledai/HarmBench | 400 | 393 | 400 | 393 | 393 | 393 | 1.0 | 1.0 |
| official_text_all | AlignmentResearch/HarmBench | 400 | 393 | 400 | 200 | 200 | 200 | 0.5089 | 1.0 |
| official_multimodal_all | walledai/HarmBench | 110 | 66 | 400 | 393 | 0 | 0 | 0.0 | 0.0 |
| official_multimodal_all | AlignmentResearch/HarmBench | 110 | 66 | 400 | 200 | 0 | 0 | 0.0 | 0.0 |
| official_text_plus_multimodal | walledai/HarmBench | 510 | 459 | 400 | 393 | 393 | 393 | 0.8562 | 1.0 |
| official_text_plus_multimodal | AlignmentResearch/HarmBench | 510 | 459 | 400 | 200 | 200 | 200 | 0.4357 | 1.0 |

## By HF Config

| hf_source | hf_config | hf_rows | overlap_with_official_text_all | overlap_with_official_multimodal_all | overlap_with_official_text_plus_multimodal |
| --- | --- | --- | --- | --- | --- |
| walledai/HarmBench | contextual | 100 | 93 | 0 | 93 |
| walledai/HarmBench | copyright | 100 | 100 | 0 | 100 |
| walledai/HarmBench | standard | 200 | 200 | 0 | 200 |
| AlignmentResearch/HarmBench | default | 200 | 200 | 0 | 200 |
| AlignmentResearch/HarmBench | pos | 200 | 200 | 0 | 200 |

Output files:

- `benchmarks/safety/harmbench_hf_overlap_summary.csv`
- `benchmarks/safety/harmbench_hf_overlap_by_config.csv`
- `benchmarks/safety/harmbench_hf_overlap_details.csv`
