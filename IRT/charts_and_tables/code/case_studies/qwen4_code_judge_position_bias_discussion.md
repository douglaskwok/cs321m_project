# Qwen3.5-4B Code-Judge Case Study

## Main finding

Qwen3.5-4B is ranked first by the selected Coding-Judge 2PL JMLE ability estimate, but this does not appear to mean that it is the best code judge in an ordinary accuracy sense. Its raw CodeJudgeBench accuracy is only 0.524, below Qwen3.5-27B, Claude Haiku 4.5, Qwen3.5-9B, and the Mistral models.

The striking pattern is response-position bias. In the final one-order CodeJudgeBench run, Qwen3.5-4B predicts `B` on 92.9% of items. Because the sampled backward-order items have `gold = B`, this yields:

- Accuracy on forward-order items: 0.103
- Accuracy on backward-order items: 0.961
- Accuracy when gold is A: 0.103
- Accuracy when gold is B: 0.961

This suggests that Qwen3.5-4B's high 2PL ability is likely driven by a strong tendency to choose `B`, combined with the particular one-order sample and the 2PL model's weighting of discriminative items.

## Candidate case-study items

There are 39 items where Qwen3.5-4B is correct while Claude Haiku 4.5, Qwen3.5-27B, and Qwen3.5-9B are all incorrect. All 39 are backward-order items with `gold = B`. There are also 15 items where Qwen3.5-4B is the only correct judge among all nine models; all 15 are also backward-order items with `gold = B`.

Examples:

- `gemini_2.5_pro:3658:48:bwd`
- `gemini_2.5_pro:abc306_e:77:bwd`
- `gemini_2.5_pro:abc345_d:100:bwd`
- `gemini_2.5_pro:abc380_d:144:bwd`
- `qwen3_235b:3308:27:bwd`
- `qwen3_235b:3406:33:bwd`
- `qwen3_235b:3464:41:bwd`
- `qwen3_235b:3482:44:bwd`
- `qwen3_235b:3587:60:bwd`
- `qwen3_235b:3604:63:bwd`
- `qwen3_235b:abc398_f:207:bwd`
- `claude_4_opus:3229:20:bwd`

These are better interpreted as position-bias diagnostics than as evidence that Qwen3.5-4B has superior code-judging ability.

## Suggested write-up

The Coding domain provides a cautionary example for interpreting IRT ability estimates on judge benchmarks. Although Qwen3.5-4B receives the highest 2PL JMLE judge ability estimate, its raw judging accuracy is only 0.524. A closer inspection shows that this estimate is confounded by response-position bias: Qwen3.5-4B predicts `B` on 92.9% of CodeJudgeBench items, achieving 0.961 accuracy on backward-order items where `B` is the correct answer, but only 0.103 accuracy on forward-order items where `A` is correct. The apparent advantage is therefore concentrated in items where the sampled ordering aligns with the model's strong preference for `B`. This highlights a limitation of fitting IRT to one-order pairwise judgments: when swapped-order evaluations are not averaged, latent judge ability can absorb systematic position bias.

## Files

- `qwen4_code_judge_position_bias_summary.csv`: model-level prediction and accuracy-by-order summary.
- `qwen4_code_judge_top3_wrong_cases.csv`: all items where Qwen3.5-4B is correct while Claude Haiku 4.5, Qwen3.5-27B, and Qwen3.5-9B are wrong.
- `qwen4_code_judge_top_cases_with_verdicts.csv`: top subset with final verdict strings.
- `qwen4_code_judge_position_bias_stats.json`: compact machine-readable summary.
