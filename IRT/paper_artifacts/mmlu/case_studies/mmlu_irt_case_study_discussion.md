# MMLU IRT Case Study: Strong Judging Despite Moderate Solving

This note focuses on the MMLU solver-vs-judge IRT comparison. It uses the exported IRT ability table and the solver-vs-judge scatter data, not safety data.

## Main Pattern

Qwen3.5 27B is the clearest outlier. It ranks only third on solver ability but first on judge ability:

- Solver ability: `-0.432`, rank `3`
- Judge ability: `1.520`, rank `1`
- Judge residual relative to a simple solver-to-judge linear trend: `+0.762`

This is the largest positive judge residual in the MMLU table. In other words, Qwen3.5 27B is much better at judging than its solver-side position would predict.

## Comparisons To Similar Or Stronger Solvers

The most informative comparison is against Claude Haiku 4.5. Haiku is substantially stronger as a solver:

- Qwen3.5 27B solver ability: `-0.432`
- Claude Haiku 4.5 solver ability: `0.647`
- Difference: `-1.079`

But Qwen3.5 27B is much stronger as a judge:

- Qwen3.5 27B judge ability: `1.520`
- Claude Haiku 4.5 judge ability: `1.093`
- Difference: `+0.427`

The judge-side difference is large relative to the combined standard error. This makes the pattern hard to explain as noise in the fitted abilities.

Against Claude Sonnet 4.6, Qwen3.5 27B is far weaker as a solver but essentially tied as a judge:

- Solver ability difference: `-2.603`
- Judge ability difference: `+0.011`

So the case is not merely that Qwen3.5 27B is a generally stronger model. It is specifically competitive on the judgment task.

## Comparisons Within The Qwen Family

The Qwen family comparison suggests a threshold effect. Qwen3.5 9B and Qwen3.5 4B have solver abilities close to Qwen3.5 27B:

- Qwen3.5 27B solver ability: `-0.432`
- Qwen3.5 9B solver ability: `-0.576`
- Qwen3.5 4B solver ability: `-0.621`

But the judge abilities are far apart:

- Qwen3.5 27B judge ability: `1.520`
- Qwen3.5 9B judge ability: `0.528`
- Qwen3.5 4B judge ability: `0.525`

This suggests that the judging setup may reward capabilities that only appear strongly in the larger Qwen model: answer comparison, consistency checking, or recognizing a correct rationale when the model does not have to produce the full solution from scratch.

## Secondary Pattern: Mistral 8B

Mistral 8B is another smaller example of judging outperforming solving:

- Solver ability: `-1.092`, rank `7`
- Judge ability: `0.694`, rank `4`
- Judge residual relative to the solver-to-judge trend: `+0.158`

This is much weaker than the Qwen3.5 27B effect, but it points in the same direction: judge ability is not just a monotone copy of solver ability.

## Interpretation

The MMLU judging task may be less demanding than solving in one specific way: the judge sees candidate answers and can evaluate correctness rather than generate the answer independently. This can let models with moderate solver ability perform well as judges if they are good at verification, comparison, and detecting plausible reasoning.

Qwen3.5 27B is the strongest example of this separation. It is not the best solver, but it behaves like a top-tier judge. This supports the paper's broader claim that solving and judging are related but not identical capabilities.

## Supporting Files

- `mmlu_irt_model_case_study_residuals.csv`: model-level solver/judge abilities, ranks, residuals, and ability gaps.
- `mmlu_irt_qwen27b_peer_comparisons.csv`: pairwise comparisons between Qwen3.5 27B and nearby or relevant peer models.
