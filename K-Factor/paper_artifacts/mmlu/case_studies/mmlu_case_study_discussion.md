# MMLU Case Study Discussion

This note summarizes qualitative patterns from the MMLU K-factor case-study extraction. The goal is to interpret the solver-vs-judge difficulty comparison, with special attention to the solver difficulty bin 5 anomaly: bin 5 is moderately difficult to solve, but unusually easy to judge.

The extracted cases use K=2 K-factor item difficulties. "Hard" items are in the top quartile of centered difficulty percentile, and "easy" items are in the bottom quartile. The case-study selection keeps the most extreme items in each region of the solver-vs-judge difficulty plane.

## Summary

The MMLU case studies support the same broad conclusion as Kudge: solver difficulty and judge difficulty are related, but they are not the same construct.

| Category | Selected | Mean solver percentile | Mean judge percentile | Mean solver score | Mean judge score |
|---|---:|---:|---:|---:|---:|
| Hard to solve + hard to judge | 10 | 86.36 | 87.14 | 0.16 | 0.45 |
| Easy to solve + hard to judge | 8 | 14.37 | 90.50 | 0.66 | 0.52 |
| Hard to solve + easy to judge | 9 | 86.44 | 11.26 | 0.20 | 0.85 |

The strongest qualitative evidence for divergence is again the contrast between:

- **Easy to solve + hard to judge**: solver models usually answer correctly, but judge models struggle.
- **Hard to solve + easy to judge**: solver models struggle, but judges can still identify the better response.

This means pairwise judging difficulty depends not only on the original item, but also on how separable the two candidate responses are.

## The Solver Bin 5 Anomaly

The bin-level trend mostly shows lower judge scores for harder solver bins, but solver bin 5 breaks the pattern.

| Solver difficulty bin | Mean solver difficulty | Mean judge score |
|---:|---:|---:|
| 1 | -7.20 | 0.769 |
| 2 | -3.42 | 0.767 |
| 3 | -1.58 | 0.660 |
| 4 | -0.62 | 0.663 |
| 5 | 0.23 | 0.803 |
| 6 | 0.73 | 0.597 |
| 7 | 1.15 | 0.593 |
| 8 | 1.86 | 0.680 |
| 9 | 3.33 | 0.653 |
| 10 | 5.57 | 0.530 |

Bin 5 has the highest mean judge score, 0.803. This is higher than the average of bins 1-4, 0.714, and much higher than the average of bins 6-10, 0.610. Across bins, the rank relationship between solver bin and judge score is negative overall (Spearman rho = -0.661, p = 0.038), and it becomes stronger if bin 5 is excluded (rho = -0.733, p = 0.025). So bin 5 looks like a local pocket of easy-to-judge items inside the broader downward trend.

The source mix helps explain why this is not just noise. Bin 5 contains many technical or structured items:

| Source | Items in bin 5 |
|---|---:|
| Engineering | 4 |
| Chemistry | 2 |
| Computer science | 2 |
| Other | 2 |
| Psychology | 2 |
| Health | 1 |
| Law | 1 |
| Math | 1 |

Several high-scoring bin 5 examples have objective answer structures:

- **Chemistry, item 4238**: hydrolysis constant of Al3+ and H3O+ concentration. Solver score is 0.40, but judge score is 1.00.
- **Computer science, item 10716**: maximum number of elements examined by binary search in a sorted list of 120 integers. Solver score is 0.50, but judge score is 0.90.
- **Engineering, item 11712**: mating spur gears, velocity ratio, and center distance. Solver score is 0.625, but judge score is 0.95.
- **Engineering, item 12029**: mass transfer coefficient over a flat plate and sphere. Solver score is 0.60, but judge score is 0.95.

These examples suggest a plausible mechanism:

> Bin 5 items are not trivial to solve from scratch, but many contain clear correctness cues once candidate responses are available. Formula choice, final numeric option, algorithmic step count, or definitional match can make the pairwise judgment easier than the original solving task.

In other words, bin 5 appears to contain a **verification-advantage** cluster. These are items where direct answering requires domain knowledge or computation, but judging is helped by response-pair contrast. If one response uses the right formula, lands on the correct option, or avoids a visible conceptual error, the judge can identify it without fully solving the problem independently. The item-level retrieval artifact `mmlu_k2_bin5_verification_advantage_retrievable_examples.csv` records the exact `solver_item_id`, `pair_id`, source line, prompt excerpt, and scores for these examples.

This interpretation also explains why these items appear specifically in a moderate solver-difficulty bin. They are not easy retrieval questions: solving hydrolysis concentration, binary-search worst-case comparisons, or spur-gear ratios requires applying a rule or computation that many models miss. But they are also not so hard that all candidate responses become speculative or indistinguishable. The middle difficulty range creates useful response contrast: some models produce the correct procedure, while others make visible local mistakes. The judge can then compare the two responses by checking a concrete rule, magnitude, option, or calculation trace.

There are exceptions. For example, bin 5 includes a law/privacy question and a religion/social-science question where judge scores are lower. So the bin is not uniformly easy to judge. But the high-judge-score technical examples are numerous enough to pull the bin mean upward.

### Why Would Bin 5 Have Stronger Judge Cues Than Other Bins?

The stronger claim is not that other bins lack cues. They almost certainly contain some easy-to-compare response pairs. The better interpretation is that bin 5 appears to have a higher concentration of items where the cues are clear, reliable, and visible to many judge models.

Several observed patterns support this interpretation:

| Solver difficulty bin | Mean solver score | Mean judge score | Share technical | Items with judge score >= 0.90 |
|---:|---:|---:|---:|---:|
| 1 | 0.674 | 0.769 | 0.375 | 7 / 16 |
| 2 | 0.522 | 0.767 | 0.200 | 7 / 15 |
| 3 | 0.572 | 0.660 | 0.333 | 3 / 15 |
| 4 | 0.528 | 0.662 | 0.438 | 4 / 16 |
| 5 | 0.514 | 0.803 | 0.600 | 9 / 15 |
| 6 | 0.388 | 0.597 | 0.400 | 5 / 15 |
| 7 | 0.354 | 0.593 | 0.125 | 1 / 16 |
| 8 | 0.285 | 0.680 | 0.267 | 5 / 15 |
| 9 | 0.162 | 0.653 | 0.600 | 1 / 15 |
| 10 | 0.200 | 0.530 | 0.250 | 2 / 16 |

Bin 5 is unusual in two ways. First, it has the highest fraction of very easy-to-judge items: 9 of 15 have mean judge score at least 0.90. Second, it has the highest share of technical items, tied with bin 9 at 0.600, but unlike bin 9 it is only moderately difficult for solvers. This combination matters. Bin 9 also contains many technical items, but those items are much harder to solve on average (mean solver score 0.162), so the response pairs may more often contain two poor, confused, or hard-to-rank answers. Bin 5 instead sits in a middle region: solver models disagree enough to create contrast, but the task structure still gives judges concrete evidence to compare.

The neighboring bins make this clearer. Bin 4 has a similar mean solver score to bin 5 (0.528 versus 0.514), and it also contains some easy-to-judge structured problems: least common multiple / greatest common factor, aluminum heat transfer, and a cylindrical-container optimization item all have mean judge scores of 0.95. But bin 4 also has a heavier low-judge tail. Examples include:

- **Physics, item 10340**: parachutist drag and distance to 0.95 terminal velocity. Solver score is 0.50, but judge score is 0.15.
- **Psychology, item 2646**: symptom validity / malingering assessment measures. Solver score is 0.444, but judge score is 0.30.
- **Biology, item 3435**: a "NOT true" meiosis question. Solver score is 0.70, but judge score is 0.35.
- **Other, item 5708**: inherent risk, control risk, and detection risk in auditing. Solver score is 0.60, but judge score is 0.35.

These low-judge bin 4 items are not simply harder to solve than bin 5 items. Rather, they look harder to judge because the response comparison likely depends on specialized terminology, negation, or fine-grained distinctions among close options. This weakens the response-pair cues even when the original item is only moderately difficult.

Bin 6 shows the other side of the boundary. It also has some high-cue technical items: a projectile motion item, blood-allele frequencies, a conductor electric-field item, and a skeletal-muscle ATP calculation all have judge scores of 0.90 or higher. But bin 6 has a lower mean solver score than bin 5 (0.388 versus 0.514), and its low-judge tail is more conceptually diffuse:

- **Philosophy, item 11002**: Singer's view of social problems from a genetic supermarket. Solver score is 0.40, but judge score is 0.10.
- **Philosophy, item 11210**: Baier's account of genuine moral rules. Solver score is 0.40, but judge score is 0.15.
- **Law, item 1748**: contract facts about ceramic dinnerware production. Solver score is 0.444, but judge score is 0.30.
- **Biology, item 3335**: marine ecosystem zones and nutrient/upwelling terminology. Solver score is 0.40, but judge score is 0.30.

This suggests that bin 6 has enough difficulty that response-pair contrast is less reliable. Some items still produce clear cues, but more items produce response pairs where both answers may be partially plausible, too domain-specific, or difficult to distinguish without knowing the underlying concept.

This suggests a "contrast plus checkability" mechanism:

1. **Lower bins are often easier to solve directly.** Items such as item 4971 on Aztec expansion, item 2826 on mutation frequency, or item 6311 on alpha-thalassemia counseling are more strongly cued or more directly retrievable. Judging is still often successful, but the gap between solving and judging is less conceptually surprising because many models can already produce the answer.
2. **Higher bins may be too hard to verify.** Items such as item 3250 on arthropod movement, item 7568 on production possibility frontiers, item 6728 on sensory assessment, or item 4789 on poetry/history interpretation have low judge scores as well as low solver scores. In these cases, both candidate responses may be flawed, domain-specific, or superficially plausible, so the judge may need the same missing knowledge as the solver.
3. **Bin 5 may be a sweet spot.** Items are nontrivial enough that responses differ, but structured enough that judges can use final options, formulas, calculations, or standard definitions as cues.

So the bin 5 anomaly should not be framed as "only bin 5 has cues." It is better framed as:

> Bin 5 has more items where response contrast and answer checkability line up. Other bins may have one of these ingredients, but less often both at once.

The retrievable contrast set is:

| Group | Item | Source | Bin | Solver score | Judge score | Retrieval note |
|---|---:|---|---:|---:|---:|---|
| Bin 5 verification advantage | 4238 | Chemistry | 5 | 0.40 | 1.00 | Hydrolysis concentration calculation |
| Bin 5 verification advantage | 10716 | Computer science | 5 | 0.50 | 0.90 | Binary-search worst-case comparisons |
| Bin 5 verification advantage | 11712 | Engineering | 5 | 0.625 | 0.95 | Spur-gear velocity ratio and center distance |
| Bin 5 verification advantage | 12029 | Engineering | 5 | 0.60 | 0.95 | Flat-plate/sphere mass-transfer calculation |
| Lower-bin contrast | 4971 | History | 2 | 0.60 | 0.85 | Aztec expansion, more retrieval/cued |
| Lower-bin contrast | 2826 | Biology | 2 | 0.429 | 0.90 | Mutation-frequency calculation, strongly cued |
| Lower-bin contrast | 6311 | Health | 1 | 0.571 | 0.75 | Alpha-thalassemia counseling pattern |
| Higher-bin hard-to-judge contrast | 3250 | Biology | 10 | 0.10 | 0.05 | Arthropod movement with exoskeleton |
| Higher-bin hard-to-judge contrast | 7568 | Economics | 10 | 0.10 | 0.25 | Production possibility frontier condition |
| Higher-bin hard-to-judge contrast | 6728 | Health | 10 | 0.00 | 0.30 | Sensory assessment clinical knowledge |
| Higher-bin hard-to-judge contrast | 4789 | History | 8 | 0.40 | 0.35 | Poetry/history interpretation |

## Category 1: Hard to Solve + Hard to Judge

These items are difficult on both axes. They tend to involve subtle conceptual distinctions, specialized factual knowledge, or options that are easy to confuse.

Representative examples:

- **Item 7568, economics**: asks when a production possibility frontier will be a straight line. Solver score is 0.10 and judge score is 0.25. The item depends on a precise economic condition: resources not being specialized. Many distractors are plausible if the model loosely associates PPFs with efficiency, competition, or opportunity cost.
- **Item 3250, biology**: asks how arthropods move despite having an exoskeleton. Solver score is 0.10 and judge score is 0.05. This is not mainly a multi-step verification case. The response pair contrasts two plausible biological mechanisms: one answer emphasizes segmented joints joined by flexible chitin, while the other emphasizes muscles attached to the exoskeleton. Both explanations contain relevant facts, so judging requires knowing which option the benchmark treats as the best answer rather than simply spotting an arithmetic or formula error.
- **Item 8286, math**: asks about the largest possible number of same-digit-length terms in a geometric sequence. Solver score is 0.00 and judge score is 0.65. This is mathematically abstract and likely produces response pairs with plausible but hard-to-check reasoning.

Interpretation:

> These are globally hard MMLU items. They are useful for showing that some benchmark items stress both direct problem solving and pairwise judging, especially when the correct answer depends on a compact but non-obvious concept.

## Category 2: Easy to Solve + Hard to Judge

These are important because they show that high solver performance does not guarantee easy judging.

Representative examples:

- **Item 3036, biology**: asks why one would call a flock of birds a society. Solver score is 0.90, but judge score is 0.55. The options are semantically close, so pairwise evaluation may hinge on wording rather than difficult biology.
- **Item 7623, economics**: asks when households demand more money as an asset. Solver score is 0.80, but judge score is 0.65. The economic fact is relatively accessible, but response comparisons may be sensitive to whether the explanation correctly separates money demand from goods demand, nominal interest rates, and asset substitution.
- **Item 4773, history**: asks about a passage on women's rights after the French Revolution. Solver score is 1.00, but judge score is 0.50. The original question may be answerable from strong contextual clues, while judging can be hard if both responses use similar historical language.

Interpretation:

> These items are answerable, but the judging task is fine-grained. The judge may need to distinguish between two responses that are both semantically plausible, partially correct, or similarly worded.

This is useful for the paper because it weakens the simple story that judge difficulty is just solver difficulty re-estimated from a different response matrix.

## Category 3: Hard to Solve + Easy to Judge

These items are difficult for solvers but comparatively easy for judges.

Representative examples:

- **Item 8762, math**: asks for the distance traveled by prey dropped from a hawk using a parabolic trajectory. Solver score is 0.00, but judge score is 0.72. The computation is hard from scratch, but a response with the correct setup or final option may be visibly stronger.
- **Item 5087, other**: asks for a survey percentage about free media without government censorship. Solver score is 0.22, but judge score is 0.85. This kind of factual recall is hard to answer directly, but response pairs may reveal better evidence use or a clearer match to the target statistic.
- **Item 9158, physics**: asks for a Fraunhofer diffraction intensity ratio. Solver score is 0.125, but judge score is 0.85. The item is specialized, but candidate responses may differ sharply in whether they identify the correct formula or value.

Interpretation:

> These cases show that judging can be easier than solving when the candidate responses expose useful evidence. A model does not necessarily need to solve the item from scratch if one answer contains an obvious formula, value, or reasoning advantage.

## Cross-Cutting Patterns

### 1. Technical items can be hard to solve but easy to compare

Bin 5 is the clearest example. Chemistry, engineering, math, and computer science items often require calculation or domain knowledge, but the pairwise response comparison may provide strong cues: a final numeric option, a recognizable formula, a binary-search count, or a standard engineering relation.

### 2. Abstract or semantically close options can make judging hard

The easy-to-solve/hard-to-judge cases often have close conceptual distractors. This makes the original answer accessible to solvers, but it makes pairwise evaluation sensitive to subtle wording differences between candidate responses.

### 3. Judging difficulty is conditioned on the response pair

The same item could be easy or hard to judge depending on whether the two candidate responses are clearly separated. If both responses are plausible, judging is hard. If one response makes a salient mistake, judging is easier, even when the original problem is difficult.

### 4. Bin 5 should be treated as a meaningful exception, not just an outlier to discard

The overall trend still points in the expected direction: harder solver bins tend to have lower judge scores. But bin 5 reveals a more interesting pattern. Moderate solver difficulty may be the region where items are nontrivial enough to separate solver responses, but still structured enough for judges to compare those responses reliably.

## Suggested Paper Framing

A concise paragraph could be:

> MMLU shows a generally negative relationship between solver difficulty and judge performance, but solver difficulty bin 5 is an informative exception. This bin has the highest mean judge score despite moderate solver difficulty. Inspection suggests that many bin 5 items are technical but highly structured: chemistry calculations, engineering formulas, binary search, or other problems with clear numerical or definitional answer cues. These items can be difficult to solve from scratch while remaining easy to judge once candidate responses are visible. Conversely, some easy-to-solve items are hard to judge when the candidate responses differ only in subtle wording or contain semantically close explanations. These cases indicate that judge difficulty is not simply task difficulty, but task difficulty filtered through the response pair presented to the judge.

## Recommended Examples to Inspect Manually

For a short paper table, inspect one or two examples per category from:

- `hard_to_solve__hard_to_judge`: items 7568, 3250, 8286
- `easy_to_solve__hard_to_judge`: items 3036, 7623, 4773
- `hard_to_solve__easy_to_judge`: items 8762, 5087, 9158
- solver bin 5 verification advantage: items 4238, 10716, 11712, 12029
- lower-bin and higher-bin contrasts for the bin 5 explanation: items 4971, 2826, 6311, 3250, 7568, 6728, 4789

Use `mmlu_k2_bin5_verification_advantage_retrievable_examples.csv` for the item IDs, prompt excerpts, source-line references, and retrieval commands used in the bin 5 discussion. Use `mmlu_k2_case_studies_full.csv` for prompt text and score dictionaries, `mmlu_k2_case_studies_compact.csv` for difficulty/score numbers, and `mmlu_k2_solver_bin5_items.csv` for all solver-bin-5 items.
