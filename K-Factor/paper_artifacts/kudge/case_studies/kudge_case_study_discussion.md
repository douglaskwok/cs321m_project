# Kudge Case Study Discussion

This note summarizes qualitative patterns from the Kudge K-factor case-study extraction. The goal is not to treat these examples as new statistical evidence, but to make the scatter between solver-side and judge-side item difficulty interpretable.

The extracted cases use K=2 K-factor item difficulties. "Hard" items are in the top quartile of centered difficulty percentile, and "easy" items are in the bottom quartile. The selection keeps the most extreme items in each region of the solver-vs-judge difficulty plane.

## Summary

The extracted cases support a useful interpretation of the weak solver/judge difficulty relationship: solving difficulty and judging difficulty can diverge for qualitatively understandable reasons.

| Category | Available | Selected | Mean solver percentile | Mean judge percentile | Mean solver score | Mean judge score |
|---|---:|---:|---:|---:|---:|---:|
| Hard to solve + hard to judge | 45 | 10 | 95.98 | 94.59 | 0.39 | 0.55 |
| Easy to solve + hard to judge | 13 | 10 | 14.67 | 88.40 | 0.74 | 0.52 |
| Hard to solve + easy to judge | 9 | 9 | 87.75 | 12.83 | 0.55 | 0.80 |

The most important case-study group is **easy to solve + hard to judge**. These items directly show that judging difficulty is not just solver difficulty in another form. Some items are answerable by solver models but still difficult for judge models to evaluate pairwise.

## Category 1: Hard to Solve + Hard to Judge

These items are difficult on both axes. They tend to combine substantive task difficulty with answer-pair ambiguity. The top examples include technical science or geography questions where both candidate responses contain partial reasoning, translation artifacts, or plausible-looking but flawed explanations.

Representative examples:

- **Item 145, Korean-Easy**: chemistry equilibrium / partial pressure. Both candidate answers appear to use similar symbolic equilibrium reasoning, but the reasoning includes confusing chemical notation and likely reaction-formulation errors. This makes the item difficult to solve and also difficult to judge, because the judge must identify subtle flaws in two mathematically styled explanations.
- **Item 271, Korean-Easy**: geography / state shape. Both responses analyze several geographic categories in natural language. The distinction between "elongated" and other state-shape categories is conceptually simple if known, but the Korean wording and translated option labels make the comparison less crisp.
- **Item 121, Korean-Easy**: radon hazard. Both responses discuss radioactivity, inhalation, decay products, and biological accumulation. The correct intuition is that inhaled radon decay products can damage lung tissue, but the answer options contain tempting incorrect references to iodine-like chemistry. This creates both factual difficulty and judge confusion.

Interpretation:

> These are globally hard benchmark items. They are useful examples when we want to show that some Kudge items stress both direct problem solving and response comparison.

One interesting detail is that half of the selected hard/hard examples are labeled `Korean-Easy`. This suggests that the nominal Easy/Hard split does not perfectly track the empirical K-factor difficulty estimated from model responses.

## Category 2: Easy to Solve + Hard to Judge

This is the clearest evidence for divergence between solving and judging. Solver models usually do well, but judge models struggle to decide which response is better.

Representative examples:

- **Item 726, Korean-Hard**: quantum/spin operator uncertainty. The chosen answer is concise and gives the correct uncertainty directly, while the rejected answer gives a longer derivation. This is easy for solvers that know the computation, but hard for judges because the longer response looks more complete even if the final comparison depends on checking the derivation.
- **Item 7, Korean-Easy**: tangent line for `y = x + e^x` at `x = 0`. This is straightforward to solve, and solver accuracy is high. But both responses compute the same derivative and describe the same point-slope reasoning, so pairwise judging becomes hard because the surface-level differences are minimal.
- **Item 162, Korean-Easy**: calcium atom composition. Both responses identify calcium's atomic number and discuss protons/electrons/neutrons. The judge must distinguish small differences in wording and option interpretation rather than solve a hard scientific problem.

Interpretation:

> These cases show that judging can be hard even when solving is easy. Pairwise evaluation may require detecting subtle quality differences between two mostly correct or similarly structured explanations.

This category is probably the strongest qualitative support for the paper's core claim that judge ability is not reducible to solver ability. The solver task is often a simple calculation or recall problem, but the judge task asks for fine-grained comparative evaluation.

## Category 3: Hard to Solve + Easy to Judge

These items are difficult for solvers but comparatively easy for judges. The common pattern is that one response contains a clearer defect, even though solving the original item from scratch is nontrivial.

Representative examples:

- **Item 262, Korean-Easy**: urban agriculture benefits. The original question asks for an exception. Solving requires careful interpretation of the "not a benefit" wording. However, the response pair is easier to judge because one response more clearly mishandles the exception framing.
- **Item 273, Korean-Easy**: U.S.-Mexico border issue. The original item is conceptually ambiguous and politically/geographically contextual. But the pairwise responses differ in how directly they identify immigration as the central issue, making the judge comparison easier than solving from scratch.
- **Item 279, Korean-Easy**: characteristics of cities in developing countries. The original question asks for the non-characteristic. Solvers may struggle with the options, but judges can compare whether a response correctly recognizes that "well-developed infrastructure" is the likely exception.

Interpretation:

> These cases show that judging can be easier than solving when the answer pair contains clear quality cues. The judge can rely on comparative evidence in the responses rather than reconstructing the full solution independently.

This category helps explain why solver difficulty and judge difficulty need not be positively correlated. Pairwise judging is conditioned on the two candidate responses; if one response visibly misreads the question or mishandles a key option, judging becomes easier even for a hard original problem.

## Late-Bin Uptick: Why Are Bins 9 and 10 Easier to Judge Than Bin 8?

The Kudge bin plot has a small but interesting late-bin uptick. Mean judge score falls through bin 8, then increases slightly in bin 9 and more clearly in bin 10.

| Solver difficulty bin | Mean solver difficulty | Mean solver score | Mean judge score | High-judge items | Low-judge items |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.41 | 0.53 | 0.58 | 7 / 38 | 15 / 38 |
| 9 | 3.02 | 0.41 | 0.59 | 5 / 38 | 13 / 38 |
| 10 | 6.52 | 0.48 | 0.65 | 12 / 38 | 10 / 38 |

Here, "high-judge" means mean judge score at least 0.75, and "low-judge" means mean judge score at most 0.50. The uptick is therefore not just a smooth statistical artifact: bin 10 has more high-judge items and fewer low-judge items than bin 8.

The examples suggest a response-pair explanation. Bin 8 contains many items where both responses are long, plausible, and similar in structure, even when the original item is not maximally hard. For example:

- **Item 702, Korean-Hard**: organic reagents for a cyanohydrin / hydrolysis sequence. Solver score is 0.27 and judge score is 0.36. Both responses discuss plausible reagent sequences and mechanistic steps, so the judge has to identify a fine chemical distinction.
- **Item 706, Korean-Hard**: the number of uniformly distributed stars per parallax interval. Solver score is 0.64 and judge score is 0.36. Both responses reason from distance-parallax scaling, so judging depends on whether the derivative transformation is handled correctly.
- **Item 33, Korean-Easy**: probability that a random `x` in `[0,3]` is less than random `y` in `[0,4]`. Solver score is 0.46 and judge score is 0.36. Both responses look like area/probability derivations, so surface cues are weak.
- **Item 276, Korean-Easy**: zero population growth. Solver score is 0.55 and judge score is 0.00. The item is conceptually simple, but the translated answer choices are awkward, so response comparison likely depends on parsing the option wording rather than spotting a clean numerical error.

In contrast, many high-judge items in bins 9 and 10 have sharper visible differences between the chosen and rejected responses. In bin 9:

- **Item 185, Korean-Easy**: Graham's law / gas effusion. Solver score is 0.55 and judge score is 0.91. The correct solution has a clear molecular-mass calculation, so the judge can use a checkable quantitative cue.
- **Item 718, Korean-Hard**: identifying a reagent from reaction clues. Solver score is 0.18 and judge score is 0.91. The chosen response commits to a specific reducing-agent interpretation, while the rejected response stays broader and less targeted.
- **Item 682, Korean-Hard**: a multistep organic synthesis ending in a symmetry / NMR-like count. Solver score is 0.09 and judge score is 0.82. The chosen response gives a concrete product sequence and final count, making the pairwise comparison more separable.

Bin 10 has even clearer examples:

- **Item 262, Korean-Easy**: urban agriculture benefits, asking for the exception. Solver score is 0.55 and judge score is 1.00. The chosen response directly handles the "not a benefit" framing, while the rejected response appears to drift into an irrelevant temperature/water detail.
- **Item 58, Korean-Easy**: blackbody energy available to melt ice when temperature doubles. Solver score is 0.73 and judge score is 1.00. The chosen response uses the `T^4` relationship, while the rejected response begins from photon energy and does not directly use the cavity-radiation scaling.
- **Item 200, Korean-Easy**: identifying second-order kinetics from a linear `1/[A]` plot. Solver score is 0.73 and judge score is 0.91. The final rule is highly checkable.
- **Item 650, Korean-Hard**: transit probability for two exoplanets with different orbital periods. Solver score is 0.36 and judge score is 0.91. The chosen response uses the transit-probability relationship and period-distance scaling, creating an identifiable reasoning advantage.
- **Item 690, Korean-Hard**: selecting the correct `1H NMR` pattern for an aromatic halogen/carbonyl compound. Solver score is 0.36 and judge score is 0.91. The chosen response appeals to symmetry and spectral structure, whereas the rejected response is more generic.

This does not mean bins 9 and 10 are intrinsically easier to judge. They still contain low-judge examples, especially in advanced organic synthesis, quantum mechanics, and multistep gas-analysis problems where both responses are technical and plausible. But compared with bin 8, bin 10 seems to include more cases where one response exposes a salient flaw or a recognizable formula/exception-handling cue.

The most plausible interpretation is:

> The hardest solver bins can become easier to judge when the response pair is strongly separated. Some bin 9/10 items are hard because the underlying problem is specialized, but the candidate responses reveal enough contrast that judges can identify the better answer. Bin 8, by contrast, contains more items where both responses are similarly plausible, so judges receive weaker comparative cues.

## Cross-Cutting Patterns

### 1. Pairwise judging is response-pair dependent

The same underlying question can become easy or hard to judge depending on the candidate responses. If both responses are similar, partially correct, or verbose in plausible ways, the judge task is hard. If one response contains an obvious conceptual or option-selection error, judging is easier.

### 2. Translation and wording artifacts matter

Several selected examples involve Korean translations of technical or multiple-choice questions. The difficulty is not purely domain knowledge; it also includes whether the model can handle translated option labels, unnatural phrasing, or subtle "not" questions.

### 3. Mean judge score and judge difficulty are related but not identical

The hard-to-judge categories have lower or moderate mean judge scores, but they are not simply "all models fail." Judge difficulty reflects uncertainty and separation in the fitted K-factor model, while mean score is just average observed correctness. The case studies are therefore best used to explain patterns in the difficulty plane, not to replace the model-based analysis.

### 4. The nominal Easy/Hard label is imperfect

Many selected examples from all three categories are labeled `Korean-Easy`. This suggests that the Easy/Hard split is useful but incomplete. Empirical K-factor difficulty captures additional structure, including response-pair ambiguity and model-specific weaknesses.

## Suggested Paper Framing

A concise paragraph could be:

> To interpret the weak association between solver-side and judge-side K-factor difficulty, we examined representative Kudge items from three regions of the difficulty plane. Hard-to-solve/hard-to-judge items often involved technical reasoning where both candidate responses contained plausible but flawed reasoning. Easy-to-solve/hard-to-judge items were especially informative: these were often straightforward problems where the two candidate responses were both close to correct, making pairwise evaluation difficult despite high solver performance. Conversely, hard-to-solve/easy-to-judge items showed that pairwise comparison can be easier than solving when one response contains a salient error. These cases suggest that judging difficulty is strongly conditioned on the response pair and is not merely a proxy for task-solving difficulty.

## Recommended Examples to Inspect Manually

For a short paper table, inspect one or two examples per category from:

- `hard_to_solve__hard_to_judge`: items 145, 271, 121
- `easy_to_solve__hard_to_judge`: items 726, 7, 162
- `hard_to_solve__easy_to_judge`: items 262, 273, 279

Use `kudge_k2_case_studies_full.csv` for prompt and response text, and `kudge_k2_case_studies_compact.csv` for difficulty/score numbers.
