# MMLU K=2 Bin 5 Verification-Advantage Examples

Concrete item IDs for the bin-5 discussion, with lower- and higher-bin contrasts. Retrieve any item by searching `solver_item_id` / `original_id`, or using the `retrieval_hint` in the CSV.

## Bin 5: Moderately Hard to Solve, Easy to Judge

- **Item 4238** (mmlu-pro-chemistry, bin 5): solver score 0.40, judge score 1.00.
  - Note: Hydrolysis calculation: requires applying a formula from scratch, but the candidate answer can be checked by magnitude/procedure.
  - Prompt excerpt: If the hydrolysis constant of Al^3+ is 1.4 × 10^-5, what is the concentration of H_3O^+ in 0.1 M AlCl_3? (A) 1.4 × 10^-5 M (B) 2.8 × 10^-6 M (C) 5.6 × 10^-5 M (D) 3.0 × 10^-4 M (E) 7.0 × 10^-3 M (F) 8.4 × 10^-6 M (G) 1.2 × 10^-3 M (H) 0.1 M (I) 4.2 × 10^-4 M (J) 1.0 × 10^-3 M If you cannot determine
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 46; pair_id `d50b6560-bbb7-5118-b329-8ed50c155365`.

- **Item 10716** (mmlu-pro-computer science, bin 5): solver score 0.50, judge score 0.90.
  - Note: Binary search: solving requires knowing the exact ceiling/log rule, but the judge can compare two explicit calculations.
  - Prompt excerpt: A sorted list of 120 integers is to be searched to determine whether the value 100 is in the list. Assuming that the most efficient searching algorithm is used, what is the maximum number of elements that must be examined? (A) 100 (B) 30 (C) 120 (D) 10 (E) 8 (F) 7 (G) 15 (H) 12 (I) 20 (J) 60 If you 
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 75; pair_id `754b1cf8-067c-513f-a0d3-c8bd401e8afb`.

- **Item 11712** (mmlu-pro-engineering, bin 5): solver score 0.62, judge score 0.95.
  - Note: Spur gear ratio: technical calculation with units/ratio cues that are easier to verify once shown.
  - Prompt excerpt: A pair of mating spur gears increases the speed of the shaft approximately by 4 times. Thediametralpitch for the gear is 8 in. and it is desired that the center distance be approximately 12 in. Determine the actual velocity ratio and the center distance for the gears. (A) V_R = 4.5, C_d = 12.5 in. (
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 104; pair_id `3239d81d-8220-5b81-99e0-1b7a470df3be`.

- **Item 12029** (mmlu-pro-engineering, bin 5): solver score 0.60, judge score 0.95.
  - Note: Engineering mass-transfer/flat-plate item: formulaic technical reasoning, easier to check when the response exposes the calculation path.
  - Prompt excerpt: (a) A mixture of air and water vapor is passing over a flat plate 2 ft long at a rate of 210 ft/sec at 1atmand 75°F. Determine the mass transfer coefficient of water vapor in air if the flow is turbulent and the concentration of water vapor in air is very low (i.e.,P_bm/ P \approx 1). (b) Find the m
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 107; pair_id `3be26fc0-ead5-5d0f-8c87-2feb0524e902`.

## Lower-Bin Contrast: Easier to Solve Directly

- **Item 4971** (mmlu-pro-history, bin 2): solver score 0.60, judge score 0.85.
  - Note: Aztec expansion: more retrieval/cued historical knowledge; easier solving and judging move together.
  - Prompt excerpt: The Aztec Empire was based on the systematic expansion of: (A) a vast network of roads and trade routes. (B) the expansion and control of waterways for transportation and irrigation. (C) the cultivation and trade of cacao and maize. (D) religious beliefs that required extensive and escalating human 
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 146; pair_id `ff5f3ce7-1423-5df5-922d-644070dfaea0`.

- **Item 2826** (mmlu-pro-biology, bin 2): solver score 0.43, judge score 0.90.
  - Note: Gene mutation frequency: structured but strongly cued; many solvers can answer directly.
  - Prompt excerpt: A gene C mutates to c with a frequency of 2 × 10^-6 per generation. There are no other forces acting on these alleles and mating is random. How many generations are needed to increase the frequency of gene c from 2 percent to 3 percent? (A) 2,500 generations (B) 6,200 generations (C) 15,000 generati
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 1; pair_id `a3f45559-827d-53c5-9f1b-200ea1876ac8`.

- **Item 6311** (mmlu-pro-health, bin 1): solver score 0.57, judge score 0.75.
  - Note: Alpha-thalassemia counseling: familiar clinical genetics pattern; answer is more directly retrievable/cued.
  - Prompt excerpt: A couple comes for preconceptional genetic counseling because they both have a family history of α-thalassemia. The woman has a minimally decreased hemoglobin concentration. Genetic studies show a single gene deletion. The man has microcytic anemia and a two-gene deletion. If the two-gene deletion i
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 122; pair_id `d61248f3-8e80-5097-8892-10c086a492d2`.

## Higher-Bin Contrast: Hard to Solve and Hard to Judge

- **Item 3250** (mmlu-pro-biology, bin 10): solver score 0.10, judge score 0.05.
  - Note: Biology hard-bin item with very low solver and judge scores; verification also appears difficult.
  - Prompt excerpt: Arthropods have an exoskeleton. How is movement accomplished? (A) They use their wings to move (B) They move by changing the shape of their exoskeleton (C) They use cilia on the surface of their exoskeleton to move (D) They move due to their muscular system (E) They have a soft, pliable inner layer 
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 9; pair_id `49c0f568-1ac2-53dc-be78-f3eea93820fd`.

- **Item 7568** (mmlu-pro-economics, bin 10): solver score 0.10, judge score 0.25.
  - Note: Economics PPF item: deceptively conceptual; candidate responses can plausibly differ in abstract wording.
  - Prompt excerpt: A production possibility frontier will be a straight line when (A) efficiency is achieved (B) there is no opportunity cost (C) utility is maximized (D) resources are not specialized (E) there is economic growth (F) the goods on the axes are complements in consumption (G) the law of diminishing retur
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 90; pair_id `a5c0fd0a-7440-5461-9a32-c6c5ff574ae0`.

- **Item 6728** (mmlu-pro-health, bin 10): solver score 0.00, judge score 0.30.
  - Note: Health hard-bin item with low solver and judge scores; checking likely requires domain-specific clinical knowledge.
  - Prompt excerpt: Which of the following is true in a sensory assessment of the arm? (A) It is acceptable to test pain with a venepuncture needle (B) Impulses for pain travel principally up the anterior columns (C) It is acceptable to test pain with a surgical scalpel (D) Proprioception is appreciated via the same tr
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 129; pair_id `0a347026-2ec5-5879-9635-e6fc818d021b`.

- **Item 4789** (mmlu-pro-history, bin 8): solver score 0.40, judge score 0.35.
  - Note: Poetry/history interpretation: candidate responses can sound plausible, so pairwise checking gives weaker cues.
  - Prompt excerpt: This question refers to the following information. On Being Brought from Africa to America 'Twas mercy brought me from my Pagan land, Taught my benighted soul to understand That there's a God, that there's a Saviour too; Once I redemption neither sought nor knew. Some view our sable race with scornf
  - Source: `benchmarks/mmlu/judgebench_mmlu_pro_questions_unique.jsonl` line 135; pair_id `14ead85f-e4c6-5ae2-a586-6388d9b9b4f1`.

Full CSV: `mmlu_k2_bin5_verification_advantage_retrievable_examples.csv`.