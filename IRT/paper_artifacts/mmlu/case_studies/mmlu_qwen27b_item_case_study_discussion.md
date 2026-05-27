# MMLU Item-Level Case Studies: Why Qwen3.5 27B Can Judge What It Fails To Solve

I used the 45 cases where Qwen3.5 27B answered the MMLU solver item incorrectly but judged the correct response in both original and swapped presentation orders as the search pool. The discussion below focuses only on five representative cases.

Supporting files:

- `mmlu_qwen27b_selected_solver_wrong_judge_right_cases.csv`
- `mmlu_qwen27b_selected_solver_wrong_judge_right_cases.json`
- `mmlu_qwen27b_solver_wrong_judge_right_summary.csv`

## Summary

Across the selected cases, the same pattern repeats: direct solving requires Qwen3.5 27B to independently retrieve or compute the answer among many options, but judging gives it two worked candidate solutions. When one candidate contains the right intermediate structure, formula, doctrine, or final answer trail, Qwen can often recognize the better answer even though it failed to produce the answer alone.

This suggests the MMLU judge task is partly a verification task rather than a pure knowledge-generation task.

## Case 1: Law, Search Incident To Arrest

- `original_id`: `1097`
- Gold answer: `A`
- Qwen solver prediction: `C`
- Winner final letter: `A`
- Loser final letter: `E`

Qwen's solver answer selected the search-incident-to-arrest path. The winning judged response, however, made the key distinction: once the driver was secured away from the vehicle, the search could not be justified merely as incident to arrest under the relevant vehicle-search rule. The losing response reasoned more broadly about the traffic stop and probable cause.

Why judging is easier here: Qwen did not need to retrieve the controlling doctrine from scratch. It only needed to compare two legal rationales and recognize that the winning response had the sharper limiting rule.

## Case 2: Computer Science, Restricted Prefix Code

- `original_id`: `10608`
- Gold answer: `D`
- Qwen solver prediction: `J`
- Winner final letter: `D`
- Loser final letter: `B`

The solver setting asks the model to derive an optimal uniquely decodable code from a nonstandard alphabet: ternary first symbol, binary thereafter. Qwen's standalone reasoning wandered among visually similar code options and selected the wrong letter.

The judged winner exposed the relevant structure more clearly: assign shortest codewords to the highest probabilities and preserve decodability under the restricted code format. The loser remained more generic and did not pin down the same option.

Why judging is easier here: the candidate responses turn an abstract combinatorial search into a comparison of two proposed constructions.

## Case 3: Health, Recurrent Laryngeal Nerve

- `original_id`: `6742`
- Gold answer: `I`
- Qwen solver prediction: `B`
- Winner final letter: `I`
- Loser final letter: `C`

Qwen's solver output recalled several correct facts about recurrent laryngeal nerve injury, but failed to map them to the right answer option. The winning response separated the crucial anatomy: the cricothyroid is spared because it is innervated by the superior laryngeal nerve, while recurrent-laryngeal-nerve-dependent muscles are impaired.

Why judging is easier here: Qwen can recognize the response that organizes the anatomy correctly, even though option mapping was brittle in direct solving.

## Case 4: Math/Physics, Projectile Motion With Side Acceleration

- `original_id`: `8857`
- Gold answer: `I`
- Qwen solver prediction: `C`
- Winner final letter: `I`
- Loser final letter: `F`

The direct solution requires tracking time of flight, horizontal velocity, side acceleration, vertical velocity at landing, and then combining components into speed. Qwen's solver produced the setup but selected the wrong option.

The winning candidate response made the verification path easier: compute time of flight from vertical motion, compute each velocity component at landing, and then take the norm.

Why judging is easier here: the judge can validate a completed calculation trace instead of performing the full calculation and option selection from scratch.

## Case 5: Chemistry, Osmotic Pressure

- `original_id`: `4382`
- Gold answer: `D`
- Qwen solver prediction: `B`
- Winner final letter: `D`
- Loser final letter: `C`

The solver task involves concentration conversion, van 't Hoff osmotic pressure, Torr conversion, and water-column conversion. Qwen's solver set up the problem but selected the wrong numeric option.

The winning response showed the more consistent dimensional path and ended at the gold option. The loser produced a plausible but wrong numeric pair.

Why judging is easier here: the response comparison highlights which calculation chain is internally consistent; the model does not need to search the full option space alone.

## Takeaway

These cases support a concrete explanation for Qwen3.5 27B's unusually high MMLU judge ability relative to solver ability. It is not simply "better at MMLU"; rather, it appears especially good at recognizing good reasoning when the reasoning is already externalized in candidate answers.

That makes Qwen3.5 27B a useful example of solver-judge dissociation: solving requires generation and option selection, while judging can exploit comparative verification.
