# HarmBench Text vs Multimodal Overlap

Compared files:

- `HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv`
- `HarmBench/data/behavior_datasets/harmbench_behaviors_multimodal_all.csv`

Counts:

- Text behaviors: 400 rows
- Multimodal behaviors: 110 rows
- Unique text `BehaviorID`: 400
- Unique multimodal `BehaviorID`: 110

Overlap:

- Overlapping `BehaviorID`: 0
- Exact overlapping `Behavior` text: 0
- Normalized overlapping `Behavior` text: 0

Output CSV:

- `benchmarks/safety/harmbench_text_multimodal_overlap.csv`

Notes:

- The CSV intentionally reports IDs/categories instead of full behavior text.
- `NormalizedBehavior` lowercases text, collapses whitespace, and removes punctuation before comparing.
