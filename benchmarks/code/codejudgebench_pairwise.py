"""CodeJudgeBench pairwise code judgment benchmark (Zhang et al., 2025).

Runs an OpenAI or Anthropic model as a pairwise judge on the code generation
subset of CodeJudgeBench, which contains 2,103 pairwise comparisons built
from LiveCodeBench v6.  Each pair is evaluated twice (with candidate positions
swapped) to mitigate position bias; accuracy is the average correctness across
both orderings.

Usage
-----
One-time secret setup in Modal::

    modal secret create openai-secret OPENAI_API_KEY=sk-...
    modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...

Then run::

    modal run benchmarks/code/codejudgebench_pairwise.py [--model MODEL]

``MODEL`` defaults to ``gpt-5.4-nano``.  Pass any OpenAI or Anthropic model
name — models whose name starts with ``claude-`` are routed to Anthropic
automatically, e.g.::

    modal run benchmarks/code/codejudgebench_pairwise.py --model claude-sonnet-4-6
    modal run benchmarks/code/codejudgebench_pairwise.py --model gpt-4o-mini

Results are written to two files (slug derived from the model name):

* ``benchmarks/code/results/codejudgebench_pairwise_<slug>.npz`` — response
  matrix and metadata (``response_matrix``, ``pair_ids``, ``difficulties``,
  ``gold``, ``predicted``, ``orderings``).  ``response_matrix`` has shape
  ``(1, 2 * n_pairs)``, dtype int8, with forward and backward orderings
  interleaved as [fwd_0, bwd_0, fwd_1, bwd_1, ...].
* ``benchmarks/code/results/codejudgebench_pairwise_<slug>_responses.jsonl`` —
  one JSON record per judgment with ``pair_id``, ``ordering``, ``difficulty``,
  ``gold``, ``predicted``, ``correct``, and the full ``raw`` model response.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal
import numpy as np

# ---------------------------------------------------------------------------
# Modal image & app
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.20",
        "openai>=1.30",
        "anthropic>=0.25",
        "numpy>=1.24",
    )
    .add_local_file(
        Path(__file__).parent.parent / "llm_client.py",
        remote_path="/root/llm_client.py",
    )
)

app = modal.App("codejudgebench-pairwise", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dataset: CodeJudgeBench code-generation subset (Zhang et al., 2025).
DATASET_ID = "livecodebench/code_judge_bench"
DEFAULT_MODEL = "gpt-5.4-nano"
TEST_LIMIT: int | None = 2  # set to None to run all 2,103 pairs


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def _out_paths(model: str) -> tuple[Path, Path]:
    slug = _model_slug(model)
    base = Path(__file__).parent / "results"
    return (
        base / f"codejudgebench_pairwise_{slug}.npz",
        base / f"codejudgebench_pairwise_{slug}_responses.jsonl",
    )


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an expert competitive programming judge. "
    "Given a problem and two candidate solutions, determine which solution is "
    "more likely to be correct (pass all test cases). "
    "Briefly explain your reasoning, then output your verdict as [[A]] if "
    "Solution A is more likely correct, or [[B]] if Solution B is more likely correct."
)


def _build_judge_query(problem: str, solution_a: str, solution_b: str) -> str:
    return (
        f"{_JUDGE_SYSTEM}\n\n"
        f"### Problem:\n{problem}\n\n"
        f"### Solution A:\n```python\n{solution_a}\n```\n\n"
        f"### Solution B:\n```python\n{solution_b}\n```"
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_BRACKET2_RE = re.compile(r"\[\[([AB])\]\]", re.IGNORECASE)
_BRACKET1_RE = re.compile(r"\[([AB])\]", re.IGNORECASE)
_BARE_AB_RE = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")


def _parse_winner(text: str) -> str | None:
    """Extract [[A]] or [[B]] verdict from model response."""
    for pattern in (_BRACKET2_RE, _BRACKET1_RE):
        m = pattern.search(text)
        if m:
            return m.group(1).upper()
    matches = _BARE_AB_RE.findall(text)
    return matches[-1].upper() if matches else None


# ---------------------------------------------------------------------------
# Modal function: load dataset and expand into forward+backward items
# ---------------------------------------------------------------------------


@app.function(image=image)
def fetch_items() -> list[dict]:
    """Download CodeJudgeBench and build forward + backward judgment items.

    Returns a flat list where every original pair produces two items:
    ordering="fwd"  → solution_correct placed as A, solution_incorrect as B
    ordering="bwd"  → positions swapped; gold answer flipped accordingly
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="test", trust_remote_code=True)
    all_columns = ds.column_names
    print(f"Dataset columns: {all_columns}")

    pairs: list[dict] = []
    for row in ds:
        # Adapt field names to whatever the dataset actually uses.
        pair_id = str(row.get("id", row.get("pair_id", row.get("question_id", len(pairs)))))
        problem = row.get("question_content", row.get("problem", row.get("prompt", "")))
        # Prefer fields named solution_{correct,incorrect}; fall back to generic 1/2 or A/B.
        sol_correct = row.get(
            "solution_correct",
            row.get("solution_1", row.get("solution_a", row.get("chosen", ""))),
        )
        sol_incorrect = row.get(
            "solution_incorrect",
            row.get("solution_2", row.get("solution_b", row.get("rejected", ""))),
        )
        difficulty = str(row.get("difficulty", "unknown"))
        pairs.append(
            {
                "pair_id": pair_id,
                "difficulty": difficulty,
                "problem": problem,
                "sol_correct": sol_correct,
                "sol_incorrect": sol_incorrect,
            }
        )

    pairs.sort(key=lambda x: x["pair_id"])
    if TEST_LIMIT is not None:
        pairs = pairs[:TEST_LIMIT]

    items: list[dict] = []
    for pair in pairs:
        # Forward: correct=A, incorrect=B → gold=A
        items.append(
            {
                "pair_id": pair["pair_id"],
                "ordering": "fwd",
                "difficulty": pair["difficulty"],
                "judge_query": _build_judge_query(
                    pair["problem"], pair["sol_correct"], pair["sol_incorrect"]
                ),
                "gold": "A",
            }
        )
        # Backward: correct=B, incorrect=A → gold=B
        items.append(
            {
                "pair_id": pair["pair_id"],
                "ordering": "bwd",
                "difficulty": pair["difficulty"],
                "judge_query": _build_judge_query(
                    pair["problem"], pair["sol_incorrect"], pair["sol_correct"]
                ),
                "gold": "B",
            }
        )

    print(f"Built {len(pairs)} pairs → {len(items)} judgment items")
    return items


# ---------------------------------------------------------------------------
# Modal function: score one item
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    max_containers=20,
    retries=2,
    timeout=300,
)
def score_item(item: dict) -> dict:
    """Send one judge query to the model and score the response."""
    from llm_client import query_model

    model = item["model"]
    gold = item["gold"]
    raw = query_model(model, item["judge_query"])
    predicted = _parse_winner(raw)
    correct = int(predicted == gold) if predicted is not None else 0

    return {
        "pair_id": item["pair_id"],
        "ordering": item["ordering"],
        "difficulty": item["difficulty"],
        "gold": gold,
        "predicted": predicted or "",
        "correct": correct,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL) -> None:
    out_path, out_jsonl = _out_paths(model)
    print(f"Model: {model}  (slug: {_model_slug(model)})")

    items = fetch_items.remote()
    for item in items:
        item["model"] = model
    n = len(items)
    n_pairs = n // 2
    print(f"Loaded {n_pairs} pairs → {n} judgment items")
    if n == 0:
        print("No items loaded — check DATASET_ID and column names printed above.")
        return

    results = list(score_item.map(items, order_outputs=True))

    responses = np.array([r["correct"] for r in results], dtype=np.int8)
    response_matrix = responses.reshape(1, -1)  # (1, 2 * n_pairs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        response_matrix=response_matrix,
        pair_ids=np.array([r["pair_id"] for r in results]),
        difficulties=np.array([r["difficulty"] for r in results]),
        gold=np.array([r["gold"] for r in results]),
        predicted=np.array([r["predicted"] for r in results]),
        orderings=np.array([r["ordering"] for r in results]),
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "pair_id": r["pair_id"],
                        "ordering": r["ordering"],
                        "difficulty": r["difficulty"],
                        "gold": r["gold"],
                        "predicted": r["predicted"],
                        "correct": r["correct"],
                        "raw": r["raw"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Per-difficulty accuracy (averaging fwd+bwd within each pair).
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"].lower() == diff]
        if subset:
            acc = np.mean([r["correct"] for r in subset])
            print(f"{diff:<8}: {acc:.3f}  (n={len(subset)} judgments, {len(subset)//2} pairs)")

    print(f"\nOverall  : {responses.mean():.3f}  ({int(responses.sum())}/{n} judgments)")
    print(f"\nSaved → {out_path}")
    print(f"Saved → {out_jsonl}")
