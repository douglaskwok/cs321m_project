"""CodeJudgeBench pairwise code judgment benchmark (Zhang et al., 2025).

Runs an OpenAI or Anthropic model as a pairwise judge on the codegen subset
of CodeJudgeBench (mattymchen/codejudgebench).  Uses the 676-pair curated
subset selected by select_codegen_subset.py — one pair per unique question_id,
stratified evenly across the claude, qwen, and gemini model families.  Each
pair is evaluated twice (with candidate positions swapped) to mitigate
position bias; accuracy is the average correctness across both orderings.

Prerequisites
-------------
Run select_codegen_subset.py once to generate the subset file::

    python benchmarks/code/select_codegen_subset.py

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
        "datasets>=2.20,<3.0",
        "openai>=1.30",
        "anthropic>=0.25",
        "numpy>=1.24",
    )
    .add_local_file(
        Path(__file__).parent.parent / "llm_client.py",
        remote_path="/root/llm_client.py",
    )
)

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers==5.9.0",
        "accelerate>=1.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "torchvision",
        "pillow",
        "mistral-common>=1.8.6",
        "kernels",
        "numpy>=1.24",
    )
    .add_local_file(
        Path(__file__).parent.parent / "llm_client.py",
        remote_path="/root/llm_client.py",
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("codejudgebench-pairwise", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "mattymchen/codejudgebench"
CODEGEN_CONFIG = "codegen"
DEFAULT_MODEL = "gpt-5.4-nano"
TEST_LIMIT: int | None = None  # set to None to run all 676 pairs
SELECTED_PAIRS_FILE = Path(__file__).parent / "codegen_selected_pairs.json"
_TARGET_FAMILIES = ["claude", "qwen", "gemini"]


def _detect_family(model_str: str) -> str | None:
    s = model_str.lower()
    for f in _TARGET_FAMILIES:
        if f in s:
            return f
    return None


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


def _build_judge_messages(problem: str, solution_a: str, solution_b: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert competitive programming judge. "
                "Your entire response must be exactly two lines — no more, no less:\n"
                "Line 1: One sentence identifying the key difference between the solutions.\n"
                "Line 2: Verdict: [[A]] or Verdict: [[B]]\n"
                "Do not add any other text, headers, or explanation."
            )
        },
        {
            "role": "user",
            "content": (
                f"### Problem:\n{problem}\n\n"
                f"### Solution A:\n```python\n{solution_a}\n```\n\n"
                f"### Solution B:\n```python\n{solution_b}\n```"
            )
        },
    ]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_BRACKET2_RE = re.compile(r"\[\[([AB])\]\]", re.IGNORECASE)
_BRACKET1_RE = re.compile(r"\[([AB])\]", re.IGNORECASE)


def _parse_winner(text: str) -> str | None:
    # Use rfind so the final verdict wins when both preliminary and final are present;
    # falls back to the preliminary verdict if the response was truncated mid-reasoning.
    pred_a = max(text.rfind(p) for p in ("[[A]]", "[A]"))
    pred_b = max(text.rfind(p) for p in ("[[B]]", "[B]"))
    if pred_a > pred_b:
        return "A"
    if pred_b > pred_a:
        return "B"
    return None


# ---------------------------------------------------------------------------
# Modal function: load dataset and expand into forward+backward items
# ---------------------------------------------------------------------------


@app.function(image=image)
def fetch_items(selected_pairs: list[dict]) -> list[dict]:
    """Download the codegen subset and build forward + backward judgment items.

    selected_pairs: output of select_codegen_subset.py — one entry per
    question_id with keys question_id, split, model, model_family.

    Returns a flat list where every original pair produces two items:
    ordering="fwd"  → solution_correct placed as A, solution_incorrect as B
    ordering="bwd"  → positions swapped; gold answer flipped accordingly
    """
    from datasets import load_dataset

    # Build lookup: question_id -> required model_family.
    selection_map: dict[str, str] = {
        p["question_id"]: p["model_family"] for p in selected_pairs
    }

    ds = load_dataset(DATASET_ID, CODEGEN_CONFIG, trust_remote_code=True)
    print(f"Splits: {list(ds.keys())}")
    first_split = next(iter(ds.values()))
    print(f"Dataset columns: {first_split.column_names}")

    pairs: list[dict] = []
    seen: set[str] = set()

    for split_ds in ds.values():
        for row in split_ds:
            qid = str(row.get("question_id", row.get("id", "")))
            if qid not in selection_map or qid in seen:
                continue
            model_str = str(row.get("model", row.get("generator", "")))
            if _detect_family(model_str) != selection_map[qid]:
                continue
            seen.add(qid)
            problem = row.get("question_content", row.get("problem", row.get("prompt", "")))
            sol_correct = row.get(
                "pos_response",
                row.get("solution_correct",
                row.get("solution_1", row.get("solution_a", row.get("chosen", "")))),
            )
            sol_incorrect = row.get(
                "neg_response",
                row.get("solution_incorrect",
                row.get("solution_2", row.get("solution_b", row.get("rejected", "")))),
            )
            difficulty = str(row.get("difficulty", "unknown"))
            pairs.append(
                {
                    "pair_id": qid,
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
    for i, pair in enumerate(pairs):
        if i % 2 == 0:
            # Forward: correct=A, incorrect=B → gold=A
            items.append(
                {
                    "pair_id": pair["pair_id"],
                    "ordering": "fwd",
                    "difficulty": pair["difficulty"],
                    "judge_messages": _build_judge_messages(
                        pair["problem"], pair["sol_correct"], pair["sol_incorrect"]
                    ),
                    "gold": "A",
                }
            )
        else:
            # Backward: correct=B, incorrect=A → gold=B
            items.append(
                {
                    "pair_id": pair["pair_id"],
                    "ordering": "bwd",
                    "difficulty": pair["difficulty"],
                    "judge_messages": _build_judge_messages(
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


def _score_item_impl(item: dict) -> dict:
    from llm_client import query_model

    model = item["model"]
    gold = item["gold"]
    judge_messages = item["judge_messages"]

    # Extract assistant prefill text (if any) so we can reconstruct the full
    # response — Anthropic and HF return only the continuation, not the prefix.
    prefill = ""
    if judge_messages and judge_messages[-1]["role"] == "assistant":
        prefill = judge_messages[-1]["content"]

    raw = query_model(model, messages=judge_messages)
    full_text = prefill + raw
    predicted = _parse_winner(full_text)
    correct = int(predicted == gold) if predicted is not None else -1

    return {
        "pair_id": item["pair_id"],
        "ordering": item["ordering"],
        "difficulty": item["difficulty"],
        "gold": gold,
        "predicted": predicted or "",
        "correct": correct,
        "raw": full_text,
    }


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
    return _score_item_impl(item)


@app.function(
    image=hf_image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-secret")],
    retries=2,
    timeout=1200,
)
def score_item_hf_a10g(item: dict) -> dict:
    """Send one judge query to a local HF model on A10G and score the response."""
    return _score_item_impl(item)


@app.function(
    image=hf_image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-secret")],
    retries=2,
    timeout=1200,
)
def score_item_hf_h100(item: dict) -> dict:
    """Send one judge query to a local HF model on H100 and score the response."""
    return _score_item_impl(item)


@app.function(
    image=hf_image,
    gpu="B200",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-secret")],
    retries=2,
    timeout=1200,
)
def score_item_hf_b200(item: dict) -> dict:
    """Send one judge query to a local HF model on B200 and score the response."""
    return _score_item_impl(item)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL) -> None:
    out_path, out_jsonl = _out_paths(model)
    print(f"Model: {model}  (slug: {_model_slug(model)})")

    if not SELECTED_PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"{SELECTED_PAIRS_FILE} not found — run select_codegen_subset.py first."
        )
    with open(SELECTED_PAIRS_FILE, encoding="utf-8") as f:
        selected_pairs = json.load(f)
    print(f"Loaded {len(selected_pairs)} selected pairs from {SELECTED_PAIRS_FILE.name}")

    items = fetch_items.remote(selected_pairs)
    for item in items:
        item["model"] = model
    n = len(items)
    n_pairs = n // 2
    print(f"Loaded {n_pairs} pairs → {n} judgment items")
    if n == 0:
        print("No items loaded — check DATASET_ID and column names printed above.")
        return

    resolved = re.sub(r"[^a-z0-9.\-]", "", model.lower())
    is_api = resolved.startswith(("gpt-", "o1", "o2", "o3", "o4", "chatgpt-", "claude-"))
    is_qwen = resolved.startswith("qwen")
    is_mistral = resolved.startswith("mistral") or resolved.startswith("ministral")
    _param_m = re.search(r"(\d+(?:\.\d+)?)b", resolved)
    is_large_hf = _param_m is not None and float(_param_m.group(1)) >= 20
    if is_api:
        scorer = score_item
    elif is_qwen:
        scorer = score_item_hf_b200
    elif is_mistral or is_large_hf:
        scorer = score_item_hf_h100
    else:
        scorer = score_item_hf_a10g

    results = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in scorer.map(items, order_outputs=False):
            results.append(r)
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
            fh.flush()

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

    # Per-difficulty accuracy (averaging fwd+bwd within each pair).
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"].lower() == diff]
        if subset:
            acc = np.mean([r["correct"] for r in subset])
            print(f"{diff:<8}: {acc:.3f}  (n={len(subset)} judgments, {len(subset)//2} pairs)")

    print(f"\nOverall  : {responses.mean():.3f}  ({int(responses.sum())}/{n} judgments)")
    print(f"\nSaved → {out_path}")
    print(f"Saved → {out_jsonl}")
