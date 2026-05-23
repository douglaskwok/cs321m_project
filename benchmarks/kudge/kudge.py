"""KUDGE Challenge (Son et al., 2024) — Korean portion response matrix.

Runs an OpenAI model on all Korean-Easy and Korean-Hard items from
``amphora/kudge-challenge`` and saves a (1 × n_items) binary
correct/incorrect response matrix.

Usage
-----
First, create the OpenAI secret in Modal (one-time setup)::

    modal secret create openai-secret OPENAI_API_KEY=sk-...

Then run::

    modal run benchmarks/kudge.py [--model MODEL]

``MODEL`` defaults to ``gpt-5.4-nano``.  Pass any OpenAI model name, e.g.::

    modal run benchmarks/kudge.py --model gpt-4o-mini

Results are written to two files (slug derived from the model name):

* ``benchmarks/results/kudge_<slug>.npz`` — response matrix and metadata
  arrays (``response_matrix``, ``item_ids``, ``subsets``, ``gold``,
  ``predicted``).  ``response_matrix`` has shape ``(1, n_items)``, dtype int8.
* ``benchmarks/results/kudge_<slug>_responses.jsonl`` — one JSON record per
  item with ``id``, ``subset``, ``gold``, ``predicted``, ``correct``, and the
  full ``raw`` model response.
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
        "numpy>=1.24",
    )
    .add_local_file(
        Path(__file__).parent.parent / "llm_client.py",
        remote_path="/root/llm_client.py",
    )
)

app = modal.App("kudge", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "amphora/kudge-challenge"
DEFAULT_MODEL = "gpt-5.4-nano"
KOREAN_SUBSETS = frozenset({"Korean-Easy"})
TEST_LIMIT: int | None = None  # set to None to run all items


def _model_slug(model: str) -> str:
    """Convert a model name to a filesystem-safe slug, e.g. gpt-5.4-nano → gpt54nano."""
    return re.sub(r"[^a-z0-9]", "", model.lower())


def _out_paths(model: str) -> tuple[Path, Path]:
    slug = _model_slug(model)
    base = Path(__file__).parent / "results"
    return base / f"kudge_{slug}.npz", base / f"kudge_{slug}_responses.jsonl"

# ---------------------------------------------------------------------------
# Helpers (pure Python, called both locally and inside Modal containers)
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"\[ANSWER\]\s*([A-Da-d])")
_BARE_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])")


def _parse_letter(text: str) -> str | None:
    """Extract the answer letter from a model response."""
    # Prefer [ANSWER] X [END] format; fall back to last bare a/b/c/d in text.
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    matches = _BARE_LETTER_RE.findall(text)
    return matches[-1].upper() if matches else None


def _gold_letter(chosen: str) -> str:
    """Extract the correct answer letter from the `chosen` reference solution."""
    m = _ANSWER_RE.search(chosen)
    return m.group(1).upper() if m else ""


# ---------------------------------------------------------------------------
# Modal function: load dataset (runs in cloud; needs `datasets` package)
# ---------------------------------------------------------------------------

@app.function(image=image)
def fetch_items() -> list[dict]:
    """Download KUDGE Challenge and return all Korean-subset rows."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")
    items = [
        {
            "id": row["id"],
            "subset": row["subset"],
            "prompt": row["prompt"],
            "chosen": row["chosen"],
        }
        for row in ds
        if row["subset"] in KOREAN_SUBSETS
    ]
    items.sort(key=lambda x: (x["subset"], int(x["id"])))
    if TEST_LIMIT is not None:
        items = items[:TEST_LIMIT]
    return items


# ---------------------------------------------------------------------------
# Modal function: score one item (runs in cloud; needs `openai` package)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    retries=2,
    timeout=120,
)
def score_item(item: dict) -> dict:
    """Query an OpenAI model on one KUDGE item; return correctness metadata."""
    from llm_client import query_model

    gold = _gold_letter(item["chosen"])
    model = item["model"]

    # IMPORTANT: this prompting strategy is not directly from the paper, but
    # the original prompts didn't specify the answer format which is needed
    prompt = item["prompt"] + (
        "\n풀이를 마친 후 반드시 '[ANSWER] (a/b/c/d) [END]' 형식으로 최종 답을 표시하세요."
    )

    raw = query_model(model, prompt, max_tokens=2048, max_attempts=3)
    predicted = _parse_letter(raw)
    correct = int(predicted == gold) if (predicted and gold) else 0

    return {
        "id": item["id"],
        "subset": item["subset"],
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
    print(f"Loaded {n} Korean items  "
          f"(Easy={sum(1 for x in items if x['subset']=='Korean-Easy')}, "
          f"Hard={sum(1 for x in items if x['subset']=='Korean-Hard')})")

    results = list(score_item.map(items, order_outputs=True))

    responses = np.array([r["correct"] for r in results], dtype=np.int8)
    response_matrix = responses.reshape(1, -1)  # (1, n_items)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        response_matrix=response_matrix,
        item_ids=np.array([r["id"] for r in results]),
        subsets=np.array([r["subset"] for r in results]),
        gold=np.array([r["gold"] for r in results]),
        predicted=np.array([r["predicted"] for r in results]),
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps({
                "id": r["id"],
                "subset": r["subset"],
                "gold": r["gold"],
                "predicted": r["predicted"],
                "correct": r["correct"],
                "raw": r["raw"],
            }, ensure_ascii=False) + "\n")

    easy = [r for r in results if r["subset"] == "Korean-Easy"]
    hard = [r for r in results if r["subset"] == "Korean-Hard"]

    print(f"\nOverall  : {responses.mean():.3f}  ({int(responses.sum())}/{n})")
    if easy:
        print(f"Easy     : {np.mean([r['correct'] for r in easy]):.3f}  (n={len(easy)})")
    if hard:
        print(f"Hard     : {np.mean([r['correct'] for r in hard]):.3f}  (n={len(hard)})")
    print(f"\nSaved → {out_path}")
    print(f"Saved → {out_jsonl}")
