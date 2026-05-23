"""KUDGE Pairwise judgment benchmark (Son et al., 2024).

Runs an OpenAI model as a pairwise judge on all items from
``HAERAE-HUB/KUDGE`` (config ``Pairwise-False``) and saves a
(1 × n_items) binary correct/incorrect response matrix together with
raw model outputs.

Usage
-----
First, create the OpenAI secret in Modal (one-time setup)::

    modal secret create openai-secret OPENAI_API_KEY=sk-...

Then run::

    modal run benchmarks/kudge_pairwise.py [--model MODEL]

``MODEL`` defaults to ``gpt-5.4-nano``.  Pass any OpenAI model name, e.g.::

    modal run benchmarks/kudge_pairwise.py --model gpt-4o-mini

Results are written to two files (slug derived from the model name):

* ``benchmarks/results/kudge_pairwise_<slug>.npz`` — response matrix and
  metadata arrays (``response_matrix``, ``item_ids``, ``gold``,
  ``predicted``).  ``response_matrix`` has shape ``(1, n_items)``, dtype int8.
* ``benchmarks/results/kudge_pairwise_<slug>_responses.jsonl`` — one JSON
  record per item with ``id``, ``gold``, ``predicted``, ``correct``, and the
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

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets>=2.20",
    "openai>=1.30",
    "numpy>=1.24",
)

app = modal.App("kudge-pairwise", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "HAERAE-HUB/KUDGE"
DATASET_CONFIG = "Pairwise"
DEFAULT_MODEL = "gpt-5.4-nano"
TEST_LIMIT: int | None = None  # set to a small int to smoke-test


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def _out_paths(model: str) -> tuple[Path, Path]:
    slug = _model_slug(model)
    base = Path(__file__).parent / "results"
    return (
        base / f"kudge_pairwise_{slug}.npz",
        base / f"kudge_pairwise_{slug}_responses.jsonl",
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches [[A]], [[B]], [[tie]] (Alpaca-eval / MT-Bench style)
_BRACKET2_RE = re.compile(r"\[\[([ABCab티tie]+)\]\]", re.IGNORECASE)
# Matches [A], [B], [tie]
_BRACKET1_RE = re.compile(r"\[([ABab티tie]+)\]", re.IGNORECASE)
# Bare A or B at the very end of a line / response
_BARE_AB_RE = re.compile(r"(?<![A-Za-z])([ABab])(?![A-Za-z])")


def _parse_winner(text: str) -> str | None:
    """Extract the judge's winner choice from the model's response.

    Returns a normalised label in the same form as the dataset's ``winner``
    field (upper-case ``"A"``, ``"B"``, or ``"tie"``), or ``None`` if
    nothing could be extracted.
    """
    for pattern in (_BRACKET2_RE, _BRACKET1_RE):
        m = pattern.search(text)
        if m:
            label = m.group(1).upper()
            if label in ("TIE", "C"):
                return "tie"
            if label in ("A", "B"):
                return label
    # Fall back to the last bare A/B in the text.
    matches = _BARE_AB_RE.findall(text)
    if matches:
        return matches[-1].upper()
    # Check for Korean "동점" (tie) as a last resort.
    if "동점" in text or "무승부" in text:
        return "tie"
    return None


def _normalise_gold(raw_winner: str) -> str:
    """Normalise the dataset's ``winner`` field to 'A', 'B', or 'tie'."""
    w = str(raw_winner).strip().upper()
    if w in ("TIE", "C", "NONE", ""):
        return "tie"
    if w in ("A", "MODEL_A", "MODELA"):
        return "A"
    if w in ("B", "MODEL_B", "MODELB"):
        return "B"
    return w  # pass through anything unexpected


# ---------------------------------------------------------------------------
# Modal function: load dataset
# ---------------------------------------------------------------------------


@app.function(image=image)
def fetch_items() -> list[dict]:
    """Download KUDGE Pairwise-False and return all rows."""
    from datasets import get_dataset_split_names, load_dataset

    splits = get_dataset_split_names(DATASET_ID, DATASET_CONFIG)
    split = splits[0]  # typically "test"

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    items = [
        {
            "id": str(i),
            "judge_query": row["judge_query"],
            "winner": _normalise_gold(row["winner"]),
        }
        for i, row in enumerate(ds)
    ]
    if TEST_LIMIT is not None:
        items = items[:TEST_LIMIT]
    return items


# ---------------------------------------------------------------------------
# Modal function: score one item
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    max_containers=13,
    retries=2,
    timeout=300,
)
def score_item(item: dict) -> dict:
    """Send one pairwise judge query to the model and score the response."""
    import time

    from openai import OpenAI

    client = OpenAI()
    model = item["model"]
    gold = item["winner"]

    import random

    from openai import RateLimitError

    raw = ""
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": item["judge_query"]}],
                max_completion_tokens=1024,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            last_exc = None
            break
        except RateLimitError as exc:
            last_exc = exc
            if attempt < 5:
                delay = min(2 ** attempt * 5, 30) + random.uniform(0, 2)
                time.sleep(delay)
        except Exception as exc:
            last_exc = exc
            if attempt < 5:
                time.sleep(2 ** attempt)
    if last_exc is not None:
        raise last_exc

    predicted = _parse_winner(raw)
    correct = int(predicted == gold) if predicted is not None else 0

    return {
        "id": item["id"],
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
    print(f"Loaded {n} pairwise items")

    results = list(score_item.map(items, order_outputs=True))

    responses = np.array([r["correct"] for r in results], dtype=np.int8)
    response_matrix = responses.reshape(1, -1)  # (1, n_items)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        response_matrix=response_matrix,
        item_ids=np.array([r["id"] for r in results]),
        gold=np.array([r["gold"] for r in results]),
        predicted=np.array([r["predicted"] for r in results]),
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "gold": r["gold"],
                        "predicted": r["predicted"],
                        "correct": r["correct"],
                        "raw": r["raw"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    gold_labels = [r["gold"] for r in results]
    for label in sorted(set(gold_labels)):
        subset = [r for r in results if r["gold"] == label]
        acc = np.mean([r["correct"] for r in subset])
        print(f"Gold={label:<4}: {acc:.3f}  (n={len(subset)})")

    print(f"\nOverall  : {responses.mean():.3f}  ({int(responses.sum())}/{n})")
    print(f"\nSaved → {out_path}")
    print(f"Saved → {out_jsonl}")
