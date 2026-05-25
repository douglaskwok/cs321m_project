"""KUDGE Pairwise judgment benchmark (Son et al., 2024).

Runs an OpenAI or Anthropic model as a pairwise judge on the ``korean-easy``
and ``korean-hard`` subsets of ``amphora/kudge-challenge``, constructing
prompts in the judge_query format from ``HAERAE-HUB/KUDGE``.  For each item
the ``chosen`` response is placed deterministically as either Response A or
Response B (alternating by index), and the gold winner is set accordingly.
Results are saved as a (1 × n_items) binary correct/incorrect response matrix
together with raw model outputs.

Usage
-----
One-time secret setup in Modal::

    modal secret create openai-secret OPENAI_API_KEY=sk-...
    modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...

Then run::

    modal run benchmarks/kudge_pairwise.py [--model MODEL]

``MODEL`` defaults to ``gpt-5.4-nano``.  Pass any OpenAI or Anthropic model
name — models whose name starts with ``claude-`` are routed to Anthropic
automatically, e.g.::

    modal run benchmarks/kudge_pairwise.py --model claude-haiku-4-5-20251001
    modal run benchmarks/kudge_pairwise.py --model claude-sonnet-4-6
    modal run benchmarks/kudge_pairwise.py --model gpt-4o-mini

Results are written to two files (slug derived from the model name):

* ``benchmarks/results/kudge_pairwise_<slug>.npz`` — response matrix and
  metadata arrays (``response_matrix``, ``item_ids``, ``subsets``, ``gold``,
  ``predicted``).  ``response_matrix`` has shape ``(1, n_items)``, dtype int8.
* ``benchmarks/results/kudge_pairwise_<slug>_responses.jsonl`` — one JSON
  record per item with ``id``, ``subset``, ``gold``, ``predicted``, ``correct``,
  and the full ``raw`` model response.
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

_llm_client = Path(__file__).parent.parent / "llm_client.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.20",
        "openai>=1.30",
        "anthropic>=0.25",
        "numpy>=1.24",
    )
    .add_local_file(_llm_client, remote_path="/root/llm_client.py")
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
    .add_local_file(_llm_client, remote_path="/root/llm_client.py")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("kudge-pairwise", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "amphora/kudge-challenge"
DEFAULT_MODEL = "gpt-5.4-nano"
TEST_LIMIT: int | None = None  # set to a small int to smoke-test


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model.lower())


def _out_paths(model: str) -> tuple[Path, Path]:
    slug = _model_slug(model)
    base = Path(__file__).parent / "results" / "kudge_judge_easy_hard"
    return (
        base / f"kudge_pairwise_{slug}.npz",
        base / f"kudge_pairwise_{slug}_responses.jsonl",
    )


# ---------------------------------------------------------------------------
# Judge prompt template (HAERAE-HUB/KUDGE format)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses "
    "provided by two AI assistants to the user question displayed below. You should "
    "choose the assistant that follows the user's instructions and answers the user's "
    "question better. Your evaluation should consider factors such as the helpfulness, "
    "relevance, accuracy, depth, creativity, and level of detail of their responses. "
    "Begin your evaluation by comparing the two responses and provide a short "
    'explanation. Avoid any position biases and ensure that the order in which the '
    "responses were presented does not influence your decision. Do not allow the length "
    "of the responses to influence your evaluation. Do not favor certain names of the "
    'assistants. Be as objective as possible. After providing your explanation, output '
    'your final verdict by strictly following this format: "[[A]]" if assistant A is '
    'better, "[[B]]" if assistant B is better.'
)


def _build_judge_query(instruction: str, response_a: str, response_b: str) -> str:
    return (
        f"{_JUDGE_SYSTEM}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response A:\n{response_a}\n\n"
        f"### Response B:\n{response_b}"
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


# ---------------------------------------------------------------------------
# Modal function: load dataset
# ---------------------------------------------------------------------------


@app.function(image=image)
def fetch_items() -> list[dict]:
    """Download kudge-challenge and build pairwise judge_query prompts."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")
    all_subsets = sorted({row["subset"] for row in ds})
    print(f"Available subsets: {all_subsets}")

    _SUBSETS = {"Korean-Easy", "Korean-Hard"}
    items = []
    for i, row in enumerate(ds):
        if row["subset"] not in _SUBSETS:
            continue
        # Alternate which response is placed as A to avoid systematic position bias.
        chosen_is_a = (i % 2 == 0)
        response_a = row["chosen"] if chosen_is_a else row["rejected"]
        response_b = row["rejected"] if chosen_is_a else row["chosen"]
        items.append(
            {
                "id": row["id"],
                "subset": row["subset"],
                "judge_query": _build_judge_query(row["prompt"], response_a, response_b),
                "winner": "A" if chosen_is_a else "B",
            }
        )

    items.sort(key=lambda x: (x["subset"], x["id"]))
    if TEST_LIMIT is not None:
        items = items[:TEST_LIMIT]
    return items


# ---------------------------------------------------------------------------
# Modal function: score one item
# ---------------------------------------------------------------------------


def _score_item_impl(item: dict) -> dict:
    from llm_client import query_model

    model = item["model"]
    gold = item["winner"]
    raw = query_model(model, item["judge_query"])
    predicted = _parse_winner(raw)
    correct = int(predicted == gold) if predicted is not None else 0

    return {
        "id": item["id"],
        "subset": item["subset"],
        "gold": gold,
        "predicted": predicted or "",
        "correct": correct,
        "raw": raw,
    }


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
    max_containers=13,
    retries=2,
    timeout=300,
)
def score_item(item: dict) -> dict:
    """Send one pairwise judge query to an API model and score the response."""
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
    """Send one pairwise judge query to a local HF model on A10G and score the response."""
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
    """Send one pairwise judge query to a local HF model on H100 and score the response."""
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
    """Send one pairwise judge query to a local HF model on B200 and score the response."""
    return _score_item_impl(item)


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
    if n == 0:
        print("No items matched the subset filter — check 'Available subsets' output above.")
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
                        "id": r["id"],
                        "subset": r["subset"],
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

    subsets = sorted({r["subset"] for r in results})
    for subset in subsets:
        subset_results = [r for r in results if r["subset"] == subset]
        acc = np.mean([r["correct"] for r in subset_results])
        print(f"{subset:<16}: {acc:.3f}  (n={len(subset_results)})")

    print(f"\nOverall  : {responses.mean():.3f}  ({int(responses.sum())}/{n})")
    print(f"\nSaved → {out_path}")
    print(f"Saved → {out_jsonl}")
