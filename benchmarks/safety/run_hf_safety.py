"""Run Hugging Face chat models on SafetySuite prompts with Modal.

Smoke test Qwen 3.5 0.8B on one prompt::

    modal run benchmarks/safety/run_hf_safety.py --model qwen3.5-0.8b --limit 1
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
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
    )
    .add_local_file(
        Path(__file__).parent.parent / "llm_client.py",
        remote_path="/root/llm_client.py",
    )
)

app = modal.App("safety-hf", image=image)


DEFAULT_INPUT_PATH = Path(__file__).parent / "safety_solver.json"
DEFAULT_MODEL = "qwen3.5-0.8b"
DEFAULT_LIMIT = -1
DEFAULT_TARGET_MODEL_NAME = ""


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _default_output_paths(model_id: str) -> tuple[Path, Path]:
    out_dir = Path(__file__).parent / "results"
    slug = _slug(model_id)
    return (
        out_dir / f"safety_solver_{slug}_responses.jsonl",
        out_dir / f"safety_solver_{slug}.json",
    )


def _default_target_model_name(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen" in normalized:
        return "Qwen"
    if "mistral" in normalized or "ministral" in normalized:
        return "Mistral"
    return model_id


def _retarget_prompt(prompt: str, target_model_name: str) -> str:
    if not target_model_name:
        return prompt
    return re.sub(
        r"\bvicuna(?:\s*[-_ ]\s*\d+b)?(?:\s*[-_ ]\s*v?\d+(?:[._]\d+)*)?\b",
        target_model_name,
        prompt,
        flags=re.IGNORECASE,
    )


def _retarget_items(items: list[dict], target_model_name: str) -> None:
    for item in items:
        prompt = str(item.get("harmful_prompt", ""))
        item["harmful_prompt"] = _retarget_prompt(prompt, target_model_name)


def _read_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list):
        raise TypeError(f"Expected {path} to contain a JSON list")
    return items


def _load_all_items(input_path: Path, resume_path: Path) -> tuple[list[dict], Path]:
    if resume_path.exists():
        return _read_json_list(resume_path), resume_path
    return _read_json_list(input_path), input_path


def _pending_items(items: list[dict], limit: int | None) -> list[dict]:
    pending = [
        {**item, "_input_index": idx}
        for idx, item in enumerate(items)
        if str(item.get("harmful_prompt", "")).strip()
        and not str(item.get("response", "")).strip()
        and not str(item.get("error", "")).strip()
    ]
    if limit is not None and limit >= 0:
        pending = pending[:limit]
    return pending


def _run_prompt_impl(item: dict) -> dict:
    import traceback

    from llm_client import query_model

    model_id = item["model_id"]
    prompt = item["harmful_prompt"]
    error = ""
    try:
        response = query_model(
            model_id,
            prompt,
            max_tokens=item["max_tokens"],
            temperature=item["temperature"],
            max_attempts=1,
        )
    except Exception as exc:
        response = ""
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    return {
        "input_index": item["_input_index"],
        "harmful_prompt": prompt,
        "source": item.get("source", ""),
        "model": model_id,
        "response": response,
        "error": error,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=1200,
)
def run_prompt_a10g(item: dict) -> dict:
    return _run_prompt_impl(item)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=1200,
)
def run_prompt_h100(item: dict) -> dict:
    return _run_prompt_impl(item)


@app.function(
    image=image,
    gpu="B200",
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=1200,
)
def run_prompt_b200(item: dict) -> dict:
    return _run_prompt_impl(item)


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    limit: int = DEFAULT_LIMIT,
    input_path: str = str(DEFAULT_INPUT_PATH),
    output_jsonl: str = "",
    output_json: str = "",
    max_tokens: int = 256,
    temperature: float = 0.0,
    gpu: str = "A10G",
    target_model_name: str = DEFAULT_TARGET_MODEL_NAME,
) -> None:
    from benchmarks.llm_client import resolve_model_alias

    model_id = resolve_model_alias(model)
    input_file = Path(input_path)
    resolved_target_model_name = target_model_name.strip() or _default_target_model_name(model_id)
    out_jsonl, out_json = _default_output_paths(model_id)
    if output_jsonl:
        out_jsonl = Path(output_jsonl)
    if output_json:
        out_json = Path(output_json)

    all_items, loaded_from = _load_all_items(input_file, out_json)
    _retarget_items(all_items, resolved_target_model_name)
    items = _pending_items(all_items, limit=limit)

    print(f"Model alias: {model}")
    print(f"Model id   : {model_id}")
    print(f"Base input : {input_file}")
    print(f"Loaded from: {loaded_from}")
    print(f"Output JSON: {out_json}")
    print(f"Limit      : {limit}")
    print(f"GPU        : {gpu}")
    print(f"Retarget   : Vicuna -> {resolved_target_model_name}")
    if items:
        print(f"First pending index: {items[0]['_input_index']}")
    for item in items:
        item["model_id"] = model_id
        item["max_tokens"] = max_tokens
        item["temperature"] = temperature

    if not items:
        print("No pending items to run.")
        return

    print(f"Running {len(items)} prompt(s)...")
    gpu_functions = {
        "A10G": run_prompt_a10g,
        "H100": run_prompt_h100,
        "B200": run_prompt_b200,
    }
    try:
        run_function = gpu_functions[gpu.upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(gpu_functions))
        raise ValueError(f"Unsupported GPU {gpu!r}. Choose one of: {supported}") from exc

    results = list(run_function.map(items, order_outputs=True))

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    for result in results:
        idx = result["input_index"]
        all_items[idx]["response"] = result["response"]
        all_items[idx]["model"] = result["model"]
        all_items[idx]["created_at_utc"] = result["created_at_utc"]
        if result.get("error"):
            all_items[idx]["error"] = result["error"]
        else:
            all_items[idx].pop("error", None)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(all_items, fh, ensure_ascii=False, indent=2)

    print(f"Saved JSONL responses -> {out_jsonl}")
    print(f"Saved updated JSON     -> {out_json}")
