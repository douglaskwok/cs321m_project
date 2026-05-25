"""Shared helper for querying OpenAI, Anthropic, and local HF models."""
from __future__ import annotations

from functools import lru_cache


HF_MODEL_ALIASES = {
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen-3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen35-27b": "Qwen/Qwen3.5-27B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen-3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen35-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen-3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen-3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen35-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen-3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen35-0.8b": "Qwen/Qwen3.5-0.8B",
    "mistral-14b": "mistralai/Ministral-3-14B-Instruct-2512",
    "ministral-14b": "mistralai/Ministral-3-14B-Instruct-2512",
    "mistral-14b-bf16": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
    "ministral-14b-bf16": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
    "mistral-8b": "mistralai/Ministral-3-8B-Instruct-2512",
    "ministral-8b": "mistralai/Ministral-3-8B-Instruct-2512",
    "mistral-8b-bf16": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "ministral-8b-bf16": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "mistral-3b": "mistralai/Ministral-3-3B-Instruct-2512",
    "ministral-3b": "mistralai/Ministral-3-3B-Instruct-2512",
    "mistral-3b-bf16": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "ministral-3b-bf16": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
}


def resolve_model_alias(model: str) -> str:
    """Resolve local shorthand names to provider model ids."""
    cleaned = model.strip()
    return HF_MODEL_ALIASES.get(cleaned.lower(), cleaned)


def _model_provider(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    return "hf"


@lru_cache(maxsize=2)
def _load_hf_model(model: str):
    import torch

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if not torch.cuda.is_available():
        model_kwargs.pop("device_map")

    if model.startswith("Qwen/Qwen3.5-"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        hf_model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)
        hf_model.eval()
        return "image_text_to_text", processor, hf_model

    if model.startswith("mistralai/Ministral-3-"):
        from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration, MistralCommonBackend

        tokenizer = MistralCommonBackend.from_pretrained(model)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            model_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        hf_model = Mistral3ForConditionalGeneration.from_pretrained(model, **model_kwargs)
        hf_model.eval()
        return "mistral3", tokenizer, hf_model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    hf_model.eval()
    return "causal_lm", tokenizer, hf_model


def _query_hf_model(
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    import torch

    model_type, processor_or_tokenizer, hf_model = _load_hf_model(model)
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    if model_type == "image_text_to_text":
        processor = processor_or_tokenizer
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(hf_model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = hf_model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

    if model_type == "mistral3":
        tokenizer = processor_or_tokenizer
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )
        device = next(hf_model.parameters()).device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16, device=device)
            generation_kwargs["image_sizes"] = [inputs["pixel_values"].shape[-2:]]

        with torch.inference_mode():
            output_ids = hf_model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    tokenizer = processor_or_tokenizer
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {"input_ids": input_ids}
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    device = next(hf_model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = hf_model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def query_model(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_attempts: int = 6,
) -> str:
    """Query a model with exponential-backoff retry.

    Routes to Anthropic for ``claude-*``, OpenAI for known OpenAI prefixes,
    and Hugging Face Transformers otherwise. Raises the last exception if all
    attempts fail.
    """
    import random
    import time

    model = resolve_model_alias(model)
    provider = _model_provider(model)
    rate_limit_error_types: tuple[type[Exception], ...] = ()
    if provider == "anthropic":
        from anthropic import RateLimitError as _RateLimitError

        rate_limit_error_types = (_RateLimitError,)
    elif provider == "openai":
        from openai import RateLimitError as _RateLimitError

        rate_limit_error_types = (_RateLimitError,)

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            if provider == "anthropic":
                from anthropic import Anthropic

                client = Anthropic()
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if model != "claude-opus-4-7":
                    kwargs["temperature"] = temperature
                resp = client.messages.create(**kwargs)
                text_parts = [
                    block.text
                    for block in resp.content
                    if getattr(block, "type", None) == "text" and getattr(block, "text", "")
                ]
                if text_parts:
                    return "\n".join(text_parts)
                stop_reason = getattr(resp, "stop_reason", None)
                raise RuntimeError(f"Anthropic returned no text content; stop_reason={stop_reason}")
            if provider == "openai":
                from openai import OpenAI

                client = OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            return _query_hf_model(
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except rate_limit_error_types as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = min(2**attempt * 5, 30) + random.uniform(0, 2)
                time.sleep(delay)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(2**attempt)

    raise last_exc  # type: ignore[misc]
