"""Shared helper for querying OpenAI and Anthropic models with retry."""
from __future__ import annotations


def query_model(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_attempts: int = 6,
) -> str:
    """Query an OpenAI or Anthropic model with exponential-backoff retry.

    Routes to Anthropic for models whose name starts with ``claude-``,
    OpenAI otherwise.  Raises the last exception if all attempts fail.
    """
    import random
    import time

    use_anthropic = model.startswith("claude-")
    if use_anthropic:
        from anthropic import RateLimitError as _RateLimitError
    else:
        from openai import RateLimitError as _RateLimitError

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            if use_anthropic:
                from anthropic import Anthropic

                client = Anthropic()
                resp = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            else:
                from openai import OpenAI

                client = OpenAI()
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
        except _RateLimitError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = min(2**attempt * 5, 30) + random.uniform(0, 2)
                time.sleep(delay)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(2**attempt)

    raise last_exc  # type: ignore[misc]
