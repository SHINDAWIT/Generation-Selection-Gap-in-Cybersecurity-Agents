"""Small stdlib client for a local OpenAI-compatible llama.cpp server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatResponse:
    content: str
    raw: dict[str, Any]
    elapsed_seconds: float


class ChatClientError(RuntimeError):
    pass


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ChatClientError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ChatClientError(str(exc)) from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ChatClientError(f"server returned invalid JSON: {body[:500]}") from exc
    if not isinstance(parsed, dict):
        raise ChatClientError("server returned a non-object response")
    return parsed


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    seed: int,
    max_tokens: int = 768,
    temperature: float = 1.0,
    top_p: float = 0.95,
    timeout: float = 600.0,
    low_effort: bool = True,
    enable_thinking: bool = True,
) -> ChatResponse:
    payload = {
        "model": model,
        "messages": messages,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "low_effort": low_effort,
        },
    }
    start = time.monotonic()
    raw = _post_json(
        f"{base_url.rstrip('/')}/chat/completions", payload, timeout=timeout
    )
    elapsed = time.monotonic() - start
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatClientError(f"unexpected completion schema: {raw}") from exc
    if not isinstance(content, str):
        raise ChatClientError("completion content is not text")
    return ChatResponse(content=content, raw=raw, elapsed_seconds=elapsed)


def healthcheck(base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/').removesuffix('/v1')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # surfaced with context to the CLI
        raise ChatClientError(f"health check failed for {url}: {exc}") from exc
