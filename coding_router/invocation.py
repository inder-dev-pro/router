"""Minimal provider adapters for invoking the model selected by the router."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .catalog import ModelProfile


class ModelInvocationError(RuntimeError):
    pass


SERVICE_KEY_ENV = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "xAI": "XAI_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
    "Alibaba Cloud (Qwen)": "DASHSCOPE_API_KEY",
    "Mistral AI": "MISTRAL_API_KEY",
    "Z.ai (Zhipu)": "ZAI_API_KEY",
    "Moonshot AI": "MOONSHOT_API_KEY",
    "MiniMax": "MINIMAX_API_KEY",
    "Self-hosted (Ollama / vLLM)": "LOCAL_LLM_API_KEY",
}


def _key_for(profile: ModelProfile) -> str | None:
    # A model-specific key supports proxies while the normal service key supports
    # direct provider access. No key is required for a local unauthenticated server.
    per_model = "MODEL_ROUTER_" + profile.catalog_key.upper().replace("-", "_") + "_API_KEY"
    return (
        os.getenv(per_model)
        or (os.getenv(profile.api_key_env) if profile.api_key_env else None)
        or os.getenv(SERVICE_KEY_ENV.get(profile.service, ""))
    )


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        detail = ""
        if isinstance(error, urllib.error.HTTPError):
            detail = error.read().decode("utf-8", errors="replace")[:500]
        raise ModelInvocationError(f"{error} {detail}".strip()) from error


def _openai_compatible(
    profile: ModelProfile, query: str, category: str, timeout: int, max_tokens: int
) -> str:
    key = _key_for(profile)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    result = _post_json(
        f"{profile.api_endpoint.rstrip('/')}/chat/completions",
        {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": f"The request was routed as {category}. Answer it directly and helpfully."},
                {"role": "user", "content": query},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        },
        headers,
        timeout,
    )
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelInvocationError(f"Unexpected OpenAI-compatible response: {result}") from error
    return content if isinstance(content, str) else json.dumps(content)


def _anthropic(
    profile: ModelProfile, query: str, category: str, timeout: int, max_tokens: int
) -> str:
    key = _key_for(profile)
    if not key:
        raise ModelInvocationError("ANTHROPIC_API_KEY is required to invoke an Anthropic model.")
    result = _post_json(
        f"{profile.api_endpoint.rstrip('/')}/messages",
        {
            "model": profile.model,
            "system": f"The request was routed as {category}. Answer it directly and helpfully.",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_tokens,
        },
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
        timeout,
    )
    try:
        return "".join(part["text"] for part in result["content"] if part.get("type") == "text")
    except (KeyError, TypeError) as error:
        raise ModelInvocationError(f"Unexpected Anthropic response: {result}") from error


def _google(profile: ModelProfile, query: str, category: str, timeout: int, max_tokens: int) -> str:
    key = _key_for(profile)
    if not key:
        raise ModelInvocationError("GOOGLE_API_KEY is required to invoke a Google model.")
    url = f"{profile.api_endpoint.rstrip('/')}/models/{profile.model}:generateContent?key={urllib.parse.quote(key)}"
    result = _post_json(
        url,
        {
            "contents": [{"parts": [{"text": f"Category: {category}\n\n{query}"}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        },
        {},
        timeout,
    )
    try:
        return "".join(part["text"] for part in result["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError, TypeError) as error:
        raise ModelInvocationError(f"Unexpected Google response: {result}") from error


def invoke_selected_model(
    profile: ModelProfile,
    query: str,
    category: str,
    timeout: int = 90,
    max_tokens: int = 1024,
) -> str:
    """Invoke an adapter appropriate to the selected catalog profile."""
    if profile.service == "Anthropic":
        return _anthropic(profile, query, category, timeout, max_tokens)
    if profile.service == "Google":
        return _google(profile, query, category, timeout, max_tokens)
    return _openai_compatible(profile, query, category, timeout, max_tokens)
