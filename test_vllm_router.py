#!/usr/bin/env python3
"""Test an adapted-arch-router-style model served by vLLM's OpenAI-compatible API.

Examples:
  python test_vllm_router.py
  python test_vllm_router.py --message "Write a Python CSV parser"
  python test_vllm_router.py --model katanemo/adapted-arch-router-1.5B --verbose

Start vLLM first, for example:
  vllm serve katanemo/adapted-arch-router-1.5B --trust-remote-code --port 8000
"""

import argparse
import ast
import json
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_ROUTES = [
    {
        "name": "code_generation",
        "description": "Generating new code snippets, functions, or boilerplate based on user prompts or requirements",
    },
    {
        "name": "bug_fixing",
        "description": "Identifying and fixing errors or bugs in the provided code across different programming languages",
    },
    {
        "name": "performance_optimization",
        "description": "Suggesting improvements to make code more efficient, readable, or scalable",
    },
    {
        "name": "api_help",
        "description": "Assisting with understanding or integrating external APIs and libraries",
    },
    {
        "name": "programming",
        "description": "Answering general programming questions, theory, or best practices",
    },
]

TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags. Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent.
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"}
"""


def make_prompt(routes: list[dict[str, str]], conversation: list[dict[str, str]]) -> str:
    """Reproduce the prompt structure supplied with the original notebook."""
    return TASK_INSTRUCTION.format(
        routes=json.dumps(routes), conversation=json.dumps(conversation)
    ) + FORMAT_PROMPT


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def discover_model(base_url: str, timeout: int) -> str:
    models = request_json(f"{base_url}/models", timeout=timeout)
    data = models.get("data", [])
    if not data or "id" not in data[0]:
        raise RuntimeError("The server returned no models from /v1/models.")
    return data[0]["id"]


def parse_route(content: str) -> str | None:
    """Extract the route from JSON or the Python-style dict some models emit."""
    candidates = [content]
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        candidates.append(content[start : end + 1])
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                if isinstance(parsed, dict) and isinstance(parsed.get("route"), str):
                    return parsed["route"]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM OpenAI API base URL")
    parser.add_argument("--model", help="served model ID; defaults to the first model at /models")
    parser.add_argument(
        "--message",
        default="fix this module 'torch.utils._pytree' has no attribute 'register_pytree_node'. did you mean: '_register_pytree_node'?",
        help="user message to route",
    )
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=30, help="request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="print prompt and full API response")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    try:
        model = args.model or discover_model(base_url, args.timeout)
        prompt = make_prompt(DEFAULT_ROUTES, [{"role": "user", "content": args.message}])
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        result = request_json(f"{base_url}/chat/completions", payload, args.timeout)
        content = result["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, RuntimeError) as error:
        print(f"vLLM test failed: {error}", file=sys.stderr)
        print(f"Check that vLLM is serving at {base_url}.", file=sys.stderr)
        return 1

    route = parse_route(content)
    print(f"model: {model}")
    print(f"raw response: {content}")
    print(f"selected route: {route if route else 'could not parse route JSON'}")
    if args.verbose:
        print("\nPrompt:\n" + prompt)
        print("\nFull API response:\n" + json.dumps(result, indent=2))
    return 0 if route else 2


if __name__ == "__main__":
    raise SystemExit(main())
