"""Configuration management for coding-router.

Handles loading the JSON model catalog and manages
config resolution with the following priority:
  1. Explicit path passed by caller
  2. LLMROUTER_CONFIG env var
  3. ./router/coding_llm.json  (project-local)
  4. Error telling user to run `coding-router init`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Where the package's bundled data lives.
_PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"

# Service name → candidate env var names (first match wins at runtime).
# API keys are NEVER stored in the JSON catalog.
SERVICE_ENV_MAP: dict[str, list[str]] = {
    "OpenAI": ["OPENAI_API_KEY"],
    "Anthropic": ["ANTHROPIC_API_KEY"],
    "Google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "xAI": ["XAI_API_KEY"],
    "DeepSeek": ["DEEPSEEK_API_KEY"],
    "Alibaba Cloud (Qwen)": ["DASHSCOPE_API_KEY"],
    "Mistral AI": ["MISTRAL_API_KEY"],
    "Z.ai (Zhipu)": ["ZAI_API_KEY"],
    "Moonshot AI": ["MOONSHOT_API_KEY"],
    "MiniMax": ["MINIMAX_API_KEY"],
    "Self-hosted (Ollama / vLLM)": [],  # no key needed
}


def _is_local_service(service: str) -> bool:
    """Return True if the service name implies a local/self-hosted model."""
    lower = service.lower()
    return any(kw in lower for kw in ("self-hosted", "ollama", "vllm", "local"))


def resolve_catalog_path(explicit: Path | None = None) -> Path:
    """Resolve the model catalog path using the priority chain.

    1. Explicit path passed by caller
    2. LLMROUTER_CONFIG env var
    3. ./router/coding_llm.json  (project-local)
    4. Raise with instructions to run ``coding-router init``
    """
    if explicit and explicit.exists():
        return explicit

    env_path = os.environ.get("LLMROUTER_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    local_path = Path("router") / "coding_llm.json"
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        "No model catalog found.\n"
        "Run 'coding-router init' to create one in ./router/coding_llm.json"
    )


def default_catalog_path() -> Path:
    """Return the project-local catalog path (./router/coding_llm.json).

    Does NOT create it — use ``coding-router init`` for that.
    """
    return Path("router") / "coding_llm.json"


def default_user_models_path() -> Path:
    """Return the path to the user's custom models file."""
    path = Path("router") / "user_models.json"
    if not path.exists():
        Path("router").mkdir(parents=True, exist_ok=True)
        path.write_text('{\n  "user_models": {}\n}\n', encoding="utf-8")
    return path


def default_index_path() -> Path:
    """Return the path to the cached embedding index."""
    return Path("router") / "model_embeddings.npz"


def load_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def bundled_catalog_path() -> Path:
    """Return the path to the bundled template catalog."""
    return _PACKAGE_DATA_DIR / "coding_llm_models.json"


def env_vars_for_service(service: str) -> list[str]:
    """Return the candidate env var names for a service."""
    return SERVICE_ENV_MAP.get(service, [])


def find_api_key(service: str) -> str | None:
    """Look up the API key for a service from the environment."""
    for var in env_vars_for_service(service):
        val = os.environ.get(var)
        if val:
            return val
    return None
