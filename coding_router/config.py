"""Configuration management for coding-router.

Handles loading the JSON model catalog and manages
the default catalog shipped with the package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Where the package's bundled data lives.
_PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"

# User-level configuration directory — the writable copy lives here.
_DEFAULT_CONFIG_DIR = Path(os.environ.get("CODING_ROUTER_CONFIG_DIR", ""))
if not _DEFAULT_CONFIG_DIR.name:
    _DEFAULT_CONFIG_DIR = Path.home() / ".config" / "coding-router"


def config_dir() -> Path:
    """Return (and create if needed) the user-level config directory."""
    _DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_CONFIG_DIR


def default_catalog_path() -> Path:
    """Return the path to the user's editable model catalog.

    On first call, if the file doesn't exist yet, the bundled template is
    copied into ``~/.config/coding-router/coding_llm_models.json``.
    """
    user_catalog = config_dir() / "coding_llm_models.json"
    if not user_catalog.exists():
        bundled = _PACKAGE_DATA_DIR / "coding_llm_models.json"
        if bundled.exists():
            user_catalog.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
    return user_catalog


def default_user_models_path() -> Path:
    """Return the path to the user's custom models file."""
    path = config_dir() / "user_models.json"
    if not path.exists():
        path.write_text('{\n  "user_models": {}\n}\n', encoding="utf-8")
    return path


def default_index_path() -> Path:
    """Return the path to the cached embedding index."""
    return config_dir() / "model_embeddings.npz"


def load_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
