"""JSONC parser and configuration management for coding-router.

Handles loading JSON files that include // line comments, and manages
the default model catalog shipped with the package.
"""

from __future__ import annotations

import json
import os
import re
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
    copied into ``~/.config/coding-router/coding_llm_models.jsonc``.
    """
    user_catalog = config_dir() / "coding_llm_models.jsonc"
    if not user_catalog.exists():
        bundled = _PACKAGE_DATA_DIR / "coding_llm_models.jsonc"
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


# ---------------------------------------------------------------------------
# JSONC parser
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(
    r"""
    (?P<string>"(?:[^"\\]|\\.)*")   # skip double-quoted strings
    | (?P<comment>//[^\n]*)          # match // line comments
    """,
    re.VERBOSE,
)


def strip_jsonc_comments(text: str) -> str:
    """Remove ``//`` line comments from JSONC text while preserving strings."""

    def _replace(match: re.Match[str]) -> str:
        if match.group("string"):
            return match.group("string")
        return ""  # drop the comment

    return _COMMENT_RE.sub(_replace, text)


def load_jsonc(path: Path) -> Any:
    """Read a ``.jsonc`` or ``.json`` file, stripping ``//`` comments first."""
    raw = path.read_text(encoding="utf-8")
    clean = strip_jsonc_comments(raw)
    return json.loads(clean)
