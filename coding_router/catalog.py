"""Load the LLM catalog into a uniform routing representation.

Supports both standard JSON and JSONC (JSON with ``//`` comments) files.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_json

# These are deliberately explicit: a model only enters the expensive group when
# it has been assessed as a strong fit for long-horizon or difficult work.
ADVANCED_MODEL_KEYS = frozenset(
    {
        "gpt-5.6-sol",
        "gpt-5.1-codex",
        "claude-opus-5",
        "gemini-3-pro",
        "qwen3-coder-plus",
        "kimi-k2",
        "minimax-m2",
        "deepseek-r1-distill-qwen-32b",
        "llama-3.3-70b",
    }
)


@dataclass(frozen=True)
class ModelProfile:
    """A provider-neutral record used for search, selection, and invocation."""

    catalog_key: str
    catalog_group: str
    model: str
    service: str
    api_endpoint: str
    feature: str
    size: str
    input_price: float
    output_price: float
    used_in: tuple[str, ...]
    tier: str = "standard"
    api_key_env: str | None = None
    user_defined: bool = False

    @property
    def advanced(self) -> bool:
        return self.tier == "advanced"

    def document(self) -> str:
        """The semantic-search text embedded and persisted in the local index."""
        tier = "advanced reasoning and long-horizon work" if self.advanced else "standard coding work"
        return "\n".join(
            (
                f"Model: {self.catalog_key} ({self.model})",
                f"Catalog group: {self.catalog_group}",
                f"Service: {self.service}",
                f"Work tier: {tier}",
                f"Size: {self.size}",
                f"Price: ${self.input_price:.2f} per million input tokens; ${self.output_price:.2f} per million output tokens",
                f"Capabilities: {self.feature}",
                f"Available in: {', '.join(self.used_in)}",
            )
        )

    def public_dict(self) -> dict[str, Any]:
        """Selection metadata safe to return from the command-line router."""
        return {
            "catalog_key": self.catalog_key,
            "model": self.model,
            "service": self.service,
            "api_endpoint": self.api_endpoint,
            "advanced": self.advanced,
            "input_price_per_million": self.input_price,
            "output_price_per_million": self.output_price,
            "tier": self.tier,
            "user_defined": self.user_defined,
        }


def load_catalog(
    path: Path, *, user_defined: bool = False, allow_empty: bool = False
) -> list[ModelProfile]:
    """Read every catalog group without assuming a particular number of models."""
    from .config import load_json
    raw = load_json(path)

    profiles: list[ModelProfile] = []
    for group_name, records in raw.items():
        if not isinstance(records, dict):
            continue
        for catalog_key, record in records.items():
            if not isinstance(record, dict):
                continue
            
            # Skip disabled models
            if not record.get("enabled", True):
                continue

            profiles.append(
                ModelProfile(
                    catalog_key=catalog_key,
                    catalog_group=group_name,
                    model=record["model"],
                    service=record["service"],
                    api_endpoint=record["api_endpoint"],
                    feature=record["feature"],
                    size=record["size"],
                    input_price=float(record["input_price"]),
                    output_price=float(record["output_price"]),
                    used_in=tuple(record.get("used_in", [])),
                    tier=record.get(
                        "tier",
                        "advanced" if catalog_key in ADVANCED_MODEL_KEYS else "standard",
                    ),
                    api_key_env=record.get("api_key_env"),
                    user_defined=user_defined,
                )
            )
    if not profiles and not allow_empty:
        raise ValueError(f"No enabled model profiles found in {path}")
    return profiles


def load_catalogs(
    catalog_path: Path,
    user_catalog_path: Path | None = None,
    *,
    candidate_pool: str = "all",
) -> list[ModelProfile]:
    """Combine curated and user-managed models, optionally using only the latter."""
    curated = load_catalog(catalog_path)
    user_models = (
        load_catalog(user_catalog_path, user_defined=True, allow_empty=True)
        if user_catalog_path and user_catalog_path.exists()
        else []
    )
    profiles = user_models if candidate_pool == "user" else [*curated, *user_models]
    keys = [profile.catalog_key for profile in profiles]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate catalog keys: {', '.join(duplicates)}")
    if not profiles:
        raise ValueError("The selected candidate pool contains no models.")
    return profiles


def add_user_model(
    path: Path,
    *,
    catalog_key: str,
    model: str,
    service: str,
    api_endpoint: str,
    feature: str,
    size: str,
    input_price: float,
    output_price: float,
    tier: str = "standard",
    api_key_env: str | None = None,
) -> None:
    """Add one local or cloud candidate to the user-managed candidate pool."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", catalog_key):
        raise ValueError("Model key must use lowercase letters, digits, and hyphens.")
    if tier not in {"standard", "advanced"}:
        raise ValueError("Tier must be either 'standard' or 'advanced'.")
    if input_price < 0 or output_price < 0:
        raise ValueError("Token prices cannot be negative.")

    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"user_models": {}}
    records = data.setdefault("user_models", {})
    if catalog_key in records:
        raise ValueError(f"A user model named '{catalog_key}' already exists.")
    records[catalog_key] = {
        "size": size,
        "feature": feature,
        "input_price": input_price,
        "output_price": output_price,
        "model": model,
        "service": service,
        "api_endpoint": api_endpoint.rstrip("/"),
        "used_in": ["User-managed candidate pool"],
        "tier": tier,
    }
    if api_key_env:
        records[catalog_key]["api_key_env"] = api_key_env
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def catalog_fingerprint(paths: Path | Iterable[Path], embedding_model: str) -> str:
    """Make an index stale when either its catalog or embedding model changes."""
    digest = hashlib.sha256()
    catalog_paths = [paths] if isinstance(paths, Path) else list(paths)
    for path in catalog_paths:
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
        digest.update(b"\0")
    digest.update(b"coding-router-index-v1\0")
    digest.update(embedding_model.encode("utf-8"))
    return digest.hexdigest()
