"""Benchmark catalog loading and verification."""

from __future__ import annotations

from pathlib import Path

from model_router.catalog import ModelProfile, load_catalog


def load_benchmark_catalog(path: Path) -> list[ModelProfile]:
    """Load the benchmark-only model catalog.

    This is intentionally separate from the production catalog.
    """
    return load_catalog(path)


def verify_model_match(
    catalog_keys: set[str], routerbench_models: set[str]
) -> None:
    """Verify exact set equality between catalog keys and RouterBench model names.

    Raises ``ValueError`` with details on mismatch.
    """
    if catalog_keys == routerbench_models:
        return
    only_catalog = catalog_keys - routerbench_models
    only_routerbench = routerbench_models - catalog_keys
    parts: list[str] = ["Model-name mismatch between catalog and RouterBench:"]
    if only_catalog:
        parts.append(f"  In catalog but NOT in RouterBench: {sorted(only_catalog)}")
    if only_routerbench:
        parts.append(f"  In RouterBench but NOT in catalog: {sorted(only_routerbench)}")
    raise ValueError("\n".join(parts))
