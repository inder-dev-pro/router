"""Load and validate the RouterBench MBPP dataset."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any


def load_mbpp_data(path: Path) -> list[dict[str, Any]]:
    """Load the RouterBench MBPP CSV into a list of row dicts.

    Parses ``performance`` and ``cost`` as floats immediately so
    downstream code never has to worry about string conversion.
    """
    raw = path.read_bytes().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    rows: list[dict[str, Any]] = []
    for row in reader:
        row["performance"] = float(row["performance"])
        row["cost"] = float(row["cost"])
        rows.append(row)
    return rows


def get_unique_prompts(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Deduplicate by sample_id, returning one row per unique prompt.

    The first occurrence of each sample_id is kept.  The returned list
    preserves the original dataset ordering for deterministic runs.
    """
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        sid = row["sample_id"]
        if sid not in seen:
            seen.add(sid)
            unique.append({"sample_id": sid, "prompt": row["prompt"], "eval_name": row["eval_name"]})
    return unique


def build_ground_truth_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Map (sample_id, model_name) → {performance, cost}.

    Raises on duplicate (sample_id, model_name) pairs.
    """
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["sample_id"], row["model_name"])
        if key in lookup:
            raise ValueError(f"Duplicate ground-truth row: {key}")
        lookup[key] = {"performance": row["performance"], "cost": row["cost"]}
    return lookup


def get_models_for_sample(
    rows: list[dict[str, Any]], sample_id: str
) -> list[dict[str, Any]]:
    """Return all RouterBench rows for a given sample_id."""
    return [r for r in rows if r["sample_id"] == sample_id]
