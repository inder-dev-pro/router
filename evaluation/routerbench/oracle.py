"""Post-hoc oracle computation using RouterBench ground truth.

The oracle is computed AFTER the router has made its selection.
It uses the same quality/cost objective as the router but with
actual RouterBench performance and cost values.
"""

from __future__ import annotations

import math
from typing import Any

from model_router.app import _normalize

from .runner import PromptResult


def _oracle_utility(
    performance: float,
    cost: float,
    perf_normalized: float,
    cost_normalized: float,
    alpha: float,
    beta: float,
) -> float:
    """Compute oracle utility for a single candidate.

    oracle_utility = alpha * perf_normalized + beta * (1 - cost_normalized)
    """
    return alpha * perf_normalized + beta * (1.0 - cost_normalized)


def compute_oracle_for_sample(
    sample_rows: list[dict[str, Any]],
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    """Find the oracle model for a single sample_id.

    Returns a dict with the oracle model name, utility, and full
    per-model rankings for auditability.
    """
    if not sample_rows:
        return {
            "oracle_model": "",
            "oracle_utility": float("nan"),
            "rankings": [],
        }

    performances = [r["performance"] for r in sample_rows]
    costs = [r["cost"] for r in sample_rows]

    # Normalise using the same _normalize() as the router.
    perf_norm = _normalize(performances, flat_value=1.0)
    cost_norm = _normalize(costs, flat_value=0.0)

    rankings: list[dict[str, Any]] = []
    for row, pn, cn in zip(sample_rows, perf_norm, cost_norm):
        util = _oracle_utility(row["performance"], row["cost"], pn, cn, alpha, beta)
        rankings.append(
            {
                "model_name": row["model_name"],
                "performance": row["performance"],
                "cost": row["cost"],
                "performance_normalized": round(pn, 8),
                "cost_normalized": round(cn, 8),
                "oracle_utility": round(util, 8),
            }
        )

    # Sort by utility descending, then performance descending, then cost ascending.
    rankings.sort(key=lambda r: (r["oracle_utility"], r["performance"], -r["cost"]), reverse=True)

    # Assign ranks
    for i, entry in enumerate(rankings):
        entry["oracle_rank"] = i + 1

    best = rankings[0]
    return {
        "oracle_model": best["model_name"],
        "oracle_utility": best["oracle_utility"],
        "rankings": rankings,
    }


def compute_router_actual_utility(
    selected_model: str,
    sample_rows: list[dict[str, Any]],
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """Compute the actual utility of the model the router selected.

    Uses the same normalisation pool as the oracle so the comparison
    is apples-to-apples.
    """
    performances = [r["performance"] for r in sample_rows]
    costs = [r["cost"] for r in sample_rows]
    perf_norm = _normalize(performances, flat_value=1.0)
    cost_norm = _normalize(costs, flat_value=0.0)

    for row, pn, cn in zip(sample_rows, perf_norm, cost_norm):
        if row["model_name"] == selected_model:
            util = _oracle_utility(row["performance"], row["cost"], pn, cn, alpha, beta)
            return {
                "router_actual_utility": round(util, 8),
                "selected_benchmark_performance": row["performance"],
                "selected_benchmark_cost": row["cost"],
            }

    # Lookup miss
    return {
        "router_actual_utility": float("nan"),
        "selected_benchmark_performance": float("nan"),
        "selected_benchmark_cost": float("nan"),
    }


def enrich_results_with_oracle(
    results: list[PromptResult],
    all_rows: list[dict[str, Any]],
    alpha: float,
    beta: float,
) -> list[dict[str, Any]]:
    """Attach oracle + actual-utility + regret to every PromptResult.

    Also returns the full oracle rankings for the detailed export.
    """
    # Build per-sample row index
    sample_index: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        sample_index.setdefault(row["sample_id"], []).append(row)

    all_oracle_rankings: list[dict[str, Any]] = []

    for result in results:
        sid = result.sample_id
        sample_rows = sample_index.get(sid, [])

        if not sample_rows:
            result.benchmark_lookup_error = f"No RouterBench rows for sample_id={sid}"
            continue

        # Oracle computation
        oracle = compute_oracle_for_sample(sample_rows, alpha, beta)
        result.oracle_model = oracle["oracle_model"]
        result.oracle_utility = oracle["oracle_utility"]

        # Add sample_id to rankings for export
        for ranking_row in oracle["rankings"]:
            ranking_row["sample_id"] = sid
            ranking_row["alpha"] = alpha
            ranking_row["beta"] = beta
        all_oracle_rankings.extend(oracle["rankings"])

        # Router actual utility
        if result.selected_model and not result.error:
            actual = compute_router_actual_utility(
                result.selected_model, sample_rows, alpha, beta
            )
            result.router_actual_utility = actual["router_actual_utility"]
            result.selected_benchmark_performance = actual["selected_benchmark_performance"]
            result.selected_benchmark_cost = actual["selected_benchmark_cost"]

            if not math.isnan(result.router_actual_utility):
                result.benchmark_lookup_success = True
                result.regret = round(
                    result.oracle_utility - result.router_actual_utility, 8
                )
                result.oracle_agreement = result.selected_model == result.oracle_model
            else:
                result.benchmark_lookup_error = (
                    f"Model '{result.selected_model}' not found in "
                    f"RouterBench rows for sample_id={sid}"
                )
        else:
            result.benchmark_lookup_error = result.error or "No selected model"

        # Task solvability
        max_perf = max(r["performance"] for r in sample_rows)
        result.max_candidate_performance = max_perf
        result.task_solvable = max_perf > 0
        result.task_unsolvable = max_perf == 0

        # Classify outcome
        if not math.isnan(result.selected_benchmark_performance):
            if result.selected_benchmark_performance == 1.0:
                result.routing_success = True
            elif result.task_solvable:
                result.routing_failure = True

    return all_oracle_rankings
