"""Aggregate and per-prompt metrics for the RouterBench evaluation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .runner import PromptResult


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _safe_median(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.median(finite)) if finite else float("nan")


def _percentile(values: list[float], pct: float) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.percentile(finite, pct)) if finite else float("nan")


def compute_summary_metrics(results: list[PromptResult]) -> dict[str, Any]:
    """Compute all primary and additional metrics from a set of prompt results."""
    n = len(results)
    if n == 0:
        return {"num_prompts": 0, "error": "No results"}

    # Extract vectors
    perfs = [r.selected_benchmark_performance for r in results]
    costs = [r.selected_benchmark_cost for r in results]
    oracle_utils = [r.oracle_utility for r in results]
    router_utils = [r.router_actual_utility for r in results]
    regrets = [r.regret for r in results]
    latencies = [r.router_latency_ms for r in results]

    # Task classification counts
    solvable = [r for r in results if r.task_solvable]
    unsolvable = [r for r in results if r.task_unsolvable]
    successes = [r for r in results if r.routing_success]
    failures = [r for r in results if r.routing_failure]
    errors = [r for r in results if r.error]
    lookup_errors = [r for r in results if r.benchmark_lookup_error]
    classifier_errors = [r for r in results if r.classifier_error]

    # Successful routing among solvable tasks
    solvable_successes = [r for r in solvable if r.routing_success]

    # Oracle agreement
    agreements = [r for r in results if r.oracle_agreement]

    # Zero regret
    zero_regret = [r for r in results if not math.isnan(r.regret) and abs(r.regret) < 1e-9]

    # Model selection frequency
    model_counter = Counter(r.selected_model for r in results if r.selected_model)

    # Selection group frequency
    group_counter = Counter(r.selection_group for r in results)

    return {
        "num_prompts": n,
        "mode": results[0].mode if results else "",
        "alpha": results[0].alpha if results else 0.0,
        "beta": results[0].beta if results else 0.0,
        "disable_gating": results[0].disable_gating if results else False,

        # §19 Primary metrics
        "routed_task_success_rate": len(successes) / n if n else 0.0,
        "solvable_task_success_rate": (
            len(solvable_successes) / len(solvable) if solvable else float("nan")
        ),
        "routing_failure_rate": len(failures) / n if n else 0.0,
        "unsolvable_rate": len(unsolvable) / n if n else 0.0,
        "avg_actual_cost": _safe_mean(costs),
        "median_actual_cost": _safe_median(costs),
        "p95_actual_cost": _percentile(costs, 95),
        "avg_oracle_utility": _safe_mean(oracle_utils),
        "avg_router_utility": _safe_mean(router_utils),
        "avg_regret": _safe_mean(regrets),
        "median_regret": _safe_median(regrets),
        "oracle_agreement_rate": len(agreements) / n if n else 0.0,

        # Zero-regret percentage
        "zero_regret_pct": len(zero_regret) / n if n else 0.0,

        # §20 Additional metrics
        "avg_router_latency_ms": _safe_mean(latencies),
        "p50_router_latency_ms": _safe_median(latencies),
        "p95_router_latency_ms": _percentile(latencies, 95),
        "classifier_error_rate": len(classifier_errors) / n if n else 0.0,
        "num_router_errors": len(errors),
        "num_benchmark_lookup_errors": len(lookup_errors),

        # Model selection distribution
        "model_selection_distribution": dict(model_counter.most_common()),
        "selection_group_distribution": dict(group_counter),

        # Counts
        "num_solvable": len(solvable),
        "num_unsolvable": len(unsolvable),
        "num_successes": len(successes),
        "num_failures": len(failures),
        "num_oracle_agreements": len(agreements),
        "num_zero_regret": len(zero_regret),
    }


def compute_model_level_stats(
    results: list[PromptResult],
) -> list[dict[str, Any]]:
    """Per-model statistics: selection count, success rate when selected."""
    model_results: dict[str, list[PromptResult]] = {}
    for r in results:
        if r.selected_model:
            model_results.setdefault(r.selected_model, []).append(r)

    stats = []
    total = len(results)
    for model, model_rs in sorted(model_results.items()):
        successes = [r for r in model_rs if r.routing_success]
        stats.append(
            {
                "model_name": model,
                "selections": len(model_rs),
                "selection_percentage": round(len(model_rs) / total * 100, 2) if total else 0.0,
                "success_when_selected": (
                    len(successes) / len(model_rs) if model_rs else 0.0
                ),
                "avg_cost_when_selected": _safe_mean(
                    [r.selected_benchmark_cost for r in model_rs]
                ),
            }
        )
    return stats
