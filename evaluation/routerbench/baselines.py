"""Baseline strategies for comparison against the full router."""

from __future__ import annotations

import random
from typing import Any

from .runner import PromptResult


def cheapest_baseline(
    all_rows: list[dict[str, Any]],
    unique_prompts: list[dict[str, str]],
) -> list[PromptResult]:
    """Select the cheapest RouterBench model for each prompt.

    Uses the actual RouterBench cost for the prompt to determine the cheapest.
    """
    sample_index: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        sample_index.setdefault(row["sample_id"], []).append(row)

    results: list[PromptResult] = []
    for prompt_row in unique_prompts:
        sid = prompt_row["sample_id"]
        rows = sample_index.get(sid, [])
        if not rows:
            continue
        cheapest = min(rows, key=lambda r: r["cost"])
        result = PromptResult(
            sample_id=sid,
            prompt=prompt_row["prompt"],
            eval_name=prompt_row.get("eval_name", "mbpp"),
            mode="cheapest_baseline",
            alpha=0.0,
            beta=0.0,
            selected_model=cheapest["model_name"],
            selected_benchmark_performance=cheapest["performance"],
            selected_benchmark_cost=cheapest["cost"],
            benchmark_lookup_success=True,
        )
        results.append(result)
    return results


def random_baseline(
    all_rows: list[dict[str, Any]],
    unique_prompts: list[dict[str, str]],
    seed: int = 42,
) -> list[PromptResult]:
    """Randomly select one RouterBench model per prompt with fixed seed."""
    rng = random.Random(seed)

    sample_index: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        sample_index.setdefault(row["sample_id"], []).append(row)

    results: list[PromptResult] = []
    for prompt_row in unique_prompts:
        sid = prompt_row["sample_id"]
        rows = sample_index.get(sid, [])
        if not rows:
            continue
        chosen = rng.choice(rows)
        result = PromptResult(
            sample_id=sid,
            prompt=prompt_row["prompt"],
            eval_name=prompt_row.get("eval_name", "mbpp"),
            mode="random_baseline",
            alpha=0.0,
            beta=0.0,
            selected_model=chosen["model_name"],
            selected_benchmark_performance=chosen["performance"],
            selected_benchmark_cost=chosen["cost"],
            benchmark_lookup_success=True,
        )
        results.append(result)
    return results


def semantic_only_baseline(
    runner,  # BenchmarkRunner
    unique_prompts: list[dict[str, str]],
) -> list[PromptResult]:
    """Select the model with the highest capability-fit score (no cost tradeoff).

    Runs the router in skill_based mode (alpha=1.0, beta=0.0) with no gating,
    which is equivalent to argmax(capability_normalized) across the full pool.
    """
    import time
    results: list[PromptResult] = []
    total = len(unique_prompts)
    for idx, prompt_row in enumerate(unique_prompts):
        print(
            f"\r  [{idx + 1}/{total}] semantic_only           "
            f"sample_id={prompt_row['sample_id']}",
            end="",
            flush=True,
        )
        sid = prompt_row["sample_id"]
        result = PromptResult(
            sample_id=sid,
            prompt=prompt_row["prompt"],
            eval_name=prompt_row.get("eval_name", "mbpp"),
            mode="semantic_only_baseline",
            alpha=1.0,
            beta=0.0,
        )
        try:
            start = time.perf_counter()
            state = runner.router.route(
                prompt_row["prompt"],
                route_only=True,
                routing_mode="skill_based",
                alpha=1.0,
                beta=0.0,
                force_no_gating=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # From ranked_models, pick argmax(perf_normalized) = capability fit
            ranked = state.get("ranked_models", [])
            if ranked:
                best = max(ranked, key=lambda m: m["perf_normalized"])
                result.selected_model = best["catalog_key"]
                result.router_similarity = best.get("similarity", 0.0)
                result.router_capability_normalized = best.get("perf_normalized", 0.0)
            result.router_latency_ms = elapsed_ms
        except Exception as exc:
            result.error = str(exc)
        results.append(result)
    print()
    return results
