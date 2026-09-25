"""Pre-run validation and smoke testing."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from .loader import get_unique_prompts, load_mbpp_data
from .runner import BenchmarkRunner, BenchmarkConfig


def run_data_validation(
    rows: list[dict[str, Any]], catalog_keys: set[str]
) -> dict[str, Any]:
    """Run all dataset validation checks from spec §33.

    Returns a report dict and prints results.
    """
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    n_rows = len(rows)
    unique_sids = set(r["sample_id"] for r in rows)
    unique_models = set(r["model_name"] for r in rows)
    perf_values = [r["performance"] for r in rows]
    cost_values = [r["cost"] for r in rows]

    perf_counter = Counter(perf_values)
    missing_perf = sum(1 for v in perf_values if math.isnan(v))
    missing_cost = sum(1 for v in cost_values if math.isnan(v))

    # Check duplicates
    seen_pairs: set[tuple[str, str]] = set()
    dup_count = 0
    for r in rows:
        pair = (r["sample_id"], r["model_name"])
        if pair in seen_pairs:
            dup_count += 1
        seen_pairs.add(pair)

    print(f"  1. Dataset shape: {n_rows} rows × 8 columns")
    print(f"  2. Unique MBPP sample_ids: {len(unique_sids)}")
    print(f"  3. Unique model_name values: {len(unique_models)}")
    print(f"  4. Exact model names:")
    for m in sorted(unique_models):
        print(f"       {m}")
    print(f"  5. Performance value distribution:")
    for val in sorted(perf_counter.keys()):
        print(f"       {val}: {perf_counter[val]}")
    print(f"     min={min(perf_values)}, max={max(perf_values)}")
    if set(perf_values) == {0.0, 1.0}:
        print("     → Performance is BINARY (0/1)")
    else:
        print("     → Performance is NOT binary")
    print(f"  6. Missing performance count: {missing_perf}")
    print(f"  7. Missing cost count: {missing_cost}")
    print(f"  8. Duplicate (sample_id, model_name) count: {dup_count}")
    print(f"  9. Router catalog model names ({len(catalog_keys)}):")
    for k in sorted(catalog_keys):
        print(f"       {k}")
    print(f" 10. Catalog-vs-dataset model-name match:")
    if catalog_keys == unique_models:
        print("       ✓ EXACT MATCH")
    else:
        only_cat = catalog_keys - unique_models
        only_rb = unique_models - catalog_keys
        if only_cat:
            print(f"       ✗ In catalog only: {sorted(only_cat)}")
        if only_rb:
            print(f"       ✗ In RouterBench only: {sorted(only_rb)}")
    print("=" * 60)

    report = {
        "n_rows": n_rows,
        "n_unique_samples": len(unique_sids),
        "n_unique_models": len(unique_models),
        "models": sorted(unique_models),
        "performance_binary": set(perf_values) == {0.0, 1.0},
        "performance_distribution": {str(k): v for k, v in sorted(perf_counter.items())},
        "missing_performance": missing_perf,
        "missing_cost": missing_cost,
        "duplicate_pairs": dup_count,
        "catalog_match": catalog_keys == unique_models,
    }
    return report


def run_smoke_test(
    runner: BenchmarkRunner,
    rows: list[dict[str, Any]],
    n: int = 5,
) -> bool:
    """Route a small number of prompts and verify the pipeline end-to-end."""
    from .oracle import enrich_results_with_oracle

    print()
    print("=" * 60)
    print(f"SMOKE TEST ({n} prompts)")
    print("=" * 60)

    unique = get_unique_prompts(rows)[:n]
    config = BenchmarkConfig(mode="mixed", alpha=0.65, beta=0.35)

    results = runner.run_configuration(unique, config, progress=True)

    # Enrich with oracle
    enrich_results_with_oracle(results, rows, config.alpha, config.beta)

    ok = True
    for r in results:
        issues = []
        if r.error:
            issues.append(f"router error: {r.error}")
        if not r.selected_model:
            issues.append("no selected_model")
        if not r.benchmark_lookup_success:
            issues.append(f"lookup failed: {r.benchmark_lookup_error}")
        if math.isnan(r.oracle_utility):
            issues.append("oracle_utility is NaN")
        if math.isnan(r.router_actual_utility):
            issues.append("router_actual_utility is NaN")
        if math.isnan(r.regret):
            issues.append("regret is NaN")
        if not math.isfinite(r.oracle_utility) and not math.isnan(r.oracle_utility):
            issues.append("oracle_utility is infinite")
        if not math.isfinite(r.regret) and not math.isnan(r.regret):
            issues.append("regret is infinite")

        status = "✓" if not issues else "✗"
        print(f"  {status} {r.sample_id}: selected={r.selected_model}, "
              f"perf={r.selected_benchmark_performance}, "
              f"regret={r.regret:.4f}" if not math.isnan(r.regret) else
              f"  {status} {r.sample_id}: {', '.join(issues)}")
        if issues:
            ok = False

    status_msg = "PASSED" if ok else "FAILED"
    print(f"\n  Smoke test: {status_msg}")
    print("=" * 60)
    return ok
