"""RouterBench MBPP offline benchmark — main entry point.

Usage:
    python -m evaluation.run_mbpp_benchmark
    python -m evaluation.run_mbpp_benchmark --limit 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.routerbench.loader import (
    build_ground_truth_lookup,
    get_unique_prompts,
    load_mbpp_data,
)
from evaluation.routerbench.catalog import load_benchmark_catalog, verify_model_match
from evaluation.routerbench.runner import BenchmarkConfig, BenchmarkRunner
from evaluation.routerbench.oracle import enrich_results_with_oracle
from evaluation.routerbench.metrics import compute_model_level_stats, compute_summary_metrics
from evaluation.routerbench.baselines import cheapest_baseline, random_baseline, semantic_only_baseline
from evaluation.routerbench.validation import run_data_validation, run_smoke_test
from evaluation.routerbench.report import (
    export_benchmark_config,
    export_combined_summary,
    export_errors,
    export_oracle_rankings,
    export_per_prompt_csv,
    export_summary_json,
    generate_report,
)
from evaluation.routerbench.plots import generate_all_plots


# ── Paths ──────────────────────────────────────────────────────────────────
DATA_FILE = PROJECT_ROOT / "data" / "routerbench_raw_mbpp.csv"
BENCHMARK_CATALOG = PROJECT_ROOT / "data" / "benchmark_models.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results" / "routerbench_mbpp"

# ── Benchmark configurations ──────────────────────────────────────────────
CONFIGS = [
    BenchmarkConfig(mode="cost_efficient", alpha=0.20, beta=0.80, label="cost_efficient"),
    BenchmarkConfig(mode="mixed",          alpha=0.65, beta=0.35, label="mixed"),
    BenchmarkConfig(mode="skill_based",    alpha=1.00, beta=0.00, label="skill_based"),
]

# Gating ablation: mixed mode without gating
ABLATION_CONFIG = BenchmarkConfig(
    mode="mixed", alpha=0.65, beta=0.35, disable_gating=True, label="mixed_no_gating"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RouterBench MBPP offline benchmark.")
    parser.add_argument("--input", type=Path, default=DATA_FILE)
    parser.add_argument("--catalog", type=Path, default=BENCHMARK_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of unique prompts")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke test")
    parser.add_argument("--classifier-url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    output_dir = args.output_dir
    benchmark_start = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # 1. LOAD DATA
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n📂 Loading data from {args.input}...")
    all_rows = load_mbpp_data(args.input)
    unique_prompts = get_unique_prompts(all_rows)
    if args.limit:
        unique_prompts = unique_prompts[: args.limit]
    print(f"   {len(all_rows)} total rows, {len(unique_prompts)} unique prompts")

    # ══════════════════════════════════════════════════════════════════════
    # 2. BUILD RUNNER + VERIFY CATALOG
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n🔧 Initialising benchmark runner (catalog: {args.catalog})...")
    runner = BenchmarkRunner(
        catalog_path=args.catalog,
        classifier_base_url=args.classifier_url,
    )

    routerbench_models = set(r["model_name"] for r in all_rows)
    try:
        verify_model_match(runner.catalog_keys, routerbench_models)
        print("   ✓ Catalog ↔ RouterBench model names: EXACT MATCH")
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nBenchmark STOPPED. Fix the mismatch before proceeding.")
        return 1

    # ══════════════════════════════════════════════════════════════════════
    # 3. DATA VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    validation_report = run_data_validation(all_rows, runner.catalog_keys)

    if validation_report["duplicate_pairs"] > 0:
        print("\n❌ Duplicate (sample_id, model_name) pairs found. Cannot proceed.")
        return 1

    # ══════════════════════════════════════════════════════════════════════
    # 4. SMOKE TEST
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_smoke:
        smoke_ok = run_smoke_test(runner, all_rows, n=5)
        if not smoke_ok:
            print("\n❌ Smoke test FAILED. Inspect errors above before running full benchmark.")
            return 1

    # ══════════════════════════════════════════════════════════════════════
    # 5. FULL BENCHMARK — ROUTER CONFIGURATIONS
    # ══════════════════════════════════════════════════════════════════════
    all_summaries: list[dict] = []
    all_model_stats: dict[str, list[dict]] = {}
    all_oracle_rankings: list[dict] = []
    all_results_flat: list = []  # for errors.csv

    configs_to_run = CONFIGS + [ABLATION_CONFIG]

    for config in configs_to_run:
        print(f"\n{'='*60}")
        print(f"RUNNING: {config.label} (α={config.alpha}, β={config.beta}, gating={'off' if config.disable_gating else 'on'})")
        print(f"{'='*60}")

        results = runner.run_configuration(unique_prompts, config, progress=True)

        # Enrich with oracle
        oracle_rankings = enrich_results_with_oracle(
            results, all_rows, config.alpha, config.beta
        )
        all_oracle_rankings.extend(oracle_rankings)
        all_results_flat.extend(results)

        # Compute metrics
        summary = compute_summary_metrics(results)
        summary["configuration"] = config.label
        all_summaries.append(summary)

        # Model-level stats
        model_stats = compute_model_level_stats(results)
        all_model_stats[config.label] = model_stats

        # Export per-config results
        config_dir = output_dir / ("ablations" if config.disable_gating else "") / config.label
        export_per_prompt_csv(results, config_dir / "per_prompt.csv")
        export_summary_json(summary, config_dir / "summary.json")

        # Print summary
        print(f"\n  Results for {config.label}:")
        print(f"    Success rate:     {summary['routed_task_success_rate']:.1%}")
        print(f"    Avg cost:         {summary['avg_actual_cost']:.6f}")
        print(f"    Avg utility:      {summary['avg_router_utility']:.4f}")
        print(f"    Avg oracle:       {summary['avg_oracle_utility']:.4f}")
        print(f"    Avg regret:       {summary['avg_regret']:.4f}")
        print(f"    Oracle agreement: {summary['oracle_agreement_rate']:.1%}")

    # ══════════════════════════════════════════════════════════════════════
    # 6. BASELINES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("RUNNING BASELINES")
    print(f"{'='*60}")

    # Use mixed mode alpha/beta for baseline oracle comparison
    baseline_alpha, baseline_beta = 0.65, 0.35

    # ── Cheapest ──
    print("\n  Cheapest baseline...")
    cheapest_results = cheapest_baseline(all_rows, unique_prompts)
    enrich_results_with_oracle(cheapest_results, all_rows, baseline_alpha, baseline_beta)
    cheapest_summary = compute_summary_metrics(cheapest_results)
    cheapest_summary["configuration"] = "baseline_cheapest"
    cheapest_summary["mode"] = "cheapest_baseline"
    all_summaries.append(cheapest_summary)
    all_model_stats["baseline_cheapest"] = compute_model_level_stats(cheapest_results)
    export_per_prompt_csv(cheapest_results, output_dir / "baselines" / "cheapest" / "per_prompt.csv")
    export_summary_json(cheapest_summary, output_dir / "baselines" / "cheapest" / "summary.json")
    all_results_flat.extend(cheapest_results)
    print(f"    Success: {cheapest_summary['routed_task_success_rate']:.1%}, "
          f"Cost: {cheapest_summary['avg_actual_cost']:.6f}, "
          f"Regret: {cheapest_summary['avg_regret']:.4f}")

    # ── Random ──
    print("\n  Random baseline (seed=42)...")
    random_results = random_baseline(all_rows, unique_prompts, seed=42)
    enrich_results_with_oracle(random_results, all_rows, baseline_alpha, baseline_beta)
    random_summary = compute_summary_metrics(random_results)
    random_summary["configuration"] = "baseline_random"
    random_summary["mode"] = "random_baseline"
    all_summaries.append(random_summary)
    all_model_stats["baseline_random"] = compute_model_level_stats(random_results)
    export_per_prompt_csv(random_results, output_dir / "baselines" / "random" / "per_prompt.csv")
    export_summary_json(random_summary, output_dir / "baselines" / "random" / "summary.json")
    all_results_flat.extend(random_results)
    print(f"    Success: {random_summary['routed_task_success_rate']:.1%}, "
          f"Cost: {random_summary['avg_actual_cost']:.6f}, "
          f"Regret: {random_summary['avg_regret']:.4f}")

    # ── Semantic-only ──
    print("\n  Semantic-only baseline...")
    semantic_results = semantic_only_baseline(runner, unique_prompts)
    enrich_results_with_oracle(semantic_results, all_rows, baseline_alpha, baseline_beta)
    semantic_summary = compute_summary_metrics(semantic_results)
    semantic_summary["configuration"] = "baseline_semantic_only"
    semantic_summary["mode"] = "semantic_only_baseline"
    all_summaries.append(semantic_summary)
    all_model_stats["baseline_semantic_only"] = compute_model_level_stats(semantic_results)
    export_per_prompt_csv(semantic_results, output_dir / "baselines" / "semantic_only" / "per_prompt.csv")
    export_summary_json(semantic_summary, output_dir / "baselines" / "semantic_only" / "summary.json")
    all_results_flat.extend(semantic_results)
    print(f"    Success: {semantic_summary['routed_task_success_rate']:.1%}, "
          f"Cost: {semantic_summary['avg_actual_cost']:.6f}, "
          f"Regret: {semantic_summary['avg_regret']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # 7. EXPORT GLOBAL RESULTS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("EXPORTING RESULTS")
    print(f"{'='*60}")

    export_combined_summary(all_summaries, output_dir / "combined_summary.csv")
    export_oracle_rankings(all_oracle_rankings, output_dir / "oracle_rankings.csv")
    export_errors(all_results_flat, output_dir / "errors.csv")

    benchmark_config = {
        "dataset_path": str(args.input),
        "catalog_path": str(args.catalog),
        "num_unique_prompts": len(unique_prompts),
        "num_models": len(runner.catalog_keys),
        "models": sorted(runner.catalog_keys),
        "configurations": [
            {"label": c.label, "mode": c.mode, "alpha": c.alpha, "beta": c.beta,
             "disable_gating": c.disable_gating}
            for c in configs_to_run
        ],
        "baselines": ["cheapest", "random", "semantic_only"],
        "baseline_oracle_alpha": baseline_alpha,
        "baseline_oracle_beta": baseline_beta,
        "random_seed": 42,
    }
    export_benchmark_config(benchmark_config, output_dir / "benchmark_config.json")

    generate_report(all_summaries, all_model_stats, validation_report, output_dir / "REPORT.md")

    # ══════════════════════════════════════════════════════════════════════
    # 8. PLOTS
    # ══════════════════════════════════════════════════════════════════════
    generate_all_plots(all_summaries, all_model_stats, output_dir / "plots")

    # ══════════════════════════════════════════════════════════════════════
    # 9. FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - benchmark_start

    print(f"\n{'='*60}")
    print("Benchmark complete.")
    print(f"{'='*60}")
    print()
    print(f"  Dataset:")
    print(f"    MBPP prompts: {len(unique_prompts)}")
    print(f"    RouterBench models: {len(runner.catalog_keys)}")
    print()
    print(f"  Full Router:")
    for s in all_summaries:
        if s.get("configuration", "").startswith("baseline"):
            continue
        label = s["configuration"]
        print(f"    {label}:")
        print(f"      success rate: {s['routed_task_success_rate']:.1%}")
        print(f"      avg cost:     {s['avg_actual_cost']:.6f}")
        print(f"      avg utility:  {s['avg_router_utility']:.4f}")
        print(f"      avg regret:   {s['avg_regret']:.4f}")
        print()

    print(f"  Baselines:")
    for s in all_summaries:
        if not s.get("configuration", "").startswith("baseline"):
            continue
        label = s["configuration"]
        print(f"    {label}:")
        print(f"      success rate: {s['routed_task_success_rate']:.1%}")
        print(f"      avg cost:     {s['avg_actual_cost']:.6f}")
        print(f"      avg regret:   {s['avg_regret']:.4f}")
        print()

    # Gating ablation comparison
    mixed_with = next((s for s in all_summaries if s.get("configuration") == "mixed"), None)
    mixed_without = next((s for s in all_summaries if s.get("configuration") == "mixed_no_gating"), None)
    if mixed_with and mixed_without:
        print(f"  Gating ablation (mixed mode):")
        print(f"    with gating:    success={mixed_with['routed_task_success_rate']:.1%}, "
              f"regret={mixed_with['avg_regret']:.4f}")
        print(f"    without gating: success={mixed_without['routed_task_success_rate']:.1%}, "
              f"regret={mixed_without['avg_regret']:.4f}")
        print()

    print(f"  Output files: {output_dir}/")
    print(f"  Total time: {elapsed:.1f}s")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
