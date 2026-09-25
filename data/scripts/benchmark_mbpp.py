"""Benchmark the ModelRouter against RouterBench MBPP prompts.

Reads prompts from ``data/routerbench_raw_mbpp.csv`` (zip-archived CSV with
columns: index, sample_id, prompt, eval_name, performance, model_response,
model_name, cost), routes each prompt through the router in three modes
(skill_based, mixed, cost_efficient), and exports the results to a CSV.

Usage:
    python -m data.scripts.benchmark_mbpp                     # defaults
    python -m data.scripts.benchmark_mbpp --limit 100         # first 100 prompts
    python -m data.scripts.benchmark_mbpp --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so the model_router package resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model_router.app import ModelRouter, RouterConfig, RoutingMode  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────
MODES: list[RoutingMode] = ["skill_based", "mixed", "cost_efficient"]
DATA_FILE = PROJECT_ROOT / "data" / "routerbench_raw_mbpp.csv"
BENCHMARK_CATALOG = PROJECT_ROOT / "data" / "benchmark_models.json"
BENCHMARK_INDEX = PROJECT_ROOT / "data" / "benchmark_embeddings_qwen3_0.6b.npz"


def load_prompts(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:
    """Load prompts from the routerbench MBPP file.

    The file may be a plain CSV **or** a zip archive containing one CSV.
    """
    raw_bytes: bytes

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                # Fall back to the first file in the archive.
                csv_names = zf.namelist()
            raw_bytes = zf.read(csv_names[0])
    else:
        raw_bytes = path.read_bytes()

    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8", errors="replace")))
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def run_benchmark(
    prompts: list[dict[str, str]],
    modes: list[RoutingMode],
    *,
    route_only: bool = True,
    catalog_path: Path = BENCHMARK_CATALOG,
    index_path: Path = BENCHMARK_INDEX,
) -> list[dict[str, object]]:
    """Route every prompt through the router for each mode and collect results."""

    # Build the router once — the expensive index/embedding init is shared.
    # When benchmarking, use the benchmark catalog (legacy models from RouterBench)
    # so that embeddings and similarity scores match the actual candidate pool.
    config = RouterConfig(
        catalog_path=catalog_path,
        user_catalog_path=None,
        index_path=index_path,
    )
    router = ModelRouter(config)
    print(f"Using catalog: {catalog_path.name}")
    total = len(prompts) * len(modes)
    results: list[dict[str, object]] = []

    for prompt_idx, row in enumerate(prompts):
        prompt_text = row.get("prompt", "")
        if not prompt_text.strip():
            continue

        for mode in modes:
            step = prompt_idx * len(modes) + modes.index(mode) + 1
            print(
                f"\r[{step}/{total}] mode={mode:<16s} sample_id={row.get('sample_id', '?')}",
                end="",
                flush=True,
            )

            start = time.perf_counter()
            try:
                state = router.route(
                    prompt_text,
                    route_only=route_only,
                    routing_mode=mode,
                )
                elapsed = time.perf_counter() - start

                selected = state["selected_model"]
                results.append(
                    {
                        # ── original data ──
                        "sample_id": row.get("sample_id", ""),
                        "eval_name": row.get("eval_name", ""),
                        "prompt": prompt_text,
                        # ── routing decision ──
                        "routing_mode": mode,
                        "alpha": selected.get("alpha"),
                        "beta": selected.get("beta"),
                        "selected_model": selected.get("catalog_key", ""),
                        "selection_group": state.get("selection_group", ""),
                        "selection_reason": state.get("selection_reason", ""),
                        "classifier_category": state.get("classifier_category", ""),
                        # ── scoring ──
                        "similarity": selected.get("similarity"),
                        "perf_normalized": selected.get("perf_normalized"),
                        "reward": selected.get("reward"),
                        "estimated_cost_usd": selected.get("estimated_request_cost_usd"),
                        "cost_normalized": selected.get("cost_normalized"),
                        "category_boost": selected.get("category_boost"),
                        # ── meta ──
                        "latency_s": round(elapsed, 4),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - start
                results.append(
                    {
                        "sample_id": row.get("sample_id", ""),
                        "eval_name": row.get("eval_name", ""),
                        "prompt": prompt_text,
                        "routing_mode": mode,
                        "alpha": "",
                        "beta": "",
                        "selected_model": "",
                        "selection_group": "",
                        "selection_reason": "",
                        "classifier_category": "",
                        "similarity": "",
                        "perf_normalized": "",
                        "reward": "",
                        "estimated_cost_usd": "",
                        "cost_normalized": "",
                        "category_boost": "",
                        "latency_s": round(elapsed, 4),
                        "error": str(exc),
                    }
                )

    print()  # newline after progress
    return results


def write_csv(results: list[dict[str, object]], output_path: Path) -> None:
    if not results:
        print("No results to write.")
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows → {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the ModelRouter on RouterBench MBPP prompts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_FILE,
        help="Path to routerbench_raw_mbpp.csv (plain or zipped)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/mbpp_benchmark_<timestamp>.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of prompts to process (default: all)",
    )
    parser.add_argument(
        "--invoke",
        action="store_true",
        help="Actually invoke the selected model (default: route only)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=BENCHMARK_CATALOG,
        help="Model catalog JSON to use (default: benchmark_models.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = PROJECT_ROOT / "data" / f"mbpp_benchmark_{stamp}.csv"

    print(f"Loading prompts from {args.input} …")
    prompts = load_prompts(args.input, limit=args.limit)
    print(f"Loaded {len(prompts)} prompts.")

    if not prompts:
        print("No prompts found — check the input file.")
        return 1

    print(f"Routing through modes: {', '.join(MODES)}")
    print(f"Route only: {not args.invoke}")
    print()

    # Derive the index path from the catalog path: same directory, same naming convention.
    index_path = args.catalog.parent / f"{args.catalog.stem}_embeddings_qwen3_0.6b.npz"
    results = run_benchmark(
        prompts, MODES,
        route_only=not args.invoke,
        catalog_path=args.catalog,
        index_path=index_path,
    )
    write_csv(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
