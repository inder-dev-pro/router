"""Run the existing router on benchmark prompts and capture decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_router.app import ModelRouter, RouterConfig, RoutingMode


# Benchmark-only index path — never overwrites the production index.
_BENCHMARK_INDEX = Path(__file__).resolve().parents[2] / "data" / "benchmark_embeddings_qwen3_0.6b.npz"
_BENCHMARK_CATALOG = Path(__file__).resolve().parents[2] / "data" / "benchmark_models.json"


@dataclass
class BenchmarkConfig:
    """Immutable description of one benchmark run configuration."""

    mode: RoutingMode
    alpha: float
    beta: float
    disable_gating: bool = False
    label: str = ""

    def __post_init__(self):
        if not self.label:
            gating_tag = "_no_gating" if self.disable_gating else ""
            object.__setattr__(self, "label", f"{self.mode}{gating_tag}")


@dataclass
class PromptResult:
    """Everything captured for a single prompt routing."""

    sample_id: str = ""
    prompt: str = ""
    eval_name: str = ""

    mode: str = ""
    alpha: float = 0.0
    beta: float = 0.0
    candidate_pool: str = "all"
    disable_gating: bool = False

    router_category: str = ""
    selection_group: str = ""
    selection_reason: str = ""

    classifier_model: str = ""
    classifier_raw_response: str = ""
    classifier_error: str = ""

    selected_model: str = ""
    selected_service: str = ""
    selected_tier: str = ""
    selected_advanced: bool = False

    router_similarity: float = 0.0
    router_capability_normalized: float = 0.0
    router_cost_normalized: float = 0.0
    router_internal_reward: float = 0.0
    router_category_boost: float = 0.0

    router_estimated_cost: float = 0.0
    router_latency_ms: float = 0.0

    error: str = ""

    # Populated post-hoc by oracle/metrics
    selected_benchmark_performance: float = float("nan")
    selected_benchmark_cost: float = float("nan")
    oracle_model: str = ""
    oracle_utility: float = float("nan")
    router_actual_utility: float = float("nan")
    regret: float = float("nan")
    oracle_agreement: bool = False
    max_candidate_performance: float = float("nan")
    task_solvable: bool = True
    routing_success: bool = False
    routing_failure: bool = False
    task_unsolvable: bool = False
    benchmark_lookup_success: bool = False
    benchmark_lookup_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class BenchmarkRunner:
    """Creates and drives the router against benchmark prompts."""

    def __init__(
        self,
        catalog_path: Path = _BENCHMARK_CATALOG,
        index_path: Path = _BENCHMARK_INDEX,
        classifier_base_url: str = "http://localhost:8000/v1",
        classifier_model: str | None = None,
    ) -> None:
        self.catalog_path = catalog_path
        self.config = RouterConfig(
            catalog_path=catalog_path,
            user_catalog_path=None,
            index_path=index_path,
            classifier_base_url=classifier_base_url,
            classifier_model=classifier_model,
            candidate_pool="all",
        )
        self.router = ModelRouter(self.config)
        # Expose catalog keys for verification
        self.catalog_keys = {p.catalog_key for p in self.router.services.profiles}

    def route_prompt(
        self,
        prompt: str,
        *,
        mode: RoutingMode,
        alpha: float,
        beta: float,
        disable_gating: bool = False,
    ) -> dict[str, Any]:
        """Route a single prompt and return the full RouterState dict."""
        start = time.perf_counter()
        state = self.router.route(
            prompt,
            route_only=True,
            routing_mode=mode,
            alpha=alpha,
            beta=beta,
            force_no_gating=disable_gating,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"state": state, "latency_ms": elapsed_ms}

    def run_configuration(
        self,
        prompts: list[dict[str, str]],
        config: BenchmarkConfig,
        *,
        progress: bool = True,
    ) -> list[PromptResult]:
        """Route every unique prompt under a single benchmark configuration."""
        results: list[PromptResult] = []
        total = len(prompts)
        for idx, prompt_row in enumerate(prompts):
            if progress:
                print(
                    f"\r  [{idx + 1}/{total}] {config.label:<24s} "
                    f"sample_id={prompt_row['sample_id']}",
                    end="",
                    flush=True,
                )
            result = PromptResult(
                sample_id=prompt_row["sample_id"],
                prompt=prompt_row["prompt"],
                eval_name=prompt_row.get("eval_name", "mbpp"),
                mode=config.mode,
                alpha=config.alpha,
                beta=config.beta,
                candidate_pool="all",
                disable_gating=config.disable_gating,
            )
            try:
                out = self.route_prompt(
                    prompt_row["prompt"],
                    mode=config.mode,
                    alpha=config.alpha,
                    beta=config.beta,
                    disable_gating=config.disable_gating,
                )
                state = out["state"]
                selected = state["selected_model"]
                result.router_category = state.get("classifier_category", "")
                result.selection_group = state.get("selection_group", "")
                result.selection_reason = state.get("selection_reason", "")
                result.classifier_model = state.get("classifier_model", "")
                result.classifier_raw_response = state.get("classifier_raw_response", "")
                result.classifier_error = state.get("classifier_error", "")
                result.selected_model = selected.get("catalog_key", "")
                result.selected_service = selected.get("service", "")
                result.selected_tier = selected.get("tier", "")
                result.selected_advanced = selected.get("advanced", False)
                result.router_similarity = selected.get("similarity", 0.0)
                result.router_capability_normalized = selected.get("perf_normalized", 0.0)
                result.router_cost_normalized = selected.get("cost_normalized", 0.0)
                result.router_internal_reward = selected.get("reward", 0.0)
                result.router_category_boost = selected.get("category_boost", 0.0)
                result.router_estimated_cost = selected.get("estimated_request_cost_usd", 0.0)
                result.router_latency_ms = out["latency_ms"]
            except Exception as exc:
                result.error = str(exc)
            results.append(result)
        if progress:
            print()
        return results
