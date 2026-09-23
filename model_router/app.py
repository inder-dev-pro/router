"""LangGraph workflow: embed → classify → rank → group-select → invoke.

Scoring follows the LLMRouter paper (Feng et al., 2026):

    π* = argmax_π  E[ perf(y | q) − λ · c(τ) ]        (Eq. 1)

For single-turn routing this becomes:

    reward_m = α · perf_normalized(m) − β · cost_normalized(m)

where (α, β) are the performance–cost trade-off weights swept across
five operating points defined in Table 2 / Figure 5 of the paper.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from .catalog import ModelProfile, catalog_fingerprint, load_catalogs
from .classifier import Classification, ClassifierError, VLLMClassifier
from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingIndex, QwenEmbeddingEncoder
from .invocation import ModelInvocationError, invoke_selected_model


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "coding_llm_models.json"
DEFAULT_USER_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "user_models.json"
DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "model_embeddings_qwen3_0.6b.npz"
RoutingMode = Literal[
    "skill_based",
    "quality_leaning",
    "mixed",
    "cost_sensitive",
    "cost_efficient",
]


class RouterState(TypedDict, total=False):
    user_query: str
    route_only: bool
    force_advanced: bool
    force_no_gating: bool  # benchmark-only: skip standard/advanced filtering
    routing_mode: RoutingMode
    alpha: float
    beta: float
    quality_gap_threshold: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    query_embedding: list[float]
    classifier_category: str
    classifier_raw_response: str
    classifier_model: str
    classifier_error: str
    ranked_models: list[dict[str, Any]]
    selection_group: str
    selection_reason: str
    selected_model: dict[str, Any]
    model_response: str
    invocation_error: str


@dataclass(frozen=True)
class RouterConfig:
    catalog_path: Path = DEFAULT_CATALOG_PATH
    user_catalog_path: Path | None = DEFAULT_USER_CATALOG_PATH
    index_path: Path = DEFAULT_INDEX_PATH
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    classifier_base_url: str = "http://127.0.0.1:8080"
    classifier_model: str | None = None
    classifier_timeout: int = 30
    target_timeout: int = 90
    candidate_pool: Literal["all", "user"] = "all"
    routing_mode: RoutingMode = "mixed"
    alpha: float | None = None
    beta: float | None = None
    quality_gap_threshold: float = 0.05
    target_max_tokens: int = 1024


def _category_boost(profile: ModelProfile, category: str) -> float:
    """Small deterministic tie-breaker; semantic similarity remains dominant."""
    text = profile.document().lower()
    terms = {
        "code_generation": ("generation", "completion", "instruct", "coding"),
        "bug_fixing": ("debug", "reasoning", "swe-bench", "agentic", "repository"),
        "performance_optimization": ("performance", "efficient", "optimization", "agentic"),
        "api_help": ("api", "tool-use", "agentic", "integration"),
        "programming": ("general-purpose", "coding", "chat", "instruction"),
    }
    matches = sum(term in text for term in terms.get(category, ()))
    return min(matches * 0.01, 0.03)


COMPLEX_WORK_PATTERN = re.compile(
    r"\b(architecture|architect|multi[- ]file|codebase|repository|refactor|migration|migrate|"
    r"redesign|long[- ]horizon|production incident|security audit|threat model|agentic|"
    r"deep analysis|complex|plan an? implementation)\b",
    re.IGNORECASE,
)


# The paper's five evaluation sweep points from Table 2 / Figure 5.
# Each maps to (α, β) where the reward is:  α · perf − β · cost.
# «skill_based» is quality-only (λ=0); «cost_efficient» is heavily
# cost-weighted; the three in between are the paper's interior points.
MODE_WEIGHTS: dict[RoutingMode, tuple[float, float]] = {
    "skill_based":     (1.0, 0.0),   # (α, β) = (1.0, 0.0)  — quality only
    "quality_leaning": (0.8, 0.2),   # (α, β) = (0.8, 0.2)
    "mixed":           (0.6, 0.4),   # (α, β) = (0.6, 0.4)
    "cost_sensitive":  (0.4, 0.6),   # (α, β) = (0.4, 0.6)
    "cost_efficient":  (0.2, 0.8),   # (α, β) = (0.2, 0.8)  — cost dominant
}


def mode_weights(
    mode: RoutingMode,
    alpha_override: float | None = None,
    beta_override: float | None = None,
) -> tuple[float, float]:
    """Return (α, β) weights for the paper's reward formula.

    If explicit overrides are given they take precedence, enabling
    arbitrary sweep points via the CLI ``--alpha`` / ``--beta`` flags.
    """
    try:
        alpha, beta = MODE_WEIGHTS[mode]
    except KeyError as error:
        raise ValueError(f"Unsupported routing mode: {mode}") from error
    if alpha_override is not None:
        alpha = alpha_override
    if beta_override is not None:
        beta = beta_override
    return alpha, beta


def _normalize(values: list[float], *, flat_value: float) -> list[float]:
    """Min-max normalize a candidate-pool value without fabricating a preference."""
    lower, upper = min(values), max(values)
    if math.isclose(lower, upper):
        return [flat_value] * len(values)
    return [(value - lower) / (upper - lower) for value in values]


def estimate_request_cost_usd(
    profile: ModelProfile, input_tokens: int, output_tokens: int
) -> float:
    """Estimate the request cost from the catalog's USD-per-million token rates."""
    return (
        profile.input_price * max(0, input_tokens)
        + profile.output_price * max(0, output_tokens)
    ) / 1_000_000


class RouterServices:
    """Dependencies captured by LangGraph nodes; created once per process."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self.profiles = load_catalogs(
            config.catalog_path,
            config.user_catalog_path,
            candidate_pool=config.candidate_pool,
        )
        self.profile_by_key = {profile.catalog_key: profile for profile in self.profiles}
        self.encoder = QwenEmbeddingEncoder(config.embedding_model)
        self.index, self.index_rebuilt = EmbeddingIndex.ensure(
            path=config.index_path,
            profiles=self.profiles,
            fingerprint=catalog_fingerprint(
                (config.catalog_path, config.user_catalog_path or Path("<none>")),
                f"{config.embedding_model}\0pool={config.candidate_pool}",
            ),
            encoder=self.encoder,
        )
        self.classifier = VLLMClassifier(
            base_url=config.classifier_base_url,
            model=config.classifier_model,
            timeout=config.classifier_timeout,
        )

    def embed_query(self, state: RouterState) -> dict[str, Any]:
        embedding = self.encoder.encode_query(state["user_query"])
        return {"query_embedding": embedding.astype(float).tolist()}

    def classify_query(self, state: RouterState) -> dict[str, Any]:
        try:
            classification: Classification = self.classifier.classify(state["user_query"])
            return {
                "classifier_category": classification.category,
                "classifier_raw_response": classification.raw_response,
                "classifier_model": classification.model,
            }
        except ClassifierError as error:
            # Similarity routing can still work during a temporary classifier outage.
            return {
                "classifier_category": "programming",
                "classifier_error": str(error),
            }

    def rank_models(self, state: RouterState) -> dict[str, Any]:
        """Score every candidate with the paper's Eq. 1 for single-turn routing.

        The scoring function g is cosine similarity between E_q(q) and
        E_m(m) plus a small category tie-breaker.  The decision rule d
        selects argmax of the weighted reward::

            reward_m = α · perf_norm(m) − β · cost_norm(m)

        Both perf and cost are min-max normalised across the active
        candidate pool so α and β are directly interpretable operating
        points identical to the paper's five-point sweep (Table 2).
        """
        category = state["classifier_category"]
        scores = self.index.cosine_scores(state["query_embedding"])

        # ── capability-fit proxy: g(E_q, E_m) + category boost ────────
        raw_capability_fit = [
            cosine_score + _category_boost(profile, category)
            for profile, cosine_score in zip(self.profiles, scores, strict=True)
        ]

        # ── cost term: estimated USD cost for this request ─────────────
        input_tokens = state["estimated_input_tokens"]
        output_tokens = state["estimated_output_tokens"]
        request_costs = [
            estimate_request_cost_usd(profile, input_tokens, output_tokens)
            for profile in self.profiles
        ]

        # ── min-max normalise both to [0, 1] ──────────────────────────
        perf_normalized = _normalize(raw_capability_fit, flat_value=1.0)
        cost_normalized = _normalize(request_costs, flat_value=0.0)

        # ── (α, β) from the named mode or explicit CLI overrides ──────
        alpha, beta = mode_weights(
            state["routing_mode"],
            alpha_override=state.get("alpha"),
            beta_override=state.get("beta"),
        )

        # ── Paper Eq. 1:  reward = α · perf − β · cost ────────────────
        ranked: list[dict[str, Any]] = []
        for profile, cosine_score, raw_q, perf_n, cost_usd, cost_n in zip(
            self.profiles,
            scores,
            raw_capability_fit,
            perf_normalized,
            request_costs,
            cost_normalized,
            strict=True,
        ):
            boost = _category_boost(profile, category)
            reward = alpha * perf_n - beta * cost_n
            ranked.append(
                {
                    "catalog_key": profile.catalog_key,
                    "similarity": round(cosine_score, 6),
                    "category_boost": round(boost, 6),
                    "capability_fit_raw": round(raw_q, 6),
                    "perf_normalized": round(perf_n, 6),
                    "estimated_request_cost_usd": round(cost_usd, 8),
                    "cost_normalized": round(cost_n, 6),
                    "alpha": alpha,
                    "beta": beta,
                    "reward": round(reward, 6),
                }
            )
        ranked.sort(
            key=lambda item: (
                item["reward"],
                item["perf_normalized"],
                -item["estimated_request_cost_usd"],
            ),
            reverse=True,
        )
        return {"ranked_models": ranked}

    def choose_model_group(self, state: RouterState) -> dict[str, Any]:
        query = state["user_query"]
        advanced = state.get("force_advanced", False) or bool(COMPLEX_WORK_PATTERN.search(query))
        if len(query) > 2_000:
            advanced = True
        # Skill-first routing can select an advanced candidate if semantic fit
        # identifies one, even when the request has no explicit complexity cue.
        if (
            state["routing_mode"] == "skill_based"
            and self.profile_by_key[state["ranked_models"][0]["catalog_key"]].advanced
        ):
            advanced = True
        return {
            "selection_group": "advanced_model_group" if advanced else "standard_model_group",
            "selection_reason": "forced by caller" if state.get("force_advanced") else (
                "advanced semantic match or complex request"
                if advanced
                else "standard request"
            ),
        }

    def _select_model(self, state: RouterState, advanced: bool) -> dict[str, Any]:
        """Pick the best candidate within the group, applying the quality-gap rule.

        The paper's cost-aware decision rule (§2.1) thresholds the quality
        gap between the top candidate and cheaper alternatives.  When the
        gap is below ``quality_gap_threshold`` the cheaper model wins.
        """
        if state.get("force_no_gating"):
            # Benchmark-only: consider all candidates regardless of group.
            matches = state["ranked_models"]
        else:
            matches = [
                item
                for item in state["ranked_models"]
                if self.profile_by_key[item["catalog_key"]].advanced is advanced
            ]
            # The catalog could later contain no models in one subgroup. In that case,
            # preserve service instead of failing a valid user request.
            if not matches:
                matches = state["ranked_models"]

        # ── Quality-gap cost-aware decision rule ──────────────────────
        threshold = state.get("quality_gap_threshold", 0.05)
        best = matches[0]
        selection = best
        if threshold > 0 and len(matches) > 1:
            for candidate in matches[1:]:
                perf_gap = best["perf_normalized"] - candidate["perf_normalized"]
                if (
                    perf_gap < threshold
                    and candidate["estimated_request_cost_usd"]
                    < best["estimated_request_cost_usd"]
                ):
                    selection = candidate
                    break  # first cheaper model within the gap wins

        profile = self.profile_by_key[selection["catalog_key"]]
        selected = profile.public_dict() | {
            "similarity": selection["similarity"],
            "perf_normalized": selection["perf_normalized"],
            "reward": selection["reward"],
            "category_boost": selection["category_boost"],
            "estimated_request_cost_usd": selection["estimated_request_cost_usd"],
            "cost_normalized": selection["cost_normalized"],
            "alpha": selection["alpha"],
            "beta": selection["beta"],
        }
        return {"selected_model": selected}

    def select_advanced_model(self, state: RouterState) -> dict[str, Any]:
        return self._select_model(state, advanced=True)

    def select_standard_model(self, state: RouterState) -> dict[str, Any]:
        return self._select_model(state, advanced=False)

    def invoke_target_model(self, state: RouterState) -> dict[str, Any]:
        if state.get("route_only"):
            return {}
        profile = self.profile_by_key[state["selected_model"]["catalog_key"]]
        try:
            response = invoke_selected_model(
                profile,
                state["user_query"],
                state["classifier_category"],
                timeout=self.config.target_timeout,
                max_tokens=state["estimated_output_tokens"],
            )
            return {"model_response": response}
        except ModelInvocationError as error:
            return {"invocation_error": str(error)}


def build_router_graph(services: RouterServices):
    """Build the concrete LangGraph DAG with explicit advanced/standard nodes."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as error:
        raise RuntimeError(
            "Missing LangGraph. Install project dependencies with: pip install -r requirements.txt"
        ) from error

    graph = StateGraph(RouterState)
    graph.add_node("embed_query", services.embed_query)
    graph.add_node("classify_query", services.classify_query)
    graph.add_node("similarity_search", services.rank_models)
    graph.add_node("choose_model_group", services.choose_model_group)
    graph.add_node("advanced_model_group", services.select_advanced_model)
    graph.add_node("standard_model_group", services.select_standard_model)
    graph.add_node("invoke_target_model", services.invoke_target_model)

    graph.add_edge(START, "embed_query")
    graph.add_edge("embed_query", "classify_query")
    graph.add_edge("classify_query", "similarity_search")
    graph.add_edge("similarity_search", "choose_model_group")
    graph.add_conditional_edges(
        "choose_model_group",
        lambda state: state["selection_group"],
        {
            "advanced_model_group": "advanced_model_group",
            "standard_model_group": "standard_model_group",
        },
    )
    graph.add_edge("advanced_model_group", "invoke_target_model")
    graph.add_edge("standard_model_group", "invoke_target_model")
    graph.add_edge("invoke_target_model", END)
    return graph.compile()


class ModelRouter:
    """Public facade for building the index and routing a single user request."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self.services = RouterServices(self.config)
        self.graph = build_router_graph(self.services)

    @property
    def index_rebuilt(self) -> bool:
        return self.services.index_rebuilt

    def route(
        self,
        query: str,
        *,
        route_only: bool = False,
        force_advanced: bool = False,
        force_no_gating: bool = False,
        routing_mode: RoutingMode | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        quality_gap_threshold: float | None = None,
        estimated_output_tokens: int | None = None,
    ) -> RouterState:
        if not query.strip():
            raise ValueError("The user query cannot be empty.")
        mode = routing_mode or self.config.routing_mode
        if mode not in MODE_WEIGHTS:
            raise ValueError(f"Unsupported routing mode: {mode}")
        output_tokens = estimated_output_tokens or self.config.target_max_tokens
        if output_tokens <= 0:
            raise ValueError("Estimated output tokens must be positive.")

        # Resolve (α, β) — CLI overrides > config overrides > mode defaults
        effective_alpha = alpha if alpha is not None else self.config.alpha
        effective_beta = beta if beta is not None else self.config.beta
        gap = quality_gap_threshold if quality_gap_threshold is not None else self.config.quality_gap_threshold

        initial_state: RouterState = {
            "user_query": query,
            "route_only": route_only,
            "force_advanced": force_advanced,
            "force_no_gating": force_no_gating,
            "routing_mode": mode,
            "estimated_input_tokens": max(1, math.ceil(len(query) / 4)),
            "estimated_output_tokens": output_tokens,
            "quality_gap_threshold": gap,
        }
        if effective_alpha is not None:
            initial_state["alpha"] = effective_alpha
        if effective_beta is not None:
            initial_state["beta"] = effective_beta
        return self.graph.invoke(initial_state)
