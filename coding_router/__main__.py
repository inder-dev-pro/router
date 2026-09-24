"""Command-line entry point for the coding-router package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .router import CodingRouter, RouterConfig, MODE_WEIGHTS
from .catalog import add_user_model
from .config import default_catalog_path, default_index_path, default_user_models_path
from .embeddings import DEFAULT_EMBEDDING_MODEL


def add_model_command(arguments: list[str]) -> int:
    """Add a model the user can actually invoke to the persisted candidate pool."""
    parser = argparse.ArgumentParser(description="Add a local or cloud model to the router.")
    parser.add_argument("--key", required=True, help="unique lowercase key, e.g. my-local-coder")
    parser.add_argument("--model", required=True, help="model ID accepted by the endpoint")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="provider base URL; OpenAI-compatible by default, Anthropic/Google when selected",
    )
    parser.add_argument("--feature", required=True, help="capabilities and ideal tasks; used for similarity")
    parser.add_argument("--service", default="OpenAI-compatible")
    parser.add_argument("--size", default="Unknown")
    parser.add_argument("--input-price", type=float, required=True, help="USD per million input tokens")
    parser.add_argument("--output-price", type=float, required=True, help="USD per million output tokens")
    parser.add_argument("--tier", choices=("standard", "advanced"), default="standard")
    parser.add_argument("--api-key-env", help="environment variable containing this model's API key")
    parser.add_argument("--user-catalog", type=Path, default=None)
    args = parser.parse_args(arguments)

    user_catalog = args.user_catalog or default_user_models_path()
    add_user_model(
        user_catalog,
        catalog_key=args.key,
        model=args.model,
        service=args.service,
        api_endpoint=args.endpoint,
        feature=args.feature,
        size=args.size,
        input_price=args.input_price,
        output_price=args.output_price,
        tier=args.tier,
        api_key_env=args.api_key_env,
    )
    print(f"Added '{args.key}' to {user_catalog}. Rebuilds automatically on next route.")
    return 0


def init_command(arguments: list[str]) -> int:
    """Copy the bundled model catalog template to the user's config directory."""
    from .config import default_catalog_path as _init_catalog
    path = _init_catalog()
    print(f"Model catalog initialized at: {path}")
    print("Edit this file to uncomment the models you want to use.")
    return 0


def route_command(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="coding-router",
        description="Classify, semantically route, and optionally invoke a coding LLM.",
    )
    parser.add_argument("query", nargs="?", help="The user request to route")
    parser.add_argument("--route-only", action="store_true", help="Select a model without invoking it")
    parser.add_argument("--force-advanced", action="store_true", help="Use the advanced model subgroup")
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_WEIGHTS),
        default="mixed",
        help="Routing policy: skill_based, quality_leaning, mixed, cost_sensitive, or cost_efficient",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override performance weight α (0.0–1.0)")
    parser.add_argument("--beta", type=float, default=None, help="Override cost weight β (0.0–1.0)")
    parser.add_argument(
        "--quality-gap-threshold",
        type=float,
        default=None,
        help="Quality-gap threshold for the cost-aware decision rule (default: 0.05)",
    )
    parser.add_argument("--build-index", action="store_true", help="Create or refresh the persisted embedding index")
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--user-catalog", type=Path, default=None)
    parser.add_argument(
        "--pool",
        choices=("all", "user"),
        default="all",
        help="all includes the bundled catalog and user models; user uses only your added models",
    )
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    # Classifier backend options
    parser.add_argument(
        "--classifier-backend",
        choices=("local", "vllm"),
        default="local",
        help="local runs GGUF locally via llama-cpp-python; vllm connects to a remote server",
    )
    parser.add_argument("--classifier-model-path", help="path to a custom GGUF classifier model")
    parser.add_argument("--classifier-base-url", default="http://127.0.0.1:8080", help="vLLM server URL (for --classifier-backend=vllm)")
    parser.add_argument("--classifier-model", help="vLLM model ID; auto-detected when omitted")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="output-token budget used for both cost estimation and invocation",
    )
    parser.add_argument("--show-ranking", action="store_true", help="Include every similarity score in output")
    args = parser.parse_args(arguments)

    router = CodingRouter(
        RouterConfig(
            catalog_path=args.catalog or default_catalog_path(),
            user_catalog_path=args.user_catalog or default_user_models_path(),
            index_path=args.index or default_index_path(),
            embedding_model=args.embedding_model,
            classifier_backend=args.classifier_backend,
            classifier_model_path=args.classifier_model_path,
            classifier_base_url=args.classifier_base_url,
            classifier_model=args.classifier_model,
            candidate_pool=args.pool,
            routing_mode=args.mode,
            target_max_tokens=args.max_output_tokens,
        )
    )
    if args.build_index:
        status = "rebuilt" if router.index_rebuilt else "already current"
        index_path = args.index or default_index_path()
        print(f"Embedding index {status}: {index_path}")
        if not args.query:
            return 0
    if not args.query:
        parser.error("query is required unless --build-index is used")

    result = router.route(
        args.query,
        route_only=args.route_only,
        force_advanced=args.force_advanced,
        routing_mode=args.mode,
        alpha=args.alpha,
        beta=args.beta,
        quality_gap_threshold=args.quality_gap_threshold,
        estimated_output_tokens=args.max_output_tokens,
    )

    selected = result["selected_model"]
    output: dict[str, object] = {
        "mode": args.mode,
        "alpha": selected.get("alpha"),
        "beta": selected.get("beta"),
        "candidate_pool": args.pool,
        "category": result["classifier_category"],
        "classifier_model": result.get("classifier_model"),
        "classifier_raw_response": result.get("classifier_raw_response"),
        "classifier_error": result.get("classifier_error"),
        "selection_group": result["selection_group"],
        "selection_reason": result["selection_reason"],
        "selected_model": selected,
        "model_response": result.get("model_response"),
        "invocation_error": result.get("invocation_error"),
    }
    if args.show_ranking:
        output["ranking"] = result["ranked_models"]
    print(json.dumps(output, indent=2))
    return 0 if not result.get("invocation_error") else 2


def main() -> int:
    if len(sys.argv) > 1:
        subcommand = sys.argv[1]
        if subcommand == "add-model":
            return add_model_command(sys.argv[2:])
        if subcommand == "init":
            return init_command(sys.argv[2:])
    return route_command(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
