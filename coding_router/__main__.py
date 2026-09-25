"""Command-line entry point for the coding-router package."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .router import CodingRouter, RouterConfig, MODE_WEIGHTS
from .catalog import add_user_model
from .config import (
    SERVICE_ENV_MAP,
    _is_local_service,
    bundled_catalog_path,
    default_index_path,
    default_user_models_path,
    env_vars_for_service,
    find_api_key,
    load_json,
    resolve_catalog_path,
)
from .embeddings import DEFAULT_EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Subcommand: init
# ---------------------------------------------------------------------------


def init_command(arguments: list[str]) -> int:
    """Set up a project-local router/ directory with the model catalog."""
    router_dir = Path("router")
    catalog_path = router_dir / "coding_llm.json"
    env_example_path = Path(".env.example")

    # 1. Create router/ directory
    router_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy the bundled catalog template (never overwrite user edits)
    if catalog_path.exists():
        print(f"  ✓ Catalog already exists: {catalog_path}  (skipped, not overwriting)")
    else:
        bundled = bundled_catalog_path()
        if not bundled.exists():
            print(f"  ✗ Bundled template not found at {bundled}", file=sys.stderr)
            return 1
        catalog_path.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  ✓ Created: {catalog_path}")

    # 3. Generate .env.example from the catalog's service fields
    data = load_json(catalog_path)
    needed_vars: dict[str, str] = {}  # var_name → service_name
    for group_name, models in data.items():
        if not isinstance(models, dict):
            continue
        for key, record in models.items():
            if not isinstance(record, dict):
                continue
            service = record.get("service", "")
            if _is_local_service(service):
                continue
            for var in env_vars_for_service(service):
                if var not in needed_vars:
                    needed_vars[var] = service

    if needed_vars:
        lines = ["# Required API keys for coding-router", "#"]
        for var, service in sorted(needed_vars.items()):
            lines.append(f"# {service}")
            lines.append(f"{var}=")
        env_example_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  ✓ Created: {env_example_path}")
    else:
        print("  ℹ No cloud API keys needed (only local models in catalog)")

    # 4. Print next steps
    print()
    print("  Next steps:")
    print(f"    1. Edit {catalog_path} — delete models you don't have access to")
    print(f"    2. Set the required env vars (see {env_example_path})")
    print("    3. Run: coding-router validate")
    print()
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------


def validate_command(arguments: list[str]) -> int:
    """Validate the catalog and check API key availability."""
    parser = argparse.ArgumentParser(
        prog="coding-router validate",
        description="Validate the model catalog and check API keys / endpoints.",
    )
    parser.add_argument("--catalog", type=Path, default=None)
    args = parser.parse_args(arguments)

    try:
        catalog_path = resolve_catalog_path(args.catalog)
    except FileNotFoundError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        return 1

    # Load and validate JSON structure
    try:
        data = load_json(catalog_path)
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON in {catalog_path}: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print(f"  ✗ {catalog_path}: Expected a JSON object at the top level", file=sys.stderr)
        return 1

    print(f"  Catalog: {catalog_path}\n")

    required_fields = ("model", "service", "api_endpoint", "feature", "size", "input_price", "output_price")
    has_errors = False
    cloud_ok = 0
    cloud_missing = 0
    local_ok = 0
    local_unreachable = 0

    for group_name, models in data.items():
        if not isinstance(models, dict):
            continue
        for key, record in models.items():
            if not isinstance(record, dict):
                print(f"  ✗ {key}: Expected a JSON object, got {type(record).__name__}", file=sys.stderr)
                has_errors = True
                continue

            # Check required fields
            missing_fields = [f for f in required_fields if f not in record]
            if missing_fields:
                print(
                    f"  ✗ {key}: Missing required fields: {', '.join(missing_fields)}",
                    file=sys.stderr,
                )
                has_errors = True
                continue

            service = record.get("service", "")
            model_id = record.get("model", key)

            if _is_local_service(service):
                # Best-effort ping the endpoint
                endpoint = record.get("api_endpoint", "")
                reachable = False
                if endpoint:
                    try:
                        url = endpoint.rstrip("/") + "/models"
                        req = urllib.request.Request(url, method="GET")
                        urllib.request.urlopen(req, timeout=3)
                        reachable = True
                    except Exception:
                        pass

                if reachable:
                    print(f"  ✓ {model_id} ({service}) — endpoint reachable")
                    local_ok += 1
                else:
                    print(f"  ⚠ {model_id} ({service}) — endpoint unreachable: {endpoint}")
                    local_unreachable += 1
            else:
                # Cloud model: check API key
                api_key = find_api_key(service)
                candidate_vars = env_vars_for_service(service)
                if api_key:
                    print(f"  ✓ {model_id} ({service}) — key found")
                    cloud_ok += 1
                else:
                    var_hint = " or ".join(candidate_vars) if candidate_vars else "???"
                    print(f"  ✗ {model_id} ({service}) — set {var_hint}")
                    cloud_missing += 1
                    has_errors = True

    print()
    print(f"  Cloud: {cloud_ok} ready, {cloud_missing} missing keys")
    print(f"  Local: {local_ok} reachable, {local_unreachable} unreachable")

    if has_errors:
        print()
        print("  ⚠ Some models are missing API keys. Set them and re-run validate.")
        return 1

    print()
    print("  ✓ All models validated successfully!")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: models  (list all models with status)
# ---------------------------------------------------------------------------


def models_command(arguments: list[str]) -> int:
    """List all available models and their status."""
    parser = argparse.ArgumentParser(
        prog="coding-router models",
        description="List all models in the catalog.",
    )
    parser.add_argument("--catalog", type=Path, default=None)
    args = parser.parse_args(arguments)

    try:
        catalog_path = resolve_catalog_path(args.catalog)
    except FileNotFoundError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        return 1

    data = load_json(catalog_path)
    print(f"  Catalog: {catalog_path}\n")

    cloud_models = []
    local_models = []

    for group_name, models in data.items():
        if not isinstance(models, dict):
            continue
        for key, record in models.items():
            if not isinstance(record, dict):
                continue
            service = record.get("service", "?")
            entry = {
                "key": key,
                "service": service,
                "size": record.get("size", "?"),
                "input_price": record.get("input_price", 0),
                "output_price": record.get("output_price", 0),
            }
            if _is_local_service(service):
                local_models.append(entry)
            else:
                local_models.append(entry) if _is_local_service(service) else cloud_models.append(entry)

    def _print_table(title: str, entries: list[dict]) -> None:
        if not entries:
            return
        print(f"  {title}")
        print(f"  {'─' * 80}")
        for e in entries:
            price = (
                f"${e['input_price']:.2f} / ${e['output_price']:.2f} per 1M tokens"
                if e["input_price"] > 0 or e["output_price"] > 0
                else "free (self-hosted)"
            )
            print(f"    {e['key']:<30s}  {e['service']:<28s}  {price}")
        print()

    _print_table("☁️  Cloud Models", cloud_models)
    _print_table("🖥️  Local / Self-Hosted Models", local_models)

    total = len(cloud_models) + len(local_models)
    print(f"  {total} models in catalog")
    print()
    print("  To remove a model, delete its entry from the catalog JSON.")
    print("  To add a model:  coding-router add-model --help")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: add-model
# ---------------------------------------------------------------------------


def add_model_command(arguments: list[str]) -> int:
    """Add a model the user can actually invoke to the persisted candidate pool."""
    parser = argparse.ArgumentParser(description="Add a local or cloud model to the router.")
    parser.add_argument("--key", required=True, help="unique lowercase key, e.g. my-local-coder")
    parser.add_argument("--model", required=True, help="model ID accepted by the endpoint")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="provider base URL; OpenAI-compatible by default",
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
    print(f"  ✓ Added '{args.key}' to {user_catalog}")
    print("  Index will rebuild automatically on next route.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: route (default)
# ---------------------------------------------------------------------------


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
        choices=("llama-server", "local", "vllm"),
        default="llama-server",
        help="llama-server (default) auto-starts llama serve; local loads GGUF in-process; vllm connects to a remote server",
    )
    parser.add_argument("--classifier-model-path", help="path to a custom GGUF classifier model")
    parser.add_argument("--classifier-base-url", default="http://127.0.0.1:8080", help="server URL (for --classifier-backend=vllm)")
    parser.add_argument("--classifier-model", help="server model ID; auto-detected when omitted")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="output-token budget used for both cost estimation and invocation",
    )
    parser.add_argument("--show-ranking", action="store_true", help="Include every similarity score in output")
    args = parser.parse_args(arguments)

    try:
        catalog_path = resolve_catalog_path(args.catalog)
    except FileNotFoundError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        return 1

    router = CodingRouter(
        RouterConfig(
            catalog_path=catalog_path,
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


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


SUBCOMMANDS = {
    "init": init_command,
    "validate": validate_command,
    "models": models_command,
    "add-model": add_model_command,
}


def main() -> int:
    if len(sys.argv) > 1:
        subcommand = sys.argv[1]
        if subcommand in SUBCOMMANDS:
            return SUBCOMMANDS[subcommand](sys.argv[2:])
    return route_command(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
