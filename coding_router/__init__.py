"""coding-router — lightweight LLM router for coding requests.

Classifies a coding request, semantically matches it against a model
catalog, and selects the best-fit model using cost-aware scoring.

Quick start::

    from coding_router import CodingRouter, RouterConfig

    router = CodingRouter()
    result = router.route("Fix this race condition", route_only=True)
    print(result["selected_model"]["catalog_key"])
"""

from .router import CodingRouter, RouterConfig, RoutingMode, MODE_WEIGHTS

__all__ = ["CodingRouter", "RouterConfig", "RoutingMode", "MODE_WEIGHTS"]
__version__ = "0.1.0"
