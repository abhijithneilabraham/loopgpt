"""Model routing — select the cheapest model tier for each task.

Usage::

    from openvibe.routing import ModelRouter, Tier
    router = ModelRouter(config)
    plan = router.select(user_prompt, history_len=5)
    # plan.model_str  → "anthropic/claude-haiku-4-5"
    # plan.tier       → Tier.SIMPLE
    # plan.reason     → "simple task → fast tier"
"""

from openvibe.routing.classifier import Tier, classify
from openvibe.routing.router import ModelRouter, RoutingPlan

__all__ = ["ModelRouter", "RoutingPlan", "Tier", "classify"]
