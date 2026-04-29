"""Model router — select the appropriate model tier for each LLM call.

The router wraps LiteLLM's model string construction and adds
complexity-aware tier selection on top.  It is instantiated once per
OpenVibe instance and shared across all sessions.

Example config (openvibe.json)::

    {
      "model": {"provider_id": "anthropic", "model_id": "claude-opus-4-6"},
      "model_tiers": {
        "fast":     {"provider_id": "anthropic", "model_id": "claude-haiku-4-5"},
        "balanced": {"provider_id": "anthropic", "model_id": "claude-sonnet-4-6"}
      }
    }

With the above config, simple queries use haiku (~20× cheaper than opus),
moderate queries use sonnet, and complex queries use opus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openvibe.config import Config, ModelRef
from openvibe.routing.classifier import Tier, classify


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RoutingPlan:
    """The resolved model and tier for one LLM iteration."""

    tier: Tier
    model_str: str   # litellm model string, e.g. "anthropic/claude-haiku-4-5"
    model_id: str    # short display name for notifications, e.g. "claude-haiku-4-5"
    reason: str      # human-readable explanation, shown in TUI
    switched: bool = False  # True if this differs from the previous plan


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def _to_litellm(ref: ModelRef) -> str:
    """Convert a ModelRef to a litellm model string."""
    return f"{ref.provider_id}/{ref.model_id}"


@dataclass
class ModelRouter:
    """Routes each LLM call to the appropriate model based on task complexity.

    Instantiate once per OpenVibe instance.  Call :meth:`select` before
    every ``llm.stream()`` call to get the model string to use.

    The router is a no-op when no cheaper tiers are configured — it returns
    the primary model unchanged.
    """

    config: Config
    # Tracks the model used in the previous iteration for change detection.
    _last_model_id: str | None = field(default=None, repr=False)
    _primary_str: str = field(default="", repr=False)
    _primary_id: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.config.model:
            self._primary_str = _to_litellm(self.config.model)
            self._primary_id = self.config.model.model_id
        else:
            self._primary_str = "anthropic/claude-sonnet-4-6"
            self._primary_id = "claude-sonnet-4-6"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when at least one cheaper tier is configured."""
        return bool(
            self.config.routing.enabled
            and (
                self.config.model_tiers.fast
                or self.config.model_tiers.balanced
            )
        )

    @property
    def tiers_summary(self) -> str:
        """Human-readable summary of configured tiers for /routing status."""
        tiers = self.config.model_tiers
        parts = [f"primary: {self._primary_id}"]
        if tiers.balanced:
            parts.append(f"balanced: {tiers.balanced.model_id}")
        if tiers.fast:
            parts.append(f"fast: {tiers.fast.model_id}")
        if not self.enabled:
            parts.append("(routing disabled — no tiers configured)")
        return " | ".join(parts)

    def select(self, prompt: str, history_len: int = 0) -> RoutingPlan:
        """Select the best model for *prompt*.

        Parameters
        ----------
        prompt:
            The user's message text to classify.
        history_len:
            Number of messages in the conversation so far.
            Long histories nudge toward stronger models.

        Returns
        -------
        A :class:`RoutingPlan` with the litellm model string and metadata.
        """
        if not self.enabled:
            plan = RoutingPlan(
                tier=Tier.COMPLEX,
                model_str=self._primary_str,
                model_id=self._primary_id,
                reason="no tiers configured — using primary model",
            )
        else:
            tier = classify(
                prompt,
                history_len=history_len,
                use_routellm=self.config.routing.use_routellm,
            )
            plan = self._resolve(tier)

        # Detect model switches for user notification.
        plan.switched = (
            self._last_model_id is not None
            and self._last_model_id != plan.model_id
        )
        self._last_model_id = plan.model_id
        return plan

    def apply_local_tiers(self, suggestions: dict[str, ModelRef]) -> None:
        """Dynamically apply discovered local model tiers to this router.

        Called after user approval of auto-discovered ollama models.
        Does NOT persist to config file — use ``/routing setup`` for that.
        """
        if "fast" in suggestions:
            self.config.model_tiers.fast = suggestions["fast"]
        if "balanced" in suggestions:
            self.config.model_tiers.balanced = suggestions["balanced"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, tier: Tier) -> RoutingPlan:
        """Map a complexity tier to a model, falling back up the chain."""
        tiers = self.config.model_tiers

        if tier == Tier.SIMPLE and tiers.fast:
            ref = tiers.fast
            return RoutingPlan(
                tier=tier,
                model_str=_to_litellm(ref),
                model_id=ref.model_id,
                reason=f"simple task → fast tier ({ref.model_id})",
            )

        if tier in (Tier.SIMPLE, Tier.MODERATE) and tiers.balanced:
            ref = tiers.balanced
            return RoutingPlan(
                tier=tier,
                model_str=_to_litellm(ref),
                model_id=ref.model_id,
                reason=f"moderate task → balanced tier ({ref.model_id})",
            )

        # Complex tier or no cheaper model available — fall through to primary.
        return RoutingPlan(
            tier=tier,
            model_str=self._primary_str,
            model_id=self._primary_id,
            reason=f"complex task → primary model ({self._primary_id})",
        )
