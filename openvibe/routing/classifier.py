"""Task complexity classifier.

Classifies a user prompt into one of three tiers:
  simple   → fast/cheap model (haiku, small local)
  moderate → balanced model (sonnet, medium local)
  complex  → strong model (opus, large local)

Primary strategy: lightweight keyword + length heuristic (zero extra deps).
Optional enhancement: RouteLLM matrix_factorization classifier.
  Install with: pip install routellm
"""

from __future__ import annotations

from enum import StrEnum


class Tier(StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# ── Keyword tables ─────────────────────────────────────────────────────────────

# Signals that the task is simple — a fast model is sufficient.
_SIMPLE_KEYWORDS = [
    "what is", "what are", "list all", "list the", "show me", "tell me",
    "explain briefly", "summarize", "translate", "format", "convert",
    "define", "describe", "what does", "how many", "who is", "when did",
    "give me a", "print the",
]

# Signals that the task is complex — needs the strong model.
_COMPLEX_KEYWORDS = [
    "implement", "build a", "create a system", "architect", "design system",
    "refactor", "rewrite", "migrate", "debug", "why does", "how does",
    "optimize", "review the", "fix the bug", "failing test",
    "multi-step", "entire codebase", "full implementation", "deeply",
    "analyze and", "reason about", "compare and contrast",
]


def classify_heuristic(prompt: str) -> Tier:
    """Classify prompt complexity via keyword matching and token count."""
    lower = prompt.lower().strip()
    words = lower.split()
    n = len(words)

    # Very short prompts are almost always simple questions.
    if n <= 8:
        return Tier.SIMPLE

    # Very long prompts with lots of context usually need more reasoning.
    if n > 200:
        return Tier.COMPLEX

    # Check complex signals first (higher specificity).
    for kw in _COMPLEX_KEYWORDS:
        if kw in lower:
            return Tier.COMPLEX

    # Check simple signals.
    for kw in _SIMPLE_KEYWORDS:
        if kw in lower:
            return Tier.SIMPLE

    # Moderate-length prompts without clear signals default to moderate.
    if n > 60:
        return Tier.COMPLEX
    return Tier.MODERATE


def classify(
    prompt: str,
    history_len: int = 0,
    use_routellm: bool = False,
) -> Tier:
    """Classify prompt complexity.

    Parameters
    ----------
    prompt:
        The raw user message to classify.
    history_len:
        Number of messages in the conversation history so far.
        Long histories shift toward stronger models.
    use_routellm:
        If True, attempt to use RouteLLM's matrix_factorization classifier
        before falling back to the heuristic.
    """
    if use_routellm:
        try:
            tier = _classify_routellm(prompt)
        except Exception:
            tier = classify_heuristic(prompt)
    else:
        tier = classify_heuristic(prompt)

    # Long conversation histories add implicit complexity.
    if history_len > 20 and tier == Tier.SIMPLE:
        tier = Tier.MODERATE

    return tier


def _classify_routellm(prompt: str) -> Tier:
    """Use RouteLLM's MF classifier to decide strong vs weak model."""
    from routellm.controller import Controller  # type: ignore[import-not-found]

    # RouteLLM is model-agnostic; the names here are just labels used
    # internally to calibrate the threshold — they are not called.
    controller = Controller(
        routers=["mf"],
        strong_model="gpt-4o",
        weak_model="gpt-4o-mini",
    )
    chosen: str = controller.route(
        messages=[{"role": "user", "content": prompt}],
        router="mf",
        threshold=0.11588,  # calibrated threshold from the RouteLLM paper
    )
    # weak_model chosen → task is moderate; strong_model → complex
    return Tier.COMPLEX if "gpt-4o-mini" not in chosen else Tier.MODERATE
