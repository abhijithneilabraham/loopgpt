"""Local model discovery via Ollama.

Discovers models available locally and maps them to routing tiers
so openvibe can use cheaper open-source models when no cloud tiers
are configured.

Usage::

    if is_ollama_available():
        available = list_ollama_models()
        suggestions = suggest_local_tiers(available)
        # suggestions → {"fast": ModelRef(provider_id="ollama", model_id="qwen2.5:3b"), ...}
"""

from __future__ import annotations

import subprocess

from openvibe.config import ModelRef


# Preferred models per tier — checked in order; first match wins.
# Chosen for quality/size tradeoff: fast ≤ 4B params, balanced ≤ 9B params.
_TIER_PREFS: dict[str, list[str]] = {
    "fast": [
        "qwen2.5:3b",
        "llama3.2:3b",
        "phi3:mini",
        "phi3.5:mini",
        "gemma2:2b",
        "gemma:2b",
        "mistral:7b-instruct-q4_0",  # quantized fits in 4 GB VRAM
    ],
    "balanced": [
        "llama3.1:8b",
        "qwen2.5:7b",
        "mistral:7b",
        "gemma2:9b",
        "llama3:8b",
        "qwen2:7b",
        "deepseek-r1:8b",
    ],
}


def is_ollama_available() -> bool:
    """Return True if the `ollama` CLI is installed and responsive."""
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def list_ollama_models() -> list[str]:
    """Return the names of all locally available ollama models."""
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return []
        # First line is a header ("NAME  ID  SIZE  MODIFIED"), skip it.
        lines = r.stdout.strip().splitlines()[1:]
        return [line.split()[0] for line in lines if line.strip()]
    except Exception:
        return []


def suggest_local_tiers(available: list[str]) -> dict[str, ModelRef]:
    """Return the best available local model for each tier.

    Parameters
    ----------
    available:
        Model names as returned by :func:`list_ollama_models`.

    Returns
    -------
    Dict mapping tier name (``"fast"``, ``"balanced"``) to a
    :class:`~openvibe.config.ModelRef` with ``provider_id="ollama"``.
    """
    suggestions: dict[str, ModelRef] = {}
    for tier, prefs in _TIER_PREFS.items():
        for pref in prefs:
            # Allow partial match: "qwen2.5:3b" matches "qwen2.5:3b-instruct-q4_0"
            base = pref.split(":")[0]
            match = next((m for m in available if base in m), None)
            if match:
                suggestions[tier] = ModelRef(provider_id="ollama", model_id=match)
                break
    return suggestions


def run_discovery() -> dict[str, ModelRef]:
    """Convenience wrapper: detect ollama + return tier suggestions.

    Returns an empty dict when ollama is not available or has no
    matching models.
    """
    if not is_ollama_available():
        return {}
    available = list_ollama_models()
    return suggest_local_tiers(available)
