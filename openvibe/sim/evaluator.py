"""ProcessEvaluator — LLM-based assessment of a process run.

Evidence used:
1. The process blueprint (what steps the agent took, in order, with purposes)
2. The filesystem delta (what files exist now that didn't before setup.sh)
3. The agent's final response text

The evaluator asks the LLM: given the goal, the blueprint, and the resulting
state, how well did the process execute? Score 0-1 and explain why.

This replaces hardcoded criteria. The LLM understands the intent of the goal
and can judge whether the outputs make sense — something deterministic rules
cannot do for arbitrary processes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openvibe.llm import LLMBackend, Message, StreamDone, TextDelta
from openvibe.sim.spec import EvalResult, ProcessSpec

logger = logging.getLogger(__name__)

_EVAL_SYSTEM = """\
You are evaluating how well an AI process agent executed a real-world process.
You have three pieces of evidence:
1. The goal (what the agent was asked to do)
2. The process blueprint (the sequence of steps the agent took)
3. The resulting filesystem (what files exist after the run)

Evaluate along these dimensions:
- Completeness: did the agent accomplish the full goal, or just part of it?
- Correctness: are the outputs sensible and accurate, or superficially correct?
- Process quality: did the agent verify its work, handle edge cases, avoid shortcuts?
- Efficiency: did it take a reasonable number of steps, or loop unnecessarily?

Be honest and specific. If outputs are missing or wrong, say so.
Reply with valid JSON only — no markdown fences, no prose outside the JSON.
"""

_EVAL_PROMPT = """\
Goal:
{goal}

Process blueprint (steps taken):
{blueprint_text}

Files present after the run (relative paths):
{file_listing}

Agent's final response:
{agent_text}

Score this run. Return JSON:
{{
  "score": <float 0.0-1.0>,
  "passed": <true if score >= 0.7>,
  "summary": "<one paragraph honest assessment>",
  "strengths": ["<specific thing done well>", ...],
  "gaps": ["<specific missing or wrong thing>", ...]
}}
"""


class ProcessEvaluator:
    """Evaluates a process run using the LLM as judge."""

    def __init__(self, llm: LLMBackend, model: str) -> None:
        self._llm = llm
        self._model = model

    async def evaluate(
        self,
        spec: ProcessSpec,
        blueprint: object | None,
        tmp_dir: Path,
        agent_text: str,
    ) -> EvalResult:
        blueprint_text = _format_blueprint(blueprint)
        file_listing = _list_outputs(tmp_dir)

        prompt = _EVAL_PROMPT.format(
            goal=spec.goal,
            blueprint_text=blueprint_text,
            file_listing=file_listing,
            agent_text=agent_text[:2000],
        )

        messages = [Message(role="user", content=prompt)]
        chunks: list[str] = []

        async for event in self._llm.stream(
            model=self._model,
            messages=messages,
            system=_EVAL_SYSTEM,
        ):
            if isinstance(event, TextDelta):
                chunks.append(event.content)

        raw = "".join(chunks).strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = "\n".join(
                l for l in raw.splitlines() if not l.startswith("```")
            ).strip()

        try:
            data = json.loads(raw)
            return EvalResult(
                score=float(data.get("score", 0.0)),
                passed=bool(data.get("passed", False)),
                summary=data.get("summary", ""),
                strengths=data.get("strengths", []),
                gaps=data.get("gaps", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Evaluator JSON parse failed: %s\nRaw: %s", exc, raw[:300])
            # Best-effort fallback: extract score from text
            score = _extract_score_fallback(raw)
            return EvalResult(
                score=score,
                passed=score >= 0.7,
                summary=raw[:500],
                gaps=["evaluation response was malformed"],
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_blueprint(bp: object | None) -> str:
    if bp is None:
        return "(no blueprint — agent made no tool calls)"
    nodes = getattr(bp, "nodes", {})
    if not nodes:
        return "(empty blueprint)"
    lines = [f"Goal: {getattr(bp, 'goal', '(unknown)')}"]
    for n in nodes.values():
        status = getattr(n, "status", "?")
        action = getattr(n, "action", "")
        purpose = getattr(n, "purpose", "")
        branch = getattr(n, "branch", 0)
        branch_tag = f" [branch {branch}]" if branch else ""
        lines.append(f"  [{status}]{branch_tag} {action}")
        if purpose:
            lines.append(f"    ↳ {purpose}")
    return "\n".join(lines)


def _list_outputs(tmp_dir: Path) -> str:
    """List all files in the directory, excluding setup.sh and hidden files."""
    files = []
    for p in sorted(tmp_dir.rglob("*")):
        if p.is_file() and p.name not in {"setup.sh"} and not p.name.startswith("."):
            rel = p.relative_to(tmp_dir)
            size = p.stat().st_size
            files.append(f"  {rel}  ({size} bytes)")
    if not files:
        return "  (no files produced)"
    return "\n".join(files)


def _extract_score_fallback(text: str) -> float:
    """Try to pull a score float from unstructured text."""
    import re
    matches = re.findall(r"\b0\.\d+\b|\b1\.0\b", text)
    if matches:
        return float(matches[0])
    return 0.0
