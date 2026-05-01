"""EnvironmentBuilder — LLM-driven sandbox setup.

Given a process description, the builder asks the LLM to generate a shell
script that creates the minimal realistic input environment the process
needs to start from.

Key constraints passed to the LLM:
- Only create INPUT files (what the process operates on)
- Do NOT create expected OUTPUTS (the agent must produce those)
- Use realistic content — not Lorem ipsum, not placeholder text
- Keep it small: the agent should be able to complete the process in one session

The generated script is executed in the temp directory. If the script fails,
we fall back to a minimal scaffold so the run still proceeds.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

from openvibe.llm import LLMBackend, Message, StreamDone, TextDelta
from openvibe.sim.spec import ProcessSpec

logger = logging.getLogger(__name__)

_BUILDER_SYSTEM = """\
You are setting up a minimal, realistic test environment for an AI process agent.
Your job is to write a bash script that creates the input files and state
the process needs to START from — not the expected outputs.

Rules:
1. ONLY create input/source files — the agent must produce the outputs itself.
2. Use realistic content: real-looking data, sensible structure, plausible values.
3. Keep it minimal: 3 to 8 files is ideal. More is not better.
4. Make it appropriately messy: real-world inputs have edge cases, inconsistencies,
   and missing information that the agent must handle.
5. The script must be self-contained and runnable as: bash setup.sh
6. Do NOT include any explanation — output ONLY the bash script.
"""

_BUILDER_PROMPT = """\
Process goal:
{goal}

Additional context:
{context}

Write a bash script that creates the minimal input environment for this process.
The script will be run in an empty temp directory — create everything relative to cwd.
"""

_ADVERSARIAL_EXTRA = """\

This is an adversarial test. Plant at least one subtle trap:
- A file that looks correct but has a silent data quality issue
- A config with a plausible-but-wrong default
- A naming inconsistency (e.g., two files that should match but use different IDs)
The trap should be realistic — something that would catch a careless agent.
"""


class EnvironmentBuilder:
    """Generates and runs the setup script for a process environment."""

    def __init__(self, llm: LLMBackend, model: str) -> None:
        self._llm = llm
        self._model = model

    async def build(self, spec: ProcessSpec, tmp_dir: Path) -> str:
        """Generate and execute the setup script. Returns the script text."""
        script = await self._generate_script(spec)
        self._execute(script, tmp_dir)
        return script

    async def _generate_script(self, spec: ProcessSpec) -> str:
        prompt = _BUILDER_PROMPT.format(
            goal=spec.goal,
            context=spec.context or "(none)",
        )
        if spec.difficulty == "adversarial":
            prompt += _ADVERSARIAL_EXTRA

        messages = [Message(role="user", content=prompt)]
        chunks: list[str] = []

        async for event in self._llm.stream(
            model=self._model,
            messages=messages,
            system=_BUILDER_SYSTEM,
        ):
            if isinstance(event, TextDelta):
                chunks.append(event.content)

        script = "".join(chunks).strip()

        # Strip markdown fences if the LLM added them
        if script.startswith("```"):
            lines = script.splitlines()
            script = "\n".join(
                l for l in lines
                if not l.startswith("```")
            ).strip()

        return script

    def _execute(self, script: str, tmp_dir: Path) -> None:
        """Write and run setup.sh in tmp_dir."""
        script_path = tmp_dir / "setup.sh"
        script_path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                ["bash", "setup.sh"],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "setup.sh exited %d:\n%s", result.returncode, result.stderr[:500]
                )
            else:
                logger.info("Environment built in %s", tmp_dir)
        except subprocess.TimeoutExpired:
            logger.warning("setup.sh timed out after 30s")
        except Exception as exc:
            logger.warning("setup.sh failed: %s", exc)
