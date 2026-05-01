"""ProcessHarness — end-to-end orchestrator for process stress-testing.

Flow for each run:
  1. EnvironmentBuilder asks the LLM to generate setup.sh and runs it
  2. ProcessRunner executes openvibe against the goal in that environment
  3. ProcessEvaluator asks the LLM to score the run using the blueprint + filesystem
  4. ProcessRun is returned with full evidence and score

Nothing is hardcoded. The LLM designs the environment and the evaluation
criteria dynamically from the user's process description.

Quick start::

    from openvibe.llm import LiteLLMBackend
    from openvibe.sim import ProcessHarness, ProcessSpec

    harness = ProcessHarness(llm=LiteLLMBackend())
    run = await harness.run(ProcessSpec(
        name="p2p_procurement",
        goal="Process the pending purchase requisitions: validate each against the "
             "approved vendor list, generate PO documents for approved items, and "
             "log rejections with reasons.",
        context="Finance team workflow for a mid-size manufacturing company.",
        difficulty="complex",
    ))
    print(run.to_markdown())

Run multiple and get a report::

    runs = [ProcessSpec(name="p2p", goal="...", difficulty="complex")]
    report = await harness.run_catalog(runs)
    print(report.to_markdown())
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openvibe.llm import LLMBackend, LiteLLMBackend
from openvibe.sim.env_builder import EnvironmentBuilder
from openvibe.sim.evaluator import ProcessEvaluator
from openvibe.sim.runner import ProcessRunner
from openvibe.sim.spec import EvalResult, ProcessRun, ProcessSpec

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str], None]


class ProcessHarness:
    """Stress-tests a process by running openvibe against an LLM-generated environment."""

    def __init__(
        self,
        llm: LLMBackend | None = None,
        model: str = "",
        config: object | None = None,
        on_token: Callable[[str], None] | None = None,
        on_progress: ProgressCb | None = None,
        keep_env: bool = False,    # if True, don't delete the temp dir after run
        timeout: int = 300,
    ) -> None:
        from openvibe.llm import resolve_model

        self._llm = llm or LiteLLMBackend()
        self._model = model or resolve_model()
        self._config = config
        self._on_token = on_token
        self._on_progress = on_progress or (lambda _: None)
        self._keep_env = keep_env
        self._timeout = timeout

        self._builder = EnvironmentBuilder(self._llm, self._model)
        self._runner = ProcessRunner(config=config, on_token=on_token, timeout_seconds=timeout)
        self._evaluator = ProcessEvaluator(self._llm, self._model)

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    async def run(self, spec: ProcessSpec) -> ProcessRun:
        """Execute one process spec end-to-end and return the evaluated run."""
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"openvibe_sim_{spec.name}_"))
        result = ProcessRun(
            spec_name=spec.name,
            goal=spec.goal,
            difficulty=spec.difficulty,
            env_dir=str(tmp_dir) if self._keep_env else None,
        )

        try:
            # 1. Build environment
            self._progress(f"[{spec.name}] Building environment…")
            await self._builder.build(spec, tmp_dir)

            # 2. Run openvibe
            self._progress(f"[{spec.name}] Running agent…")
            bp, agent_text, tool_calls, branches, elapsed = self._runner.execute(
                spec.goal, tmp_dir
            )
            result.elapsed_seconds = elapsed
            result.tool_calls = tool_calls
            result.dag_branches = branches
            result.agent_text = agent_text

            if self._keep_env:
                blueprint_path = tmp_dir / "process-blueprint.json"
                result.blueprint_path = str(blueprint_path) if blueprint_path.exists() else None

            # 3. Evaluate
            self._progress(f"[{spec.name}] Evaluating…")
            eval_result: EvalResult = await self._evaluator.evaluate(
                spec, bp, tmp_dir, agent_text
            )
            result.score = eval_result.score
            result.passed = eval_result.passed
            result.evaluation = eval_result.summary
            result.strengths = eval_result.strengths
            result.gaps = eval_result.gaps

        except Exception as exc:
            logger.exception("[%s] run crashed", spec.name)
            result.run_error = str(exc)
        finally:
            if not self._keep_env:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        status = "PASS" if result.passed else "FAIL"
        self._progress(
            f"[{spec.name}] {status}  score={result.score:.2f}  "
            f"tool_calls={result.tool_calls}  {result.elapsed_seconds:.1f}s"
        )
        return result

    # ------------------------------------------------------------------
    # Catalog run
    # ------------------------------------------------------------------

    async def run_catalog(
        self,
        specs: list[ProcessSpec],
        abort: asyncio.Event | None = None,
    ) -> "CatalogReport":
        target = specs
        abort = abort or asyncio.Event()
        runs: list[ProcessRun] = []

        for i, spec in enumerate(target, 1):
            if abort.is_set():
                break
            self._progress(f"[{i}/{len(target)}] Starting '{spec.name}'")
            run = await self.run(spec)
            runs.append(run)

        return CatalogReport(runs=runs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _progress(self, msg: str) -> None:
        logger.info(msg)
        self._on_progress(msg)


# ---------------------------------------------------------------------------
# Catalog report
# ---------------------------------------------------------------------------


@dataclass
class CatalogReport:
    runs: list[ProcessRun] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.passed) / len(self.runs)

    @property
    def mean_score(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.score for r in self.runs) / len(self.runs)

    @property
    def by_difficulty(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for r in self.runs:
            buckets.setdefault(r.difficulty, []).append(r.score)
        return {k: sum(v) / len(v) for k, v in buckets.items()}

    def to_markdown(self) -> str:
        lines = [
            "# Process Stress-Test Report",
            "",
            f"**Runs:** {len(self.runs)}  |  "
            f"**Pass rate:** {self.pass_rate:.1%}  |  "
            f"**Mean score:** {self.mean_score:.2f}",
            "",
        ]
        if self.by_difficulty:
            lines.append("**By difficulty:**")
            for diff, score in sorted(self.by_difficulty.items()):
                lines.append(f"  - {diff}: {score:.2f}")
            lines.append("")

        for run in self.runs:
            lines.append(run.to_markdown())
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backwards-compatible sync wrapper
# ---------------------------------------------------------------------------


def run_process(
    goal: str,
    name: str = "process",
    context: str = "",
    difficulty: str = "complex",
    llm: LLMBackend | None = None,
    on_progress: ProgressCb | None = None,
    keep_env: bool = False,
) -> ProcessRun:
    """Blocking wrapper for use from CLI and scripts."""
    spec = ProcessSpec(name=name, goal=goal, context=context, difficulty=difficulty)
    harness = ProcessHarness(llm=llm, on_progress=on_progress, keep_env=keep_env)
    return asyncio.run(harness.run(spec))
