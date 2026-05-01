"""Process specification — a description of a real process to stress-test.

A ProcessSpec is nothing more than a goal description and some metadata.
There is no embedded test data, no hardcoded environments, no expected outputs.

The environment is built dynamically from the goal by the EnvironmentBuilder.
The success criteria are derived dynamically by the ProcessEvaluator.
openvibe itself decides how to approach the process.

This means the harness works on ANY process — not just the ones someone
thought to write test cases for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProcessSpec:
    """Describes one process to stress-test.

    The goal is given verbatim to the openvibe agent.
    The context helps the EnvironmentBuilder understand what kind of
    input files or state the process needs to start from.
    """

    name: str
    goal: str                          # prompt given to the agent
    context: str = ""                  # extra background for env setup
    difficulty: str = "complex"        # foundational | complex | adversarial
    tags: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.difficulty}] {self.name}: {self.goal[:80]}"


@dataclass
class EvalResult:
    score: float           # 0.0 – 1.0
    passed: bool
    summary: str           # one-paragraph natural language evaluation
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


@dataclass
class ProcessRun:
    """Everything produced by one execution of a ProcessSpec."""

    spec_name: str
    goal: str
    difficulty: str

    # Execution
    elapsed_seconds: float = 0.0
    tool_calls: int = 0
    dag_branches: int = 0
    agent_text: str = ""
    blueprint_path: str | None = None
    env_dir: str | None = None         # path to the temp environment (kept for inspection)

    # Evaluation
    score: float = 0.0
    passed: bool = False
    evaluation: str = ""
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    # Error if the run itself crashed
    run_error: str | None = None

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"## {self.spec_name}  [{self.difficulty}]",
            f"**{status}** — score: {self.score:.2f}  |  "
            f"tool calls: {self.tool_calls}  |  "
            f"branches: {self.dag_branches}  |  "
            f"time: {self.elapsed_seconds:.1f}s",
        ]
        if self.evaluation:
            lines += ["", self.evaluation]
        if self.strengths:
            lines += ["", "**What worked:**"]
            lines += [f"- {s}" for s in self.strengths]
        if self.gaps:
            lines += ["", "**Gaps:**"]
            lines += [f"- {g}" for g in self.gaps]
        if self.run_error:
            lines += ["", f"**Run error:** {self.run_error}"]
        if self.env_dir:
            lines += ["", f"*Environment preserved at: `{self.env_dir}`*"]
        return "\n".join(lines)
