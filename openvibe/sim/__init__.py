"""Process stress-testing harness for openvibe.

The harness takes a process description, uses the LLM to scaffold a minimal
realistic environment on disk, runs openvibe against it, then evaluates the
run using the process blueprint and filesystem state as evidence.

Nothing is hardcoded — environments and success criteria are derived
dynamically from the process goal.

Quick start::

    from openvibe.llm import LiteLLMBackend
    from openvibe.sim import ProcessHarness, ProcessSpec

    harness = ProcessHarness(llm=LiteLLMBackend())
    run = await harness.run(ProcessSpec(
        name="p2p",
        goal="Process purchase requisitions: validate against vendor list, "
             "generate POs for approved items, log rejections with reasons.",
        context="Finance workflow, manufacturing company.",
        difficulty="complex",
    ))
    print(run.to_markdown())
"""

from openvibe.sim.spec import EvalResult, ProcessRun, ProcessSpec
from openvibe.sim.env_builder import EnvironmentBuilder
from openvibe.sim.runner import ProcessRunner
from openvibe.sim.evaluator import ProcessEvaluator
from openvibe.sim.harness import CatalogReport, ProcessHarness, run_process

__all__ = [
    # Core types
    "ProcessSpec",
    "ProcessRun",
    "EvalResult",
    "CatalogReport",
    # Components
    "EnvironmentBuilder",
    "ProcessRunner",
    "ProcessEvaluator",
    # Harness
    "ProcessHarness",
    "run_process",
]
