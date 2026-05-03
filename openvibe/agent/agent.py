"""Agent definitions.

An agent is a named configuration that controls how the LLM behaves:
which model to use, what system prompt to use, which tools are available,
and what permission rules apply.

Built-in agents
---------------
- **build**    — full-access coding agent for writing, editing, and running code (default)
- **plan**     — read-only agent for analysis and planning
- **general**  — read-only subagent for research and multi-step searches
- **computer** — desktop automation agent (screen + mouse/keyboard)

Custom agents can be defined in ``openvibe.json`` under the ``agent`` key
and will override built-in defaults with the same name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openvibe.config import AgentConfig, AgentMode, ModelRef, PermissionAction
from openvibe.permission.permission import Rule

if TYPE_CHECKING:
    from openvibe.config import Config


# ---------------------------------------------------------------------------
# Agent runtime info
# ---------------------------------------------------------------------------


@dataclass
class AgentInfo:
    """Fully resolved agent configuration used at runtime."""

    name: str
    description: str
    system_prompt: str
    model: ModelRef | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_steps: int | None = None
    mode: AgentMode = AgentMode.PRIMARY
    permission_rules: list[Rule] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)
    extra_instructions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default system prompts
# ---------------------------------------------------------------------------

_STEP_CONTEXT_INSTRUCTION = """\
Before each tool call, write one concise sentence explaining what you are about \
to do and why it is relevant to the overall goal. This helps the user track \
progress through the process.\
"""

_BUILD_SYSTEM_PROMPT = """\
You are openvibe, an AI coding agent. You help users write, edit, debug, and \
understand code — and can execute multi-step workflows involving files, shell \
commands, web resources, and desktop automation from start to finish.

Guidelines:
- Before each tool call, briefly state what you are doing and why (one sentence).
- Think step-by-step. Use the todo tool to track progress on long tasks.
- Read files before editing them; understand existing patterns first.
- Prefer targeted edits (edit tool) over full rewrites (write tool).
- Verify results after key steps.
- Never guess at file paths — use glob or grep to locate files first.
- When in doubt, ask a clarifying question rather than guessing.
"""

_PLAN_SYSTEM_PROMPT = """\
You are openvibe in plan mode — a read-only coding agent. You can explore \
files, search for patterns, and answer questions, but you MUST NOT modify \
files or run shell commands with side effects.

Before each tool call, briefly note what you are looking for and why. \
Provide clear, structured analysis with headings and bullet points.
"""

_GENERAL_SYSTEM_PROMPT = """\
You are a general-purpose research subagent. Gather information, search code, \
fetch web resources, and return findings. Before each tool call, note in one \
sentence what you are retrieving and how it contributes to the goal. \
Do not write or modify files.
"""

_COMPUTER_SYSTEM_PROMPT = """\
You are openvibe in computer-use mode. You can see and control the desktop.

MANDATORY FIRST STEP — PRE-FLIGHT:
  Before taking ANY action, call `pre_flight` with:
    • task: one-sentence description of the goal
    • actions: every tool you plan to use, in order (app, screenshot, ui, mouse, keyboard)
  This collects all permissions upfront in a single step. The user approves once;
  you then proceed without further permission interruptions.
  Do NOT skip pre_flight. Do NOT call any other tool before pre_flight succeeds.

TOOL PRIORITY — after pre_flight, always follow this order:

1. ui tool (FIRST CHOICE — no coordinates needed, most reliable)
   • Use `ui get_tree` to list clickable elements in an app by name.
   • Use `ui click` with the element title — never guess coordinates.
   • Use `ui click_menu` to trigger menu items (File → Save, etc.).
   • Use `ui type` to enter text — handles Unicode and clipboard correctly.
   • Use `ui press_key` for keys/chords (return, escape, cmd+s, etc.).
   • ui is auto-allowed — no permission prompt.

2. app tool — open, close, focus, list applications.

3. screenshot tool — take a screenshot to observe the current screen state.
   Always take one after opening an app to confirm it appeared.
   The output includes the image dimensions — note them for step 4.

4. mouse tool (LAST RESORT — only for unlabelled canvas areas)
   • Only use when `ui get_tree` shows no accessible elements for the target.
   • Use the pixel coordinates directly from the screenshot image — scaling
     is applied automatically, no need to pass image_width or image_height.
   • Example: mouse click x=450 y=300

5. keyboard tool — raw keystroke fallback when `ui type` / `ui press_key`
   cannot be used (rare).

WORKFLOW:
  pre_flight → app open → screenshot → ui get_tree → ui click/type → screenshot → verify

VERIFICATION:
  Every screenshot compares automatically to the previous one and reports
  what percentage of the screen changed. If you see "No visible change
  detected" after an action, the action failed — do NOT repeat it blindly.
  Instead: try ui get_tree to find the element by name, or take a fresh
  screenshot and reassess coordinates.

WHEN AN ELEMENT IS NOT FOUND:
  If `ui click` fails because no element has that exact title:
  1. Call `ui get_tree` on the target app to list ALL available elements.
  2. Find the closest match by appearance or function (not exact label).
  3. Click that element instead.
  If `ui get_tree` returns nothing useful, fall back to mouse click using
  coordinates from a fresh screenshot.

NEVER STOP AND ASK THE USER:
  - If something fails, adapt and retry using a different approach.
  - Recorded element labels are hints, not exact requirements — find
    the element visually or by exploring the accessibility tree.
  - For text inputs: if the label doesn't match, find any visible text
    field near the recorded coordinates and type into it.
  - Keep iterating until the task is complete or all approaches exhausted.

Avoid moving the mouse to extreme screen corners as some systems use corner gestures.
"""


# ---------------------------------------------------------------------------
# Built-in permission rulesets
# ---------------------------------------------------------------------------

_A = PermissionAction  # local alias for brevity

_BUILD_RULES: list[Rule] = [
    # Allow common read tools by default
    Rule(tool="read", action=_A.ALLOW),
    Rule(tool="glob", action=_A.ALLOW),
    Rule(tool="grep", action=_A.ALLOW),
    Rule(tool="web_fetch", action=_A.ALLOW),
    Rule(tool="todo_read", action=_A.ALLOW),
    Rule(tool="todo_write", action=_A.ALLOW),
    # Ask before write operations
    Rule(tool="write", action=_A.ASK),
    Rule(tool="edit", action=_A.ASK),
    Rule(tool="bash", action=_A.ASK),
]

_PLAN_RULES: list[Rule] = [
    Rule(tool="read", action=_A.ALLOW),
    Rule(tool="glob", action=_A.ALLOW),
    Rule(tool="grep", action=_A.ALLOW),
    Rule(tool="web_fetch", action=_A.ALLOW),
    # Deny all write / execute operations
    Rule(tool="write", action=_A.DENY),
    Rule(tool="edit", action=_A.DENY),
    Rule(tool="bash", action=_A.DENY),
    Rule(tool="todo_write", action=_A.DENY),
]

_GENERAL_RULES: list[Rule] = [
    Rule(tool="read", action=_A.ALLOW),
    Rule(tool="glob", action=_A.ALLOW),
    Rule(tool="grep", action=_A.ALLOW),
    Rule(tool="web_fetch", action=_A.ALLOW),
    Rule(tool="write", action=_A.DENY),
    Rule(tool="edit", action=_A.DENY),
    Rule(tool="bash", action=_A.DENY),
]

# Computer-use: screenshot + ui (accessibility) + pre_flight are always allowed;
# raw mouse/keyboard/app require consent (they affect the running system).
_COMPUTER_RULES: list[Rule] = [
    Rule(tool="pre_flight", action=_A.ALLOW),  # Planning step — manages its own permission prompts
    Rule(tool="screenshot", action=_A.ALLOW),
    Rule(tool="ui", action=_A.ALLOW),   # Accessibility API (atomacos/AT-SPI/pywinauto) — preferred
    Rule(tool="mouse", action=_A.ASK),
    Rule(tool="keyboard", action=_A.ASK),
    Rule(tool="app", action=_A.ASK),
    # Standard tools remain available
    Rule(tool="read", action=_A.ALLOW),
    Rule(tool="glob", action=_A.ALLOW),
    Rule(tool="grep", action=_A.ALLOW),
    Rule(tool="bash", action=_A.ASK),
    Rule(tool="write", action=_A.ASK),
    Rule(tool="edit", action=_A.ASK),
]


# ---------------------------------------------------------------------------
# Built-in agent definitions
# ---------------------------------------------------------------------------

_BUILTIN_AGENTS: dict[str, AgentInfo] = {
    "build": AgentInfo(
        name="build",
        description="Full-access coding agent — writes code, edits files, runs commands.",
        system_prompt=_BUILD_SYSTEM_PROMPT,
        mode=AgentMode.PRIMARY,
        permission_rules=_BUILD_RULES,
    ),
    "plan": AgentInfo(
        name="plan",
        description="Read-only coding agent — explores and plans without side effects.",
        system_prompt=_PLAN_SYSTEM_PROMPT,
        mode=AgentMode.PRIMARY,
        permission_rules=_PLAN_RULES,
        disabled_tools=["bash", "write", "edit", "todo_write"],
    ),
    "general": AgentInfo(
        name="general",
        description="General-purpose research subagent — gathers information and returns findings.",
        system_prompt=_GENERAL_SYSTEM_PROMPT,
        mode=AgentMode.SUBAGENT,
        permission_rules=_GENERAL_RULES,
        disabled_tools=["bash", "write", "edit", "todo_write"],
    ),
    "computer": AgentInfo(
        name="computer",
        description=(
            "Computer-use agent: observes the screen and controls mouse/keyboard "
            "to automate desktop workflows. Requires mss, pillow, pynput."
        ),
        system_prompt=_COMPUTER_SYSTEM_PROMPT,
        mode=AgentMode.PRIMARY,
        permission_rules=_COMPUTER_RULES,
        disabled_tools=[],
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(config: "Config", name: str | None = None) -> AgentInfo:
    """Return a fully resolved AgentInfo for *name* (or the default agent).

    User config overrides are merged on top of the built-in defaults.
    Always returns a fresh copy — never mutates the global builtins.
    """
    import dataclasses

    agent_name = name or config.default_agent or "build"

    # Start from built-in or create a shell — always copy to avoid mutating globals.
    builtin = _BUILTIN_AGENTS.get(agent_name)
    if builtin is not None:
        base = dataclasses.replace(builtin)
    else:
        base = AgentInfo(name=agent_name, description="", system_prompt="")

    # Apply user overrides from config
    user_cfg: AgentConfig | None = config.agent.get(agent_name)
    if user_cfg:
        base = _apply_config(base, user_cfg)

    # Apply global model override (config.model) if agent has no model set
    if base.model is None and config.model:
        base.model = config.model

    # Append global instructions
    base.extra_instructions = list(config.instructions)

    return base


def list_agents(config: "Config") -> list[AgentInfo]:
    """Return all available agents (built-in + user-defined)."""
    names = set(_BUILTIN_AGENTS) | set(config.agent)
    return [resolve(config, n) for n in sorted(names)]


def _apply_config(base: AgentInfo, cfg: AgentConfig) -> AgentInfo:
    """Return a copy of *base* with *cfg* overrides applied."""
    import dataclasses

    updates: dict[str, object] = {}

    if cfg.model:
        updates["model"] = cfg.model
    if cfg.description:
        updates["description"] = cfg.description
    if cfg.prompt:
        # Append custom prompt to the built-in system prompt
        updates["system_prompt"] = base.system_prompt + "\n\n" + cfg.prompt
    if cfg.temperature is not None:
        updates["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        updates["top_p"] = cfg.top_p
    if cfg.max_steps is not None:
        updates["max_steps"] = cfg.max_steps
    if cfg.mode:
        updates["mode"] = cfg.mode

    return dataclasses.replace(base, **updates)
