"""WatchScreenTool — read live screen activity into the agent's context.

Every mouse and keyboard action automatically records focus state and screen
diffs via the ScreenObserver.  This tool lets the agent explicitly pull the
accumulated observations into its reasoning context — useful when:

  - The agent wants to verify the current screen state before the next action
  - Something unexpected may have happened (notification, dialog, focus steal)
  - The agent needs to know the current focused app/window without taking a
    full screenshot

Usage::

    # Agent calls this between actions to stay oriented:
    watch_screen(mode="summary")
    # → "Recent screen activity (5 events): ..."
    # → "Current focus: App: Chrome | Window: Gmail | Element: AXTextField"

    watch_screen(mode="focus")
    # → Just the current focus state (fast, no screenshot)
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openvibe.tool.base import Tool, ToolContext, ToolResult


class WatchScreenTool(Tool):
    """Read live screen activity and current input focus.

    Does NOT capture a screenshot — that's the screenshot tool's job.
    This tool reads the observer's event buffer: what changed, when, and
    which app/window/element currently has input focus.
    """

    name = "watch_screen"
    description = (
        "Read current screen activity and input focus state without taking a screenshot. "
        "Use this to verify where input focus is, check what changed on screen recently, "
        "or confirm an action landed in the right place. Especially useful before typing "
        "to confirm the correct text field is focused."
    )

    class Params(Tool.Params):
        mode: Literal["focus", "summary", "full"] = Field(
            default="summary",
            description=(
                "What to return:\n"
                "  focus   — current focused app/window/element only (fastest)\n"
                "  summary — recent screen events + current focus (default)\n"
                "  full    — all buffered events + current focus"
            ),
        )

    async def execute(self, ctx: ToolContext, params: "WatchScreenTool.Params") -> ToolResult:
        import asyncio
        from openvibe.computer.observer import get_observer

        observer = get_observer(ctx.session_id)
        loop = asyncio.get_event_loop()

        if params.mode == "focus":
            focus = await loop.run_in_executor(None, observer.current_focus)
            output = f"Current focus: {focus}" if focus else "Current focus: (unavailable)"
            return ToolResult(title="Screen focus", output=output)

        if params.mode == "full":
            n = observer._max_events
        else:
            n = 8

        summary = await loop.run_in_executor(None, observer.get_summary, n)
        focus = observer.current_focus()

        parts = [summary]
        if focus:
            parts.append(f"\nCurrent focus: {focus}")

        return ToolResult(title="Screen activity", output="\n".join(parts))
