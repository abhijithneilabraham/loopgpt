"""PreFlightTool — declare and pre-approve all planned computer-use actions.

The agent MUST call this as the very first step of any computer-use task.
It collects all required permissions upfront in a single planning phase so
the task can proceed without being interrupted for every individual action.

Usage::

    pre_flight(
        task="Open TextEdit and write a poem",
        actions=[
            {"tool": "app",        "description": "Open TextEdit"},
            {"tool": "screenshot", "description": "Observe the screen"},
            {"tool": "ui",         "description": "Find and click the document area"},
            {"tool": "keyboard",   "description": "Type the poem text"},
        ]
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from openvibe.tool.base import Tool, ToolContext, ToolResult

# Tools that are always auto-allowed (no permission prompt needed).
_AUTO_ALLOWED = {"screenshot", "ui"}

# Tools that require an explicit user permission check.
_NEEDS_PERMISSION = {"mouse", "keyboard", "app"}


class PreFlightTool(Tool):
    """Mandatory first step for every computer-use task.

    Call this BEFORE any other computer-use tool. Declare every tool you
    plan to use; permissions are collected once upfront so the task runs
    without per-action interruptions.
    """

    name = "pre_flight"
    description = (
        "MANDATORY FIRST STEP for every computer-use task. "
        "Declare all tools you plan to use (app, screenshot, ui, mouse, keyboard) "
        "and collect all required permissions upfront before taking any action. "
        "The user approves the full plan once here rather than being prompted for "
        "every individual mouse click or keystroke."
    )

    class ActionPlan(BaseModel):
        tool: str = Field(
            description=(
                "Tool name to be used: 'app', 'screenshot', 'ui', 'mouse', or 'keyboard'."
            )
        )
        description: str = Field(
            description="Human-readable description of what this tool will do in the task."
        )

    class Params(Tool.Params):
        task: str = Field(
            description="One-sentence description of the overall task to be performed."
        )
        actions: list["PreFlightTool.ActionPlan"] = Field(
            description=(
                "Ordered list of ALL computer-use tools you plan to call during this task. "
                "Include every tool, including screenshot. List them in execution order."
            )
        )

    async def execute(self, ctx: ToolContext, params: "PreFlightTool.Params") -> ToolResult:
        from openvibe.computer.sandbox import get_sandbox
        from openvibe.computer.observer import get_observer
        from openvibe.permission.permission import PermissionDenied, PermissionRejected

        sandbox = get_sandbox(ctx.session_id)

        # Start live screen observer for this session (no-op if already running).
        try:
            observer = get_observer(ctx.session_id)
            observer.start()
        except Exception:
            pass

        approved: list[str] = []
        denied: list[str] = []

        # Deduplicate: collect one representative action per tool.
        seen: set[str] = set()
        unique_actions: list[PreFlightTool.ActionPlan] = []
        for action in params.actions:
            tool_name = action.tool.lower().strip()
            if tool_name not in seen:
                seen.add(tool_name)
                unique_actions.append(
                    PreFlightTool.ActionPlan(tool=tool_name, description=action.description)
                )

        for action in unique_actions:
            tool_name = action.tool

            if tool_name in _AUTO_ALLOWED:
                # Always granted — no prompt needed.
                sandbox.pre_approve(tool_name)
                approved.append(f"{tool_name}: {action.description} [auto-allowed]")
                continue

            if tool_name in _NEEDS_PERMISSION:
                try:
                    await ctx.check_permission(
                        tool=tool_name,
                        argument=action.description,
                        description=f"[{params.task}] {action.description}",
                    )
                    sandbox.pre_approve(tool_name)
                    approved.append(f"{tool_name}: {action.description}")
                except (PermissionDenied, PermissionRejected) as exc:
                    denied.append(f"{tool_name}: {action.description} ({exc})")
                continue

            # Unknown tool — still request permission generically.
            try:
                await ctx.check_permission(
                    tool=tool_name,
                    argument=action.description,
                    description=f"[{params.task}] {action.description}",
                )
                sandbox.pre_approve(tool_name)
                approved.append(f"{tool_name}: {action.description}")
            except (PermissionDenied, PermissionRejected) as exc:
                denied.append(f"{tool_name}: {action.description} ({exc})")

        lines = [f"Pre-flight plan: {params.task}", ""]
        if approved:
            lines.append("Approved:")
            lines.extend(f"  + {a}" for a in approved)
        if denied:
            lines.append("\nDenied:")
            lines.extend(f"  - {d}" for d in denied)

        if denied:
            lines.append(
                "\nSome permissions were denied. Only proceed with approved tools. "
                "Do NOT call any denied tools."
            )
        else:
            lines.append(
                "\nAll permissions granted. Proceed with the task. "
                "You will NOT be prompted again for these tools."
            )

        return ToolResult(
            title=f"Pre-flight: {len(approved)} approved, {len(denied)} denied",
            output="\n".join(lines),
            error=bool(denied),
        )
