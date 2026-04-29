"""MouseTool — control the mouse pointer via pynput.

pynput provides reliable, cross-platform mouse control with native OS APIs:
  macOS  — Quartz Event Services
  Linux  — Xlib (X11) or evdev (Wayland)
  Windows — SendInput (Win32)

Supports click, double-click, right-click, middle-click, move, scroll, and drag.
Coordinates are auto-scaled from image pixels to logical screen pixels so that
Retina / HiDPI displays work correctly when coordinates come from a screenshot.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import Field

from openvibe.tool.base import Tool, ToolContext, ToolResult


class MouseTool(Tool):
    """Move, click, scroll, or drag the mouse pointer."""

    name = "mouse"
    description = (
        "Control the mouse: move, click (left/right/middle), double-click, "
        "scroll, or drag. Coordinates are in screenshot image pixels — always "
        "provide image_width and image_height so Retina/HiDPI scaling is handled "
        "automatically."
    )

    class Params(Tool.Params):
        action: Literal[
            "move", "click", "double_click", "right_click", "middle_click",
            "scroll", "drag",
        ] = Field(
            description=(
                "Mouse action:\n"
                "  move         — move pointer to (x, y) without clicking\n"
                "  click        — left-click at (x, y)\n"
                "  double_click — double left-click at (x, y)\n"
                "  right_click  — right-click at (x, y)\n"
                "  middle_click — middle-button click at (x, y)\n"
                "  scroll       — scroll at (x, y); positive amount = up\n"
                "  drag         — drag from (x, y) to (end_x, end_y)"
            )
        )
        x: int = Field(description="X coordinate in screenshot image pixels.")
        y: int = Field(description="Y coordinate in screenshot image pixels.")
        end_x: int | None = Field(
            default=None,
            description="Target X for drag (image pixels).",
        )
        end_y: int | None = Field(
            default=None,
            description="Target Y for drag (image pixels).",
        )
        image_width: int | None = Field(
            default=None,
            description=(
                "Width of the screenshot image the coordinates were taken from. "
                "ALWAYS provide this — it enables automatic Retina/HiDPI scaling. "
                "Use the width reported by the screenshot tool."
            ),
        )
        image_height: int | None = Field(
            default=None,
            description="Height of the screenshot image. Provide alongside image_width.",
        )
        button: Literal["left", "right", "middle"] = Field(
            default="left",
            description="Mouse button for click actions (default: left).",
        )
        amount: int = Field(
            default=3,
            description="Scroll ticks (positive = up, negative = down). Used only for scroll.",
        )
        duration: float = Field(
            default=0.2,
            description="Smooth-movement duration in seconds.",
        )
        settle_ms: int = Field(
            default=500,
            description=(
                "Milliseconds to wait after the action for the UI to settle. "
                "Increase to 1000–2000 ms for slow apps or animations."
            ),
        )

    async def execute(self, ctx: ToolContext, params: "MouseTool.Params") -> ToolResult:
        from openvibe.computer.sandbox import ActionType, get_sandbox
        from openvibe.computer.input import screen_size

        # ── Retina / HiDPI coordinate scaling ─────────────────────────────
        # Screenshots may be downscaled to ≤1920 px wide; logical screen
        # pixels can differ (e.g. 1440-wide on a Retina MacBook).
        scaled = params
        scale_note = ""
        if params.image_width and params.image_height:
            try:
                sw, sh = await asyncio.get_event_loop().run_in_executor(None, screen_size)
                sx = sw / params.image_width
                sy = sh / params.image_height
                if abs(sx - 1.0) > 0.02 or abs(sy - 1.0) > 0.02:
                    scaled = params.model_copy(update={
                        "x": round(params.x * sx),
                        "y": round(params.y * sy),
                        "end_x": round(params.end_x * sx) if params.end_x is not None else None,
                        "end_y": round(params.end_y * sy) if params.end_y is not None else None,
                    })
                    scale_note = (
                        f" [scaled ({params.x},{params.y})→({scaled.x},{scaled.y})"
                        f" @ {sx:.3f}×{sy:.3f}]"
                    )
            except Exception:
                pass  # best-effort; never block the action
        # ──────────────────────────────────────────────────────────────────

        sandbox = get_sandbox(ctx.session_id)
        if not sandbox.is_pre_approved("mouse"):
            await ctx.check_permission(
                tool="mouse",
                argument=f"{params.action} at ({params.x}, {params.y})",
                description=f"Mouse {params.action} at ({params.x}, {params.y})",
            )
        if not sandbox.is_coordinate_allowed(scaled.x, scaled.y):
            return ToolResult(
                title="Mouse action denied",
                output=(
                    f"({scaled.x}, {scaled.y}) is outside the permitted screen region."
                ),
                error=True,
            )

        action_params = {
            "action": params.action, "x": params.x, "y": params.y,
            "amount": params.amount, "duration": params.duration,
        }

        try:
            loop = asyncio.get_event_loop()
            result_msg = await loop.run_in_executor(None, self._do_action, scaled)
        except Exception as exc:
            await sandbox.record_action(ActionType.MOUSE_CLICK, params=action_params, error=str(exc))
            return ToolResult(title="Mouse error", output=str(exc), error=True)

        action_type_map = {
            "move": ActionType.MOUSE_MOVE,
            "click": ActionType.MOUSE_CLICK,
            "double_click": ActionType.MOUSE_CLICK,
            "right_click": ActionType.MOUSE_CLICK,
            "middle_click": ActionType.MOUSE_CLICK,
            "scroll": ActionType.MOUSE_SCROLL,
            "drag": ActionType.MOUSE_DRAG,
        }
        await sandbox.record_action(
            action_type_map.get(params.action, ActionType.MOUSE_CLICK),
            params=action_params,
            result=result_msg,
        )
        return ToolResult(
            title=f"Mouse: {params.action}",
            output=result_msg + scale_note,
        )

    @staticmethod
    def _do_action(params: "MouseTool.Params") -> str:
        from openvibe.computer import input as inp

        if params.action == "move":
            return inp.mouse_move(params.x, params.y, params.duration, params.settle_ms)

        if params.action == "click":
            return inp.mouse_click(params.x, params.y, "left", 1, params.duration, params.settle_ms)

        if params.action == "double_click":
            return inp.mouse_click(params.x, params.y, "left", 2, params.duration, params.settle_ms)

        if params.action == "right_click":
            return inp.mouse_click(params.x, params.y, "right", 1, params.duration, params.settle_ms)

        if params.action == "middle_click":
            return inp.mouse_click(params.x, params.y, "middle", 1, params.duration, params.settle_ms)

        if params.action == "scroll":
            return inp.mouse_scroll(params.x, params.y, dy=params.amount, settle_ms=params.settle_ms)

        if params.action == "drag":
            if params.end_x is None or params.end_y is None:
                raise ValueError("end_x and end_y are required for drag.")
            return inp.mouse_drag(
                params.x, params.y,
                params.end_x, params.end_y,
                params.duration, params.settle_ms,
            )

        raise ValueError(f"Unknown mouse action: {params.action!r}")
