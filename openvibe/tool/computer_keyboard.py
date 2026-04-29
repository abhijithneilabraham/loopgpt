"""KeyboardTool — type text and press key combinations via pynput.

pynput drives native OS input APIs:
  macOS  — Quartz Event Services
  Linux  — Xlib (X11) or evdev (Wayland)
  Windows — SendInput (Win32)

Text is always typed via clipboard paste (pbcopy/xclip/pyperclip + paste hotkey)
so that full Unicode — including CJK, emoji, RTL scripts — works on every platform.

Key names follow a readable convention (case-insensitive):
    enter, escape, tab, space, backspace, delete,
    up, down, left, right, home, end, pageup, pagedown,
    f1–f12, ctrl, shift, alt, cmd, and any single character.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import Field

from openvibe.tool.base import Tool, ToolContext, ToolResult


class KeyboardTool(Tool):
    """Type text or press keyboard keys and shortcuts."""

    name = "keyboard"
    description = (
        "Simulate keyboard input: type a Unicode string, press a single named key, "
        "or send a key combination (hotkey). Click into a text field first, then use "
        "this tool to enter input."
    )

    class Params(Tool.Params):
        action: Literal["type", "press", "hotkey"] = Field(
            description=(
                "Keyboard action:\n"
                "  type   — type a string of text (Unicode-safe, uses clipboard)\n"
                "  press  — press a single key by name, e.g. 'enter', 'f5', 'escape'\n"
                "  hotkey — press a key combination, e.g. keys=['ctrl','c'] for copy,\n"
                "           keys=['ctrl','shift','i'] for DevTools"
            )
        )
        text: str | None = Field(
            default=None,
            description="Text to type. Required for action='type'. Supports full Unicode.",
        )
        key: str | None = Field(
            default=None,
            description=(
                "Key name for action='press'. Examples: 'enter', 'escape', 'tab', "
                "'backspace', 'delete', 'up', 'down', 'f5', 'space'."
            ),
        )
        keys: list[str] | None = Field(
            default=None,
            description=(
                "Ordered list of key names for action='hotkey'. "
                "Examples: ['ctrl','c'], ['ctrl','shift','i'], ['cmd','space']."
            ),
        )
        settle_ms: int = Field(
            default=300,
            description=(
                "Milliseconds to wait after the action for the UI to settle. "
                "Increase for slow apps or when the next action depends on the result."
            ),
        )

    async def execute(self, ctx: ToolContext, params: "KeyboardTool.Params") -> ToolResult:
        from openvibe.computer.sandbox import ActionType, get_sandbox

        if params.action == "type":
            arg_desc = f"type: {(params.text or '')[:60]!r}"
        elif params.action == "press":
            arg_desc = f"press: {params.key!r}"
        else:
            arg_desc = f"hotkey: {'+'.join(params.keys or [])}"

        sandbox = get_sandbox(ctx.session_id)
        if not sandbox.is_pre_approved("keyboard"):
            await ctx.check_permission(
                tool="keyboard",
                argument=arg_desc,
                description=f"Keyboard {arg_desc}",
            )

        try:
            loop = asyncio.get_event_loop()
            result_msg = await loop.run_in_executor(None, self._do_action, params)
        except Exception as exc:
            await sandbox.record_action(
                ActionType.KEYBOARD_TYPE,
                params={"action": params.action},
                error=str(exc),
            )
            return ToolResult(title="Keyboard error", output=str(exc), error=True)

        action_type_map = {
            "type": ActionType.KEYBOARD_TYPE,
            "press": ActionType.KEYBOARD_PRESS,
            "hotkey": ActionType.KEYBOARD_HOTKEY,
        }
        await sandbox.record_action(
            action_type_map.get(params.action, ActionType.KEYBOARD_TYPE),
            params={"action": params.action},
            result=result_msg,
        )
        return ToolResult(title=f"Keyboard: {params.action}", output=result_msg)

    @staticmethod
    def _do_action(params: "KeyboardTool.Params") -> str:
        from openvibe.computer import input as inp

        if params.action == "type":
            if not params.text:
                raise ValueError("text is required for action='type'.")
            return inp.keyboard_type(params.text, params.settle_ms)

        if params.action == "press":
            if not params.key:
                raise ValueError("key is required for action='press'.")
            return inp.keyboard_press(params.key, params.settle_ms)

        if params.action == "hotkey":
            if not params.keys:
                raise ValueError("keys list is required for action='hotkey'.")
            return inp.keyboard_hotkey(params.keys, params.settle_ms)

        raise ValueError(f"Unknown keyboard action: {params.action!r}")
