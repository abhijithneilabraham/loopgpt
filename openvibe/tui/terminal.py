"""Terminal-native chat UI: prompt_toolkit input + rich output.

Output flows inline — every message stays in the terminal's own scrollback
buffer so you can always scroll back and read everything, exactly like a
normal terminal program (no full-screen alternate buffer, no disappearing
content).
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as _e

# ── shared console ─────────────────────────────────────────────────────────
_con = Console(highlight=False, markup=True)

# ── colour palette ─────────────────────────────────────────────────────────
_BRAND  = "#4a9f5a"
_DIM    = "#666666"
_XDIM   = "#3a3a3a"
_WARN   = "#cc8800"
_ERR    = "#cc4444"
_OK     = "#3a7a4a"
_SPIN   = "#cc9900"
_USER   = "#888888"
_RULE   = "#2a2a2a"

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_TOOL_ICON  = {"pending": "○", "running": "◎", "completed": "●", "error": "✗"}
_TOOL_COLOR = {"pending": _XDIM, "running": _SPIN, "completed": _OK, "error": _ERR}


# ── helpers ────────────────────────────────────────────────────────────────

def _hr() -> None:
    _con.rule(style=_RULE)


def _print_tool(state: dict, prev: dict | None = None) -> None:
    """Print a tool-call status line (only on status transitions)."""
    status = state.get("status", "")
    if prev and prev.get("status") == status:
        return
    action  = state.get("action") or state.get("tool_name", "tool")
    icon    = _TOOL_ICON.get(status, "?")
    col     = _TOOL_COLOR.get(status, _DIM)
    purpose = state.get("purpose", "")
    _con.print(f"  [{col}]{icon}[/{col}] [{_DIM}]{_e(action)}[/{_DIM}]")
    if purpose:
        _con.print(f"    [{_XDIM}]↳ {_e(purpose)}[/{_XDIM}]")
    if status in ("completed", "error"):
        out = state.get("output", "") or state.get("error", "")
        if out:
            for line in out[:2000].splitlines():
                _con.print(f"    [{_XDIM}]{_e(line)}[/{_XDIM}]")


def _osc52_copy(text: str) -> None:
    """Copy text to system clipboard via OSC 52 (works in most terminals)."""
    try:
        encoded = base64.b64encode(text.encode()).decode()
        sys.stdout.write(f"\033]52;c;{encoded}\a")
        sys.stdout.flush()
    except Exception:
        pass


# ── main class ─────────────────────────────────────────────────────────────

class TerminalUI:
    """Interactive terminal chat — output flows inline, stays in scrollback."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self._dir    = project_dir or Path.cwd()
        self._ov: Any     = None
        self._session: Any = None
        self._auto   = False
        self._last_assistant = ""
        self._pt: Any = None   # prompt_toolkit PromptSession (lazy)

    # ── entry point ────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            asyncio.run(self._main())
        except KeyboardInterrupt:
            _con.print(f"\n[{_DIM}]Interrupted.[/{_DIM}]")

    # ── startup ────────────────────────────────────────────────────────────

    async def _main(self) -> None:
        from openvibe.tui.app import _needs_setup

        # Clear visible screen (leave scrollback intact — let the terminal own it)
        _con.clear()

        # Check setup
        if _needs_setup(self._dir):
            _con.print(
                f"[{_WARN}]No model configured.[/{_WARN}]  "
                f"[{_DIM}]Edit ~/.config/openvibe/openvibe.json to set your provider.[/{_DIM}]"
            )
            return

        # Start
        _con.print(
            f"[{_BRAND} bold]openvibe[/{_BRAND} bold]"
            f"  [{_DIM}]starting…[/{_DIM}]",
            end="\r",
        )
        from openvibe.api import OpenVibe
        self._ov = await OpenVibe(project_dir=self._dir).start_async()
        _con.print(
            f"[{_BRAND} bold]openvibe[/{_BRAND} bold]"
            f"  [{_DIM}]ready        [/{_DIM}]"
        )

        session_id = await self._welcome()
        if session_id is None:
            await self._ov.close_async()
            return

        self._session = self._ov.get_session(session_id)
        self._print_session_header()

        # Replay existing history so the user can read it
        await self._print_history()

        await self._chat_loop()

        _con.print(f"\n[{_DIM}]Goodbye.[/{_DIM}]")
        await self._ov.close_async()

    # ── welcome ────────────────────────────────────────────────────────────

    async def _welcome(self) -> str | None:
        from openvibe import __version__

        _con.print()
        _con.print(
            f"  [{_BRAND} bold]openvibe[/{_BRAND} bold]"
            f"  [{_DIM}]AI process agent  v{__version__}[/{_DIM}]"
        )
        _hr()

        recent = self._ov.list_sessions()[:8]
        if recent:
            _con.print(f"  [{_DIM}]Recent sessions:[/{_DIM}]")
            for i, s in enumerate(recent, 1):
                title   = s.title or f"session {s.id[:8]}"
                updated = (s.updated_at or "")[:10]
                _con.print(
                    f"    [{_XDIM}]{i}.[/{_XDIM}]"
                    f" {_e(title)}"
                    f"  [{_XDIM}]{updated}[/{_XDIM}]"
                )
            _con.print()

        _con.print(
            f"  [{_DIM}]n[/{_DIM}] new session"
            f"  [{_DIM}]a[/{_DIM}] new auto-accept"
            f"  [{_DIM}]q[/{_DIM}] quit"
        )
        _hr()

        while True:
            try:
                choice = (await self._readline("→ ")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None

            if choice in ("q", ":q"):
                return None
            if choice == "a":
                self._auto = True
                s = self._ov.create_session(auto_accept=True)
                return s.id
            if choice in ("", "n"):
                s = self._ov.create_session()
                return s.id
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(recent):
                    return recent[idx].id
            _con.print(f"  [{_DIM}]Type a number, n, a, or q.[/{_DIM}]")

    # ── session header ─────────────────────────────────────────────────────

    def _print_session_header(self) -> None:
        sid      = self._session.id[:12] if self._session else "?"
        auto_tag = f"  [{_WARN}]⚡ auto[/{_WARN}]" if self._auto else ""
        _con.print()
        _con.print(
            f"[{_BRAND} bold]openvibe[/{_BRAND} bold]"
            f"  [{_DIM}]{sid}[/{_DIM}]{auto_tag}"
        )
        _hr()
        _con.print(
            f"[{_DIM}]enter[/{_DIM}] send"
            f"  [{_DIM}]ctrl+j[/{_DIM}] newline"
            f"  [{_DIM}]↑↓[/{_DIM}] history"
            f"  [{_DIM}]ctrl+y[/{_DIM}] copy last"
            f"  [{_DIM}]:q[/{_DIM}] quit"
        )
        _hr()

    # ── history replay ─────────────────────────────────────────────────────

    async def _print_history(self) -> None:
        """Print existing session messages so resuming a session shows context."""
        from openvibe.config import MessageRole
        from openvibe.session.models import TextPart, ToolPart

        messages = self._session.messages()
        if not messages:
            return

        _con.print(f"[{_DIM}]── session history ──[/{_DIM}]")
        for msg in messages:
            role = str(msg.role)
            for part in msg.parts:
                if isinstance(part, TextPart) and part.content:
                    if role == "user":
                        _con.print(f"[{_USER}]>[/{_USER}] {_e(part.content)}")
                    elif role == "assistant":
                        try:
                            _con.print(Markdown(part.content))
                        except Exception:
                            _con.print(part.content)
                        self._last_assistant = part.content
                    elif role in ("error", "permission"):
                        _con.print(f"[{_WARN}]{_e(part.content)}[/{_WARN}]")
                elif isinstance(part, ToolPart) and part.state.output is not None:
                    _print_tool(part.state.model_dump())
        _hr()

    # ── chat loop ──────────────────────────────────────────────────────────

    async def _chat_loop(self) -> None:
        while True:
            try:
                text = await self._readline("> ")
            except (EOFError, KeyboardInterrupt):
                break

            text = text.strip()
            if not text:
                continue
            if text in (":q", ":q!", ":quit", "quit"):
                break

            await self._send(text)

    # ── send ───────────────────────────────────────────────────────────────

    async def _send(self, text: str) -> None:
        from openvibe.api import SessionState
        from openvibe.commands import _COMMANDS, is_command  # noqa: PLC2701

        # Slash commands handled separately
        if is_command(text):
            parts = text[1:].split(None, 1)
            name = parts[0].lower() if parts else ""
            if name in _COMMANDS:
                await self._run_command(text)
                return

        _con.print(f"\n[{_USER}]>[/{_USER}] {_e(text)}\n")

        tool_states: dict[int, dict] = {}
        text_buf: list[str] = []
        _in_tool = [False]

        def on_token(token: str) -> None:
            if _in_tool[0]:
                _con.print()   # newline after tool block before text continues
                _in_tool[0] = False
            _con.print(token, end="", markup=False)
            sys.stdout.flush()
            text_buf.append(token)

        def on_tool(msg_id: str, idx: int, state: dict) -> None:
            prev = tool_states.get(idx)
            tool_states[idx] = state
            if not _in_tool[0] and text_buf:
                _con.print()   # newline after streaming text before tool
            _in_tool[0] = True
            _print_tool(state, prev)

        def on_message(msg_id: str, role: str) -> None:
            pass

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._session.send(
                text,
                on_token=on_token,
                on_message=on_message,
                on_tool=on_tool,
            ),
        )

        if text_buf:
            self._last_assistant = "".join(text_buf)

        _con.print("\n")

        if response.state == SessionState.WAITING:
            await self._handle_permission(response)
        elif hasattr(response.state, "name") and response.state.name == "ERROR":
            err = response.error.message if response.error else "unknown error"
            _con.print(f"[{_ERR}]Error: {_e(err)}[/{_ERR}]")

        _hr()

    # ── permission ─────────────────────────────────────────────────────────

    async def _handle_permission(self, response: Any) -> None:
        from openvibe.api import SessionState

        req  = response.request
        desc = req.description or f"run {req.tool}"
        _con.print(f"[{_WARN}]⚠  {_e(req.tool or 'tool')}[/{_WARN}]: {_e(desc)}")
        if req.argument and req.argument != desc:
            _con.print(f"  [{_DIM}]{_e(req.argument)}[/{_DIM}]")

        if self._auto:
            _con.print(f"[{_OK}]→ auto-allowed[/{_OK}]")
            choice = "allow"
        else:
            _con.print(
                f"[{_DIM}]1[/{_DIM}] allow"
                f"  [{_DIM}]2[/{_DIM}] always"
                f"  [{_DIM}]3[/{_DIM}] deny"
                f"  [{_XDIM}](enter = allow)[/{_XDIM}]"
            )
            try:
                ans = (await self._readline("→ ")).strip()
            except (EOFError, KeyboardInterrupt):
                ans = "3"

            if ans == "2":
                choice = "allow_always"
                self._auto = True
                _con.print(f"[{_OK}]→ always allowed[/{_OK}]")
            elif ans == "3":
                choice = "deny"
                _con.print(f"[{_ERR}]→ denied[/{_ERR}]")
            else:
                choice = "allow"
                _con.print(f"[{_OK}]→ allowed[/{_OK}]")

        tool_states: dict[int, dict] = {}

        def on_token(token: str) -> None:
            _con.print(token, end="", markup=False)
            sys.stdout.flush()

        def on_tool(msg_id: str, idx: int, state: dict) -> None:
            prev = tool_states.get(idx)
            tool_states[idx] = state
            _print_tool(state, prev)

        loop = asyncio.get_event_loop()
        resumed = await loop.run_in_executor(
            None,
            lambda: self._session.reply(req.id, choice),
        )

        _con.print("\n")

        if resumed.state == SessionState.WAITING:
            await self._handle_permission(resumed)

    # ── slash commands ─────────────────────────────────────────────────────

    async def _run_command(self, text: str) -> None:
        _con.print(f"\n[{_USER}]>[/{_USER}] {_e(text)}\n")
        response = self._session.send(text)
        result   = response.command_result
        if result is None:
            _hr()
            return
        if result.quit:
            raise SystemExit(0)
        if result.clear:
            _con.clear()
            self._print_session_header()
            _hr()
            return
        if result.output:
            _con.print(result.output)
        if result.forward_to_session:
            await self._send(result.forward_to_session)
        _hr()

    # ── input ──────────────────────────────────────────────────────────────

    def _make_pt_session(self) -> Any:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("c-j")
        def _newline(event: Any) -> None:
            """Ctrl+J → insert newline (multi-line input)."""
            event.current_buffer.insert_text("\n")

        @kb.add("c-y")
        def _copy(event: Any) -> None:
            """Ctrl+Y → copy last assistant message via OSC 52."""
            if self._last_assistant:
                _osc52_copy(self._last_assistant)
                _con.print(f"\n[{_OK}]Copied to clipboard.[/{_OK}]", end="")

        return PromptSession(
            history=InMemoryHistory(),
            key_bindings=kb,
            enable_history_search=True,
            multiline=False,
        )

    async def _readline(self, prompt: str = "> ") -> str:
        if self._pt is None:
            try:
                self._pt = self._make_pt_session()
            except ImportError:
                # Fallback to plain input if prompt_toolkit unavailable
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: input(prompt))

        result = await self._pt.prompt_async(prompt)
        return result if result is not None else ""
