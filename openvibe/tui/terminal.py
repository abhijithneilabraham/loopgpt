"""Terminal-native chat UI: prompt_toolkit input + rich output.

Output flows inline — every message stays in the terminal's own scrollback
buffer. No alternate screen — all content is permanently in scrollback.
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
_BRAND = "#4a9fd4"   # openvibe blue
_DIM   = "#666666"   # muted labels
_XDIM  = "#444444"   # extra dim (tool output, dividers)
_WARN  = "#cc8800"   # amber — permission / warning
_ERR   = "#cc4444"   # red — errors
_OK    = "#3a7aaa"   # blue — completed / allowed
_SPIN  = "#cc9900"   # amber — streaming / running tool
_USER  = "#555555"   # user message prefix
_RULE  = "#2a2a2a"   # horizontal rule lines
_PERM  = "#cc8800"   # permission prompt text

_TOOL_ICON  = {"pending": "○", "running": "◎", "completed": "●", "error": "✗"}
_TOOL_COLOR = {"pending": _XDIM, "running": _SPIN, "completed": _OK, "error": _ERR}

_LOGO = (
    " ██╗   ██╗██╗██████╗ ███████╗\n"
    " ██║   ██║██║██╔══██╗██╔════╝\n"
    " ██║   ██║██║██████╔╝█████╗  \n"
    " ╚██╗ ██╔╝██║██╔══██╗██╔══╝  \n"
    "  ╚████╔╝ ██║██████╔╝███████╗\n"
    "   ╚═══╝  ╚═╝╚═════╝ ╚══════╝"
)


# ── internal signals ───────────────────────────────────────────────────────

class _GoHomeSignal(Exception):
    """User pressed Ctrl+H — go back to the home screen."""

class _NewSessionSignal(Exception):
    """User pressed Ctrl+N — start a brand-new session."""


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
            for line in str(out)[:2000].splitlines():
                _con.print(f"    [{_XDIM}]{_e(line)}[/{_XDIM}]")


def _copy_to_clipboard(text: str) -> bool:
    """OSC 52 clipboard copy, with subprocess fallbacks."""
    try:
        encoded = base64.b64encode(text.encode()).decode()
        seq = f"\033]52;c;{encoded}\a"
        try:
            with open("/dev/tty", "w") as tty:
                tty.write(seq)
                tty.flush()
        except OSError:
            sys.stdout.write(seq)
            sys.stdout.flush()
        return True
    except Exception:
        pass
    import subprocess
    for cmd in (["pbcopy"], ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"], ["wl-copy"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
            return True
        except Exception:
            continue
    return False


# ── welcome (interactive arrow-key home screen) ───────────────────────────

async def _run_welcome(ov: Any) -> str | None:
    """
    Full-featured interactive home screen.

    Returns a session ID to open, or None to quit.
    Uses a prompt_toolkit Application for arrow-key navigation so every
    item — recent sessions AND actions — is selectable with ↑↓ + Enter.
    """
    from openvibe import __version__
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    sessions = ov.list_sessions()[:8]

    # Menu items: ("session"|"action", id_or_key, label, subtitle)
    items: list[tuple[str, str, str, str]] = []
    for s in sessions:
        title   = s.title or f"session {s.id[:8]}"
        updated = (s.updated_at or "")[:10]
        items.append(("session", s.id, title, updated))

    items.append(("action", "new",  "New Session",          "start fresh"))
    items.append(("action", "auto", "Auto-Accept Session",  "⚡ allow all tools"))
    items.append(("action", "quit", "Quit",                 ""))

    selected = [0]
    result: list[str | None] = [None]

    # ── render ──────────────────────────────────────────────────────────
    def build_ft() -> FormattedText:
        import shutil
        term_width = shutil.get_terminal_size((100, 40)).columns

        ver       = f"v{__version__}"
        content_w = 64          # width of the content block
        line      = "─" * content_w
        # Left margin so the content block is horizontally centred
        margin    = max(0, (term_width - content_w) // 2)
        pad       = " " * margin

        # Logo lines are 32 chars wide; centre them inside the content block
        logo_margin = max(0, (content_w - 32) // 2)
        logo_pad    = " " * logo_margin

        ft: list[tuple[str, str]] = []

        def ln(style: str, text: str) -> None:
            ft.append(("", pad))
            ft.append((style, text))
            ft.append(("", "\n"))

        def blank(n: int = 1) -> None:
            for _ in range(n):
                ft.append(("", "\n"))

        def centred(style: str, text: str) -> None:
            inner_pad = max(0, (content_w - len(text)) // 2)
            ln(style, " " * inner_pad + text)

        # ── logo ────────────────────────────────────────────────────────
        blank(2)
        for logo_line in _LOGO.splitlines():
            ln(f"fg:{_BRAND} bold", logo_pad + logo_line)

        blank(2)
        centred(f"fg:{_BRAND}", "AI  Coding  Agent")
        blank()
        centred(f"fg:{_XDIM}", ver)
        blank(2)
        ln(f"fg:{_XDIM}", line)
        blank(2)

        # ── sessions ────────────────────────────────────────────────────
        if sessions:
            ln(f"fg:{_DIM}", "  Recent Sessions")
            blank()

        for i, (kind, key, label, sub) in enumerate(items):
            is_sel = i == selected[0]

            # Divider between sessions and actions
            if kind == "action" and (i == 0 or items[i - 1][0] == "session"):
                blank()
                ln(f"fg:{_XDIM}", line)
                blank()

            if kind == "session":
                cursor      = "▶" if is_sel else " "
                label_style = f"fg:{_BRAND} bold" if is_sel else "fg:#aaaaaa"
                sub_style   = f"fg:{_XDIM}"
                ft.append(("", pad))
                ft.append((f"fg:{_BRAND}" if is_sel else f"fg:{_XDIM}", f"  {cursor}  "))
                ft.append((label_style, f"{label:<42}"))
                ft.append((sub_style,   sub))
                ft.append(("", "\n"))
            else:
                if key == "quit":
                    base_col = _ERR
                elif key == "auto":
                    base_col = _WARN
                else:
                    base_col = _BRAND

                cursor      = "▶" if is_sel else " "
                label_style = f"fg:{base_col} bold" if is_sel else f"fg:{_DIM}"
                sub_style   = f"fg:{_XDIM}"
                ft.append(("", pad))
                ft.append((f"fg:{base_col}" if is_sel else f"fg:{_XDIM}", f"  {cursor}  "))
                ft.append((label_style, f"{label:<42}"))
                if sub:
                    ft.append((sub_style, sub))
                ft.append(("", "\n"))

        # ── footer ──────────────────────────────────────────────────────
        blank(2)
        ln(f"fg:{_XDIM}", line)
        blank()
        centred(f"fg:{_XDIM}", "↑↓ navigate    enter select    n new    a auto    q quit")
        blank(2)

        return FormattedText(ft)

    # ── key bindings ────────────────────────────────────────────────────
    kb = KeyBindings()

    @kb.add("up")
    def _up(event: Any) -> None:
        selected[0] = (selected[0] - 1) % len(items)

    @kb.add("down")
    def _down(event: Any) -> None:
        selected[0] = (selected[0] + 1) % len(items)

    @kb.add("enter")
    def _enter(event: Any) -> None:
        result[0] = items[selected[0]][1]
        event.app.exit()

    @kb.add("n")
    def _new(event: Any) -> None:
        result[0] = "new"
        event.app.exit()

    @kb.add("a")
    def _auto(event: Any) -> None:
        result[0] = "auto"
        event.app.exit()

    @kb.add("q")
    @kb.add("c-c")
    @kb.add("c-q")
    def _quit(event: Any) -> None:
        result[0] = "quit"
        event.app.exit()

    # digit shortcuts for recent sessions
    for _i in range(min(9, len(sessions))):
        def _make_digit_handler(idx: int):
            def _handler(event: Any) -> None:
                result[0] = items[idx][1]
                event.app.exit()
            return _handler
        kb.add(str(_i + 1))(_make_digit_handler(_i))

    style = Style.from_dict({"": f"bg:#111111 fg:{_DIM}"})
    layout = Layout(Window(
        content=FormattedTextControl(build_ft, focusable=True),
        dont_extend_height=True,
    ))
    app: Application = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
        refresh_interval=None,
    )
    await app.run_async()

    val = result[0]
    if val is None or val == "quit":
        return None
    if val == "new":
        return ov.create_session().id
    if val == "auto":
        s = ov.create_session(auto_accept=True)
        s._auto = True  # type: ignore[attr-defined]
        return s.id
    return val   # session ID


# ── main class ─────────────────────────────────────────────────────────────

class TerminalUI:
    """Interactive terminal chat — output flows inline, stays in scrollback."""

    def __init__(self, project_dir: Path | None = None) -> None:
        self._dir            = project_dir or Path.cwd()
        self._ov: Any        = None
        self._session: Any   = None
        self._auto           = False
        self._last_assistant = ""
        self._pt: Any        = None   # prompt_toolkit PromptSession (lazy)

    # ── entry point ────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            asyncio.run(self._main())
        except KeyboardInterrupt:
            _con.print(f"\n[{_DIM}]Interrupted.[/{_DIM}]")

    # ── startup ────────────────────────────────────────────────────────────

    async def _main(self) -> None:
        from openvibe.tui.app import _needs_setup

        _con.clear()

        if _needs_setup(self._dir):
            _con.print(
                f"[{_WARN}]No model configured.[/{_WARN}]  "
                f"[{_DIM}]Edit ~/.config/openvibe/openvibe.json to set your provider.[/{_DIM}]"
            )
            return

        _con.print(f"[{_BRAND} bold]openvibe[/{_BRAND} bold]  [{_DIM}]starting…[/{_DIM}]", end="\r")
        from openvibe.api import OpenVibe
        self._ov = await OpenVibe(project_dir=self._dir).start_async()
        _con.print(f"[{_BRAND} bold]openvibe[/{_BRAND} bold]  [{_DIM}]ready        [/{_DIM}]")

        while True:
            session_id = await _run_welcome(self._ov)
            if session_id is None:
                break

            self._session = self._ov.get_session(session_id)
            self._auto = getattr(self._session, "auto_accept", False) or getattr(self._session, "_auto", False)
            # Reset the per-session prompt_toolkit session so history starts fresh
            self._pt = None

            _con.clear()
            self._print_session_header()
            await self._print_history()

            result = await self._chat_loop()
            if result == "quit":
                break
            # "home" → loop back to welcome

        _con.print(f"\n[{_DIM}]Goodbye.[/{_DIM}]")
        await self._ov.close_async()

    # ── session header ─────────────────────────────────────────────────────

    def _print_session_header(self) -> None:
        info     = getattr(self._session, "info", None)
        title    = (getattr(info, "title", None) if info else None) or self._session.id[:12]
        auto_tag = f"  [{_WARN}]⚡ auto[/{_WARN}]" if self._auto else ""
        _con.print(f"[{_BRAND} bold]openvibe[/{_BRAND} bold]  [{_DIM}]{_e(title)}[/{_DIM}]{auto_tag}")
        _hr()
        _con.print(
            f"  [{_XDIM}]enter[/{_XDIM}] send"
            f"   [{_XDIM}]ctrl+j[/{_XDIM}] newline"
            f"   [{_XDIM}]↑↓[/{_XDIM}] history"
            f"   [{_XDIM}]ctrl+y[/{_XDIM}] copy"
            f"   [{_XDIM}]ctrl+h[/{_XDIM}] home"
            f"   [{_XDIM}]ctrl+n[/{_XDIM}] new"
            f"   [{_XDIM}]:q[/{_XDIM}] quit"
        )
        _hr()

    # ── history replay ─────────────────────────────────────────────────────

    async def _print_history(self) -> None:
        from openvibe.session.models import TextPart, ToolPart

        messages = self._session.messages()
        if not messages:
            return

        _con.print(f"  [{_XDIM}]── history ──[/{_XDIM}]")
        _con.print()
        for msg in messages:
            role = str(msg.role)
            for part in msg.parts:
                if isinstance(part, TextPart) and part.content:
                    if role == "user":
                        _con.print(f"[{_USER}]>[/{_USER}] [{_DIM}]{_e(part.content)}[/{_DIM}]")
                        _con.print()
                    elif role == "assistant":
                        try:
                            _con.print(Markdown(part.content))
                        except Exception:
                            _con.print(part.content)
                        self._last_assistant = part.content
                        _con.print()
                    elif role == "error":
                        _con.print(f"  [{_ERR}]⚠  {_e(part.content)}[/{_ERR}]")
                        _con.print()
                    elif role == "permission":
                        _con.print(f"  [{_PERM}]{_e(part.content)}[/{_PERM}]")
                        _con.print()
                elif isinstance(part, ToolPart) and part.state.output is not None:
                    _print_tool(part.state.model_dump())
        _hr()

    # ── chat loop ──────────────────────────────────────────────────────────

    async def _chat_loop(self) -> str:
        """Returns 'home' or 'quit'."""
        while True:
            try:
                text = await self._readline("> ")
            except EOFError:
                return "quit"
            except KeyboardInterrupt:
                return "quit"
            except _GoHomeSignal:
                _con.print(f"\n  [{_DIM}]← returning home…[/{_DIM}]\n")
                return "home"
            except _NewSessionSignal:
                _con.print(f"\n  [{_DIM}]↺ new session…[/{_DIM}]\n")
                # Create and switch to a new session immediately
                new_s = self._ov.create_session()
                self._session = new_s
                self._auto = False
                self._pt = None
                _con.clear()
                self._print_session_header()
                continue

            text = text.strip()
            if not text:
                continue
            if text in (":q", ":q!", ":quit"):
                return "quit"

            await self._send(text)

    # ── send ───────────────────────────────────────────────────────────────

    async def _send(self, text: str) -> None:
        from openvibe.api import SessionState
        from openvibe.commands import _COMMANDS, is_command  # noqa: PLC2701

        if is_command(text):
            parts = text[1:].split(None, 1)
            name = parts[0].lower() if parts else ""
            if name in _COMMANDS:
                await self._run_command(text)
                return

        _con.print()
        _con.print(f"[{_USER}]>[/{_USER}] {_e(text)}")
        _con.print()

        tool_states: dict[int, dict] = {}
        text_buf: list[str] = []
        _in_tool = [False]

        def on_token(token: str) -> None:
            if _in_tool[0]:
                _con.print()
                _in_tool[0] = False
            _con.print(token, end="", markup=False)
            sys.stdout.flush()
            text_buf.append(token)

        def on_tool(msg_id: str, idx: int, state: dict) -> None:
            prev = tool_states.get(idx)
            tool_states[idx] = state
            if not _in_tool[0] and text_buf:
                _con.print()
            _in_tool[0] = True
            _print_tool(state, prev)

        def on_message(msg_id: str, role: str) -> None:
            pass

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._session.send(
                    text,
                    on_token=on_token,
                    on_message=on_message,
                    on_tool=on_tool,
                ),
            )
        except KeyboardInterrupt:
            _con.print(f"\n  [{_WARN}]↯ cancelled[/{_WARN}]")
            _hr()
            return

        if text_buf:
            self._last_assistant = "".join(text_buf)

        _con.print("\n")

        if response.state == SessionState.WAITING:
            await self._handle_permission(response)
        elif hasattr(response.state, "name") and response.state.name == "ERROR":
            err = response.error.message if response.error else "unknown error"
            _con.print(f"  [{_ERR}]⚠  {_e(err)}[/{_ERR}]")

        _hr()

    # ── permission ─────────────────────────────────────────────────────────

    async def _handle_permission(self, response: Any) -> None:
        from openvibe.api import SessionState

        req  = response.request
        desc = req.description or f"run {req.tool}"

        _con.print(f"  [{_PERM}]⚠  [bold]{_e(req.tool or 'tool')}[/bold]: {_e(desc)}[/{_PERM}]")
        if req.argument and req.argument != desc:
            _con.print(f"    [{_DIM}]{_e(req.argument)}[/{_DIM}]")

        if self._auto:
            _con.print(f"    [{_OK}]→ auto-allowed[/{_OK}]")
            choice = "allow"
        else:
            _con.print(
                f"    [{_DIM}]1[/{_DIM}] allow"
                f"  [{_DIM}]2[/{_DIM}] always"
                f"  [{_DIM}]3[/{_DIM}] deny"
                f"  [{_XDIM}](enter = allow)[/{_XDIM}]"
            )
            try:
                ans = (await self._readline("  → ")).strip()
            except (EOFError, KeyboardInterrupt, _GoHomeSignal, _NewSessionSignal):
                ans = "3"

            if ans == "2":
                choice = "allow_always"
                self._auto = True
                _con.print(f"    [{_OK}]→ always allowed[/{_OK}]")
            elif ans == "3":
                choice = "deny"
                _con.print(f"    [{_ERR}]→ denied[/{_ERR}]")
            else:
                choice = "allow"
                _con.print(f"    [{_OK}]→ allowed[/{_OK}]")

        tool_states: dict[int, dict] = {}
        text_buf: list[str] = []
        _in_tool = [False]

        def on_token(token: str) -> None:
            if _in_tool[0]:
                _con.print()
                _in_tool[0] = False
            _con.print(token, end="", markup=False)
            sys.stdout.flush()
            text_buf.append(token)

        def on_tool(msg_id: str, idx: int, state: dict) -> None:
            prev = tool_states.get(idx)
            tool_states[idx] = state
            if not _in_tool[0] and text_buf:
                _con.print()
            _in_tool[0] = True
            _print_tool(state, prev)

        loop = asyncio.get_event_loop()
        resumed = await loop.run_in_executor(
            None,
            lambda: self._session.reply(req.id, choice),
        )

        if text_buf:
            self._last_assistant = "".join(text_buf)

        _con.print("\n")

        if resumed.state == SessionState.WAITING:
            await self._handle_permission(resumed)

    # ── slash commands ─────────────────────────────────────────────────────

    async def _run_command(self, text: str) -> None:
        _con.print()
        _con.print(f"[{_USER}]>[/{_USER}] {_e(text)}")
        _con.print()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self._session.send(text))
        result = response.command_result

        if result is None:
            _hr()
            return
        if result.quit:
            raise SystemExit(0)
        if result.clear:
            _con.clear()
            self._print_session_header()
            return
        if result.output:
            _con.print(result.output)
        if result.forward_to_session:
            await self._send(result.forward_to_session)
        _hr()

    # ── prompt_toolkit input session ───────────────────────────────────────

    def _make_pt_session(self) -> Any:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style

        kb = KeyBindings()

        @kb.add("c-j")
        def _newline(event: Any) -> None:
            """Ctrl+J → insert newline."""
            event.current_buffer.insert_text("\n")

        @kb.add("c-y")
        def _copy(event: Any) -> None:
            """Ctrl+Y → copy last assistant message."""
            if self._last_assistant:
                ok = _copy_to_clipboard(self._last_assistant)
                msg = "Copied to clipboard." if ok else "Clipboard unavailable."
                col = _OK if ok else _WARN
                _con.print(f"\n  [{col}]{msg}[/{col}]", end="")

        @kb.add("c-h")
        def _go_home(event: Any) -> None:
            """Ctrl+H → go back to home screen."""
            # Inject sentinel via buffer then submit
            event.current_buffer.text = "\x00__home__"
            event.current_buffer.validate_and_handle()

        @kb.add("c-n")
        def _new_session(event: Any) -> None:
            """Ctrl+N → start a new session."""
            event.current_buffer.text = "\x00__new__"
            event.current_buffer.validate_and_handle()

        style = Style.from_dict({
            "prompt": _USER,
            "":       "#e0e0e0",
        })

        return PromptSession(
            history=InMemoryHistory(),
            key_bindings=kb,
            enable_history_search=True,
            multiline=False,
            style=style,
        )

    async def _readline(self, prompt: str = "> ") -> str:
        if self._pt is None:
            try:
                self._pt = self._make_pt_session()
            except ImportError:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: input(prompt))

        result = await self._pt.prompt_async(prompt)
        if result is None:
            return ""
        if result == "\x00__home__":
            raise _GoHomeSignal()
        if result == "\x00__new__":
            raise _NewSessionSignal()
        return result
