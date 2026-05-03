"""Message list and message widgets."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as _escape
from rich.syntax import Syntax as _Syntax
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from openvibe.tui.markdown import render_markdown

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_diff(text: str) -> bool:
    """Return True if the text looks like a unified diff."""
    for line in text.splitlines()[:10]:
        if line.startswith(("diff --git ", "--- ", "+++ ", "@@ ")):
            return True
    return False


# ---------------------------------------------------------------------------
# Tool widget
# ---------------------------------------------------------------------------

_STATUS_ICON: dict[str, str] = {
    "pending": "○",
    "running": "◎",
    "completed": "●",
    "error": "✗",
}

_STATUS_STYLE: dict[str, str] = {
    "pending": "#444444",
    "running": "#cc9900",
    "completed": "#3a7a4a",
    "error": "#cc4444",
}


class ToolWidget(Widget):
    """Renders one tool call: status icon + name, expandable output on click."""

    DEFAULT_CSS = """
    ToolWidget {
        height: auto;
        padding: 0 0 0 2;
        margin: 0;
    }
    ToolWidget .purpose {
        height: auto;
        padding: 0 0 0 5;
        color: #444444;
    }
    ToolWidget .output {
        height: auto;
        padding: 0 2 1 5;
        color: #555555;
    }
    ToolWidget Static {
        height: auto;
    }
    """

    def __init__(self, state: dict[str, Any], **kwargs: Any) -> None:
        self._state = state
        self._expanded = state.get("status") in ("completed", "error")
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static(self._header_markup(), id="header")
        yield Static(self._purpose_markup(), classes="purpose", id="purpose")
        initial_output = self._render_output() if self._expanded else ""
        yield Static(initial_output, classes="output", id="output")

    def _header_markup(self) -> str:
        status = self._state.get("status", "pending")
        icon = _STATUS_ICON.get(status, "?")
        style = _STATUS_STYLE.get(status, "white")
        # Prefer action label; fall back to raw tool name + primary arg
        action = self._state.get("action", "")
        if action:
            display = action
        else:
            name = self._state.get("tool_name", "unknown")
            argument = self._extract_argument()
            display = f"{name} {argument}".strip() if argument else name
        return f"[{style}]{icon}[/{style}] [dim]{_escape(display)}[/dim]"

    def _purpose_markup(self) -> str:
        purpose = self._state.get("purpose", "")
        if not purpose:
            return ""
        return f"[dim]↳ {_escape(purpose)}[/dim]"

    def _extract_argument(self) -> str:
        """Return a short display string for the primary tool argument."""
        inp: dict = self._state.get("input") or {}
        if not inp:
            return ""
        # Prefer well-known single-argument keys in priority order.
        for key in ("command", "path", "url", "query", "code", "input"):
            if key in inp:
                val = str(inp[key])
                if len(val) > 200:
                    val = val[:200] + "…"
                return val
        # Fall back to the first value.
        val = str(next(iter(inp.values())))
        if len(val) > 200:
            val = val[:200] + "…"
        return val

    def update_state(self, state: dict[str, Any]) -> None:
        prev_status = self._state.get("status")
        self._state = state
        self.query_one("#header", Static).update(self._header_markup())
        self.query_one("#purpose", Static).update(self._purpose_markup())
        new_status = state.get("status")
        if new_status in ("completed", "error") and prev_status not in ("completed", "error"):
            self._expanded = True
        if self._expanded:
            self._refresh_output()

    def _render_output(self) -> str | _Syntax:
        content = self._state.get("output") or self._state.get("error") or ""
        if len(content) > 2000:
            content = content[:2000] + "\n… (truncated)"
        if _looks_like_diff(content):
            return _Syntax(content, "diff", theme="monokai", word_wrap=True)
        return f"[dim]{_escape(content)}[/dim]"

    def _refresh_output(self) -> None:
        out = self.query_one("#output", Static)
        if not self._expanded:
            out.update("")
            return
        out.update(self._render_output())


# ---------------------------------------------------------------------------
# Single message widget
# ---------------------------------------------------------------------------


class MessageWidget(Widget):
    """Renders one message: streaming text and optional tool parts."""

    DEFAULT_CSS = """
    MessageWidget {
        height: auto;
        padding: 1 4;
        margin: 0;
    }
    MessageWidget Static {
        height: auto;
    }
    MessageWidget Static.hidden-text {
        display: none;
    }
    MessageWidget Static.code-block {
        background: #0d1117;
        padding: 0 2;
        margin: 0;
    }
    MessageWidget.user {
        background: #1a1a1a;
        padding: 1 4;
        border-left: solid #3a7a4a;
    }
    MessageWidget.assistant {
        padding: 1 4;
    }
    MessageWidget.error {
        background: #1a0808;
        padding: 1 4;
        color: #cc4444;
        border-left: solid #cc4444;
    }
    MessageWidget.permission {
        background: #1a1500;
        padding: 1 4;
        color: #cc8800;
        border-left: solid #cc8800;
    }
    """

    def __init__(self, message_id: str, role: str, **kwargs: Any) -> None:
        self._message_id = message_id
        self._role = role
        self._text = ""
        self._tools: dict[int, ToolWidget] = {}
        # Tracks segment Static widgets for assistant markdown rendering.
        self._segments: list[tuple[str, Static]] = []
        self._seg_gen: int = 0
        super().__init__(classes=role, **kwargs)

    def compose(self) -> ComposeResult:
        # Start hidden — revealed on first text so tool-only assistant messages
        # don't render a blank line before the tool widgets.
        yield Static("", id=f"text-{self._message_id}", classes="hidden-text")

    def _simple_widget(self) -> Static:
        """Return the single Static used for non-assistant roles."""
        return self.query_one(f"#text-{self._message_id}", Static)

    def _update_segments(self) -> None:
        """Re-render markdown into segment widgets, reusing where possible."""
        segments = render_markdown(self._text)
        old_kinds = [k for k, _ in self._segments]
        new_kinds = [k for k, _ in segments]

        if old_kinds == new_kinds:
            # Structure unchanged — update content in place.
            for (_, widget), (_, content) in zip(self._segments, segments):
                widget.update(content)
            return

        # Structure changed — remove old segment widgets and mount new ones.
        # Hide the compose-time placeholder first.
        self._simple_widget().add_class("hidden-text")
        for _, widget in self._segments:
            widget.remove()
        self._segments.clear()
        self._seg_gen += 1

        # Find the insertion point: before the first ToolWidget if any,
        # otherwise append at the end.
        tool_widgets = self.query(ToolWidget)
        before = tool_widgets.first() if tool_widgets else None

        for i, (kind, content) in enumerate(segments):
            css_class = "code-block" if kind == "code" else ""
            seg_id = f"seg-{self._message_id}-{self._seg_gen}-{i}"
            widget = Static(content, id=seg_id, classes=css_class)
            self._segments.append((kind, widget))
            if before is not None:
                self.mount(widget, before=before)
            else:
                self.mount(widget)

    def append_text(self, content: str) -> None:
        if self._role == "assistant":
            self._text += content
            self._update_segments()
        else:
            widget = self._simple_widget()
            widget.remove_class("hidden-text")
            safe = _escape(content)
            if not self._text and self._role == "user":
                self._text = f"[dim]>[/dim] {safe}"
            elif not self._text and self._role == "error":
                self._text = f"⚠  {safe}"
            else:
                self._text += safe
            widget.update(self._text)

    def set_markup(self, content: str) -> None:
        """Display pre-rendered Rich markup directly, bypassing markdown processing."""
        widget = self._simple_widget()
        widget.remove_class("hidden-text")
        self._text = content
        widget.update(content)

    def replace_text(self, content: str) -> None:
        self._text = content
        if self._role == "assistant":
            self._update_segments()
        else:
            widget = self._simple_widget()
            widget.remove_class("hidden-text")
            widget.update(self._text)

    async def add_tool(self, index: int, state: dict[str, Any]) -> None:
        tool = ToolWidget(state, id=f"tool-{self._message_id}-{index}")
        self._tools[index] = tool
        await self.mount(tool)

    def update_tool(self, index: int, state: dict[str, Any]) -> None:
        if tool := self._tools.get(index):
            tool.update_state(state)


# ---------------------------------------------------------------------------
# Message list (scrollable container)
# ---------------------------------------------------------------------------


class MessageList(VerticalScroll):
    """Scrollable list of MessageWidgets; receives updates from SessionScreen."""

    DEFAULT_CSS = """
    MessageList {
        height: 1fr;
        padding: 1 0 0 0;
        background: #111111;
        scrollbar-color: #2a2a2a #111111;
        scrollbar-size: 1 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        self._messages: dict[str, MessageWidget] = {}
        super().__init__(**kwargs)

    def _at_bottom(self) -> bool:
        """Return True if the scroll position is at or near the bottom."""
        return self.scroll_y >= self.max_scroll_y - 3

    async def add_message(self, message_id: str, role: str) -> MessageWidget:
        if message_id in self._messages:
            return self._messages[message_id]
        widget = MessageWidget(message_id, role, id=f"msg-{message_id}")
        self._messages[message_id] = widget
        at_bottom = self._at_bottom()
        await self.mount(widget)
        if at_bottom:
            self.scroll_end(animate=False)
        return widget

    def get_message(self, message_id: str) -> MessageWidget | None:
        return self._messages.get(message_id)

    def append_text(self, message_id: str, content: str) -> None:
        if widget := self._messages.get(message_id):
            widget.append_text(content)
            if self._at_bottom():
                self.scroll_end(animate=False)

    def get_last_assistant_text(self) -> str:
        """Return the raw text of the most recent assistant message, or ''."""
        for widget in reversed(list(self._messages.values())):
            if widget._role == "assistant" and widget._text:
                return widget._text
        return ""
