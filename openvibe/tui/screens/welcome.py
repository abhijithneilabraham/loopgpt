"""Welcome screen — the first thing the user sees when they run `vibe`."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, ListItem, ListView, Static

from openvibe import __version__

_LOGO = r"""
 ██╗   ██╗██╗██████╗ ███████╗
 ██║   ██║██║██╔══██╗██╔════╝
 ██║   ██║██║██████╔╝█████╗
 ╚██╗ ██╔╝██║██╔══██╗██╔══╝
  ╚████╔╝ ██║██████╔╝███████╗
   ╚═══╝  ╚═╝╚═════╝ ╚══════╝"""

_TAGLINE = "AI coding agent for your terminal"

_HELP = """\
[#888888]Ctrl+N[/#888888] new session    \
[#888888]Ctrl+A[/#888888] auto-accept    \
[#888888]Ctrl+S[/#888888] sessions    \
[#888888]:q[/#888888] quit\
"""


class WelcomeScreen(Screen[None]):
    """Splash / home screen shown on first launch."""

    DEFAULT_CSS = """
    WelcomeScreen {
        align: center middle;
        background: #111111;
    }
    #panel {
        width: 66;
        height: auto;
        border: solid #2a2a2a;
        background: #1a1a1a;
        padding: 2 4;
    }
    #logo {
        color: #4a9f5a;
        text-align: center;
        margin-bottom: 0;
    }
    #tagline {
        text-align: center;
        color: #666666;
        margin-bottom: 0;
    }
    #version {
        text-align: center;
        color: #3a3a3a;
        margin-bottom: 1;
    }
    #divider {
        border-bottom: solid #2a2a2a;
        margin-bottom: 1;
        height: 1;
    }
    #help {
        text-align: center;
        color: #555555;
        margin-bottom: 1;
    }
    #recent-label {
        color: #555555;
        margin-bottom: 0;
    }
    #recent-list {
        height: auto;
        max-height: 6;
        border: solid #222222;
        background: #111111;
        margin-bottom: 1;
    }
    #no-recent {
        color: #3a3a3a;
        text-align: center;
        padding: 0 0 1 0;
    }
    #actions {
        height: auto;
        margin-top: 1;
        align-horizontal: center;
    }
    #actions Button {
        margin: 0 1;
        border: solid #333333;
    }
    Button.primary {
        background: #1e3a1e;
        color: #4a9f5a;
        border: solid #2d5a2d;
    }
    Button.primary:hover {
        background: #2d5a2d;
        color: #80c080;
    }
    Button.warning {
        background: #3a2a00;
        color: #cc8800;
        border: solid #553d00;
    }
    Button.warning:hover {
        background: #553d00;
        color: #ffb347;
    }
    Button.error {
        background: #3a1010;
        color: #cc4444;
        border: solid #5a2020;
    }
    Button.error:hover {
        background: #5a2020;
        color: #ff8080;
    }
    Button.default {
        background: #1e1e1e;
        color: #666666;
        border: solid #2a2a2a;
    }
    Button.default:hover {
        background: #2a2a2a;
        color: #888888;
    }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New session", show=False),
        Binding("ctrl+a", "auto_accept_session", "Auto-Accept session", show=False),
        Binding("ctrl+s", "all_sessions", "All sessions", show=False),
    ]

    def __init__(self) -> None:
        self._recent: list = []
        self._last_colon: bool = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="panel"):
                yield Static(_LOGO, id="logo")
                yield Static(_TAGLINE, id="tagline")
                yield Static(f"v{__version__}", id="version")
                yield Static("", id="divider")
                yield Static(_HELP, id="help")
                yield Static("Recent sessions", id="recent-label")
                yield ListView(id="recent-list")
                yield Static("No sessions yet — start one below.", id="no-recent")
                with Center(id="actions"):
                    yield Button("New Session", id="new", variant="primary")
                    yield Button("Auto-Accept", id="new-auto", variant="warning")
                    yield Button("All Sessions", id="all", variant="default")
                    yield Button("Quit", id="quit-btn", variant="error")

    def on_mount(self) -> None:
        ov = self.app.ov  # type: ignore[attr-defined]
        self._recent = ov.list_sessions()[:5]
        self._populate_recent()
        # Start focus on the New Session button so Tab cycles through buttons.
        self.query_one("#new", Button).focus()

    def _on_key(self, event: object) -> None:
        """Handle :q quit sequence on the home screen (no input box here)."""
        from textual import events
        if not isinstance(event, events.Key):
            return
        if self._last_colon and event.character == "q":
            self._last_colon = False
            self.app.exit()
            return
        self._last_colon = event.character == ":"

    def _populate_recent(self) -> None:
        lv = self.query_one("#recent-list", ListView)
        no_recent = self.query_one("#no-recent", Static)

        if not self._recent:
            lv.display = False
            no_recent.display = True
            return

        no_recent.display = False
        lv.display = True
        for s in self._recent:
            title = s.title or f"session {s.id[:8]}"
            updated = (s.updated_at or "")[:10]
            lv.append(
                ListItem(
                    Label(f"{title}  [dim]{updated}[/dim]"),
                    id=f"s-{s.id}",
                )
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id:
            session_id = event.item.id[2:]  # strip "s-"
            self._open_session(session_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        match event.button.id:
            case "new":
                self.action_new_session()
            case "new-auto":
                self.action_new_session(auto_accept=True)
            case "all":
                self.action_all_sessions()
            case "quit-btn":
                self.app.exit()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_auto_accept_session(self) -> None:
        self.action_new_session(auto_accept=True)

    def action_new_session(self, auto_accept: bool = False) -> None:
        ov = self.app.ov  # type: ignore[attr-defined]
        session = ov.create_session(auto_accept=auto_accept)
        self.app.cache_session(session)  # type: ignore[attr-defined]
        self._open_session(session.id)

    def action_all_sessions(self) -> None:
        from openvibe.tui.screens.session import SessionScreen
        from openvibe.tui.screens.sessions import SessionListScreen

        def on_dismiss(session_id: str | None) -> None:
            if session_id:
                self._open_session(session_id)

        self.app.push_screen(SessionListScreen(), on_dismiss)

    def _open_session(self, session_id: str) -> None:
        from openvibe.tui.screens.session import SessionScreen

        self.app.push_screen(SessionScreen(session_id))
