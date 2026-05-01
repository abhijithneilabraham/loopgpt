"""Computer Use — screen capture, mouse, keyboard, app control, and live observation.

Public API
----------
- :class:`ComputerSandbox` — session-scoped sandbox with audit log
- :func:`get_sandbox` — retrieve (or create) the sandbox for a session
- :func:`clear_sandbox` — discard a session's sandbox
- :class:`ScreenObserver` — background live screen + focus monitor
- :func:`get_observer` — retrieve (or create) the observer for a session
- :func:`get_focus_context` — one-shot current focus query
"""

from openvibe.computer.sandbox import (
    ActionType,
    AuditEntry,
    ComputerSandbox,
    clear_sandbox,
    get_sandbox,
)
from openvibe.computer.observer import ScreenObserver, get_observer, remove_observer
from openvibe.computer.focus import get_focus_context

__all__ = [
    "ActionType",
    "AuditEntry",
    "ComputerSandbox",
    "get_sandbox",
    "clear_sandbox",
    "ScreenObserver",
    "get_observer",
    "remove_observer",
    "get_focus_context",
]
