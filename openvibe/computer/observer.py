"""Background thread that captures frames at ~5 fps, diffs them, and logs screen/focus events."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass
class ScreenEvent:
    """One observed moment of screen or focus change."""

    timestamp: float
    event_type: str          # "change" | "focus_shift" | "action"
    description: str         # human-readable summary
    change_fraction: float = 0.0            # 0.0–1.0 fraction of pixels changed
    changed_region: list[int] | None = None # [x, y, w, h] bounding box
    focus_context: str = ""                 # from focus.get_focus_context()


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


class ScreenObserver:
    """Per-session background monitor: captures frames, diffs them, logs ScreenEvents."""

    def __init__(
        self,
        session_id: str,
        fps: float = 5.0,
        max_events: int = 60,
        change_threshold: float = 0.005,
        focus_poll_interval: float = 0.4,
    ) -> None:
        self.session_id = session_id
        self._fps = fps
        self._interval = 1.0 / max(fps, 0.5)
        self._max_events = max_events
        self._change_threshold = change_threshold
        self._focus_poll_interval = focus_poll_interval

        self._events: Deque[ScreenEvent] = deque(maxlen=max_events)
        self._last_frame: bytes | None = None
        self._last_focus: str = ""
        self._last_focus_ts: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread (no-op if already running)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name=f"screen-observer-{self.session_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and wait for it to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Public API — called from tools and agents
    # ------------------------------------------------------------------

    def get_recent_events(self, n: int = 10) -> list[ScreenEvent]:
        """Return up to *n* most recent events (oldest first)."""
        with self._lock:
            events = list(self._events)
        return events[-n:]

    def get_summary(self, n: int = 8) -> str:
        """Compact text of recent screen activity for LLM context."""
        events = self.get_recent_events(n)
        if not events:
            return "Screen observer: no activity recorded yet."

        lines = [f"Recent screen activity ({len(events)} events):"]
        for ev in reversed(events):
            age = time.time() - ev.timestamp
            age_str = f"{age:.1f}s ago" if age < 60 else f"{age/60:.0f}m ago"
            lines.append(f"  [{age_str}] {ev.description}")
            if ev.focus_context:
                lines.append(f"    → {ev.focus_context}")
        return "\n".join(lines)

    def record_action(self, action_desc: str) -> str:
        """Capture screen + focus after an action; return a feedback string for ToolResult."""
        try:
            from openvibe.computer.focus import get_focus_context
            from openvibe.computer.capture import capture_screen, diff_screenshots
        except ImportError:
            return ""

        # Focus after action
        try:
            focus = get_focus_context()
        except Exception:
            focus = "(focus unavailable)"
        self._last_focus = focus
        self._last_focus_ts = time.time()

        # Screen diff
        diff_line = ""
        try:
            new_frame, _, _ = capture_screen()
            if self._last_frame:
                diff = diff_screenshots(self._last_frame, new_frame, threshold=10)
                if diff["changed"]:
                    diff_line = diff["summary"]
                    ev = ScreenEvent(
                        timestamp=time.time(),
                        event_type="action",
                        description=f"{action_desc} → {diff['summary']}",
                        change_fraction=diff.get("change_fraction", 0.0),
                        changed_region=diff.get("changed_region"),
                        focus_context=focus,
                    )
                    with self._lock:
                        self._events.append(ev)
                else:
                    diff_line = "no visible screen change"
                    ev = ScreenEvent(
                        timestamp=time.time(),
                        event_type="action",
                        description=f"{action_desc} → no visible change (action may have missed target)",
                        focus_context=focus,
                    )
                    with self._lock:
                        self._events.append(ev)
            self._last_frame = new_frame
        except Exception:
            pass

        parts = []
        if focus:
            parts.append(f"Focus: {focus}")
        if diff_line:
            parts.append(f"Screen: {diff_line}")
        return "\n".join(parts)

    def current_focus(self) -> str:
        """Return cached focus string, refreshing if stale (>0.5s old)."""
        now = time.time()
        if now - self._last_focus_ts > 0.5:
            try:
                from openvibe.computer.focus import get_focus_context
                self._last_focus = get_focus_context()
                self._last_focus_ts = now
            except Exception:
                pass
        return self._last_focus

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        try:
            from openvibe.computer.capture import capture_screen, diff_screenshots
            from openvibe.computer.focus import get_focus_context
        except ImportError:
            return  # computer use deps not installed

        while self._running:
            try:
                now = time.time()
                new_frame, _, _ = capture_screen()

                if self._last_frame:
                    diff = diff_screenshots(self._last_frame, new_frame, threshold=12)
                    if diff["changed"] and diff.get("change_fraction", 0) >= self._change_threshold:
                        focus = self._poll_focus(now, get_focus_context)
                        # Detect focus shift
                        if focus != self._last_focus and self._last_focus:
                            ev = ScreenEvent(
                                timestamp=now,
                                event_type="focus_shift",
                                description=f"Focus changed: {focus}",
                                focus_context=focus,
                            )
                            with self._lock:
                                self._events.append(ev)
                        else:
                            ev = ScreenEvent(
                                timestamp=now,
                                event_type="change",
                                description=diff["summary"],
                                change_fraction=diff.get("change_fraction", 0.0),
                                changed_region=diff.get("changed_region"),
                                focus_context=focus,
                            )
                            with self._lock:
                                self._events.append(ev)
                        self._last_focus = focus

                self._last_frame = new_frame

            except Exception:
                pass  # never crash the observer thread

            time.sleep(self._interval)

    def _poll_focus(self, now: float, get_focus_context) -> str:
        """Return cached focus, refreshing every focus_poll_interval seconds."""
        if now - self._last_focus_ts >= self._focus_poll_interval:
            try:
                self._last_focus = get_focus_context()
            except Exception:
                pass
            self._last_focus_ts = now
        return self._last_focus


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

_observers: dict[str, ScreenObserver] = {}
_registry_lock = threading.Lock()


def get_observer(session_id: str) -> ScreenObserver:
    """Return (creating if needed) the observer for *session_id*."""
    with _registry_lock:
        if session_id not in _observers:
            _observers[session_id] = ScreenObserver(session_id=session_id)
        return _observers[session_id]


def remove_observer(session_id: str) -> None:
    """Stop and discard the observer for *session_id*."""
    with _registry_lock:
        obs = _observers.pop(session_id, None)
    if obs:
        obs.stop()
