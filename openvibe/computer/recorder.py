"""Demonstration recorder: captures interactions, saves semantic steps, replays via computer use.

Pipeline
--------
Record  — pynput captures raw events + AX element info on each click press.
Stop    — raw events compiled immediately to semantic steps; only steps saved to disk.
Replay  — TaskGraph.build_replay_prompt() produces a natural-language task description
          that is returned to the LLM as tool output. The LLM then uses the existing
          computer use tools (screenshot, mouse, keyboard) to execute it autonomously.
"""

from __future__ import annotations

import datetime
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def recordings_dir() -> Path:
    return Path.cwd() / "openvibe_recordings"


def name_from_prompt(prompt: str) -> str:
    words = re.findall(r"\w+", prompt.lower())[:5]
    return "_".join(words) or "recording"


# ---------------------------------------------------------------------------
# Semantic model — this is what gets saved to disk
# ---------------------------------------------------------------------------


@dataclass
class UIElementDescriptor:
    """Semantic description of a UI element at record time."""

    role: str           # e.g. "button", "textfield"
    label: str          # visible title / accessible name
    value: str          # current field value at record time
    bounds: list[int]   # [x, y, w, h] in screen coords
    click_x: int        # raw recorded click position
    click_y: int
    screenshot_b64: str = ""  # full screenshot at the moment of clicking
    # Cursor position just before the user moved to click this element.
    # Enables the replay agent to calculate relative movement distance.
    cursor_before_x: int = 0
    cursor_before_y: int = 0
    movement_dx: int = 0   # click_x - cursor_before_x
    movement_dy: int = 0   # click_y - cursor_before_y

    def to_dict(self) -> dict:
        return {
            "role": self.role, "label": self.label, "value": self.value,
            "bounds": self.bounds, "click_x": self.click_x, "click_y": self.click_y,
            "screenshot_b64": self.screenshot_b64,
            "cursor_before_x": self.cursor_before_x,
            "cursor_before_y": self.cursor_before_y,
            "movement_dx": self.movement_dx,
            "movement_dy": self.movement_dy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UIElementDescriptor":
        return cls(
            role=d.get("role", ""), label=d.get("label", ""),
            value=d.get("value", ""), bounds=d.get("bounds", []),
            click_x=d.get("click_x", 0), click_y=d.get("click_y", 0),
            screenshot_b64=d.get("screenshot_b64", ""),
            cursor_before_x=d.get("cursor_before_x", 0),
            cursor_before_y=d.get("cursor_before_y", 0),
            movement_dx=d.get("movement_dx", 0),
            movement_dy=d.get("movement_dy", 0),
        )


@dataclass
class SemanticStep:
    """One recorded interaction."""

    type: str                           # "click" | "type" | "key" | "hotkey" | "scroll"
    target: UIElementDescriptor | None  # set for clicks
    value: str                          # typed text or key name (for key steps)
    # Hotkey: ordered list of key names e.g. ["cmd", "shift", "z"]
    keys: list[str] = field(default_factory=list)
    # Scroll: position and total tick amounts (positive dy = up, negative = down)
    scroll_x: int = 0
    scroll_y: int = 0
    scroll_dx: int = 0   # horizontal: positive = right
    scroll_dy: int = 0   # vertical:   positive = up

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "target": self.target.to_dict() if self.target else None,
            "value": self.value,
            "keys": self.keys,
            "scroll_x": self.scroll_x,
            "scroll_y": self.scroll_y,
            "scroll_dx": self.scroll_dx,
            "scroll_dy": self.scroll_dy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticStep":
        t = d.get("target")
        return cls(
            type=d["type"],
            target=UIElementDescriptor.from_dict(t) if t else None,
            value=d.get("value", ""),
            keys=d.get("keys", []),
            scroll_x=d.get("scroll_x", 0),
            scroll_y=d.get("scroll_y", 0),
            scroll_dx=d.get("scroll_dx", 0),
            scroll_dy=d.get("scroll_dy", 0),
        )


@dataclass
class TaskGraph:
    """Compiled semantic recording. This is the on-disk format."""

    name: str
    prompt: str
    recorded_at: str
    duration: float
    steps: list[SemanticStep]

    # ------------------------------------------------------------------ persist

    def save(self, directory: Path | None = None) -> Path:
        d = directory or recordings_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.name}.json"
        path.write_text(json.dumps({
            "name": self.name,
            "prompt": self.prompt,
            "recorded_at": self.recorded_at,
            "duration": self.duration,
            "steps": [s.to_dict() for s in self.steps],
        }, indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> "TaskGraph | None":
        d = directory or recordings_dir()
        for path in [d / name, d / f"{name}.json"]:
            if path.exists():
                return cls._from_path(path)
        if d.exists():
            for p in d.glob("*.json"):
                if name.lower() in p.stem.lower():
                    return cls._from_path(p)
        return None

    @classmethod
    def _from_path(cls, path: Path) -> "TaskGraph":
        data = json.loads(path.read_text())
        return cls(
            name=data["name"],
            prompt=data["prompt"],
            recorded_at=data.get("recorded_at", ""),
            duration=data.get("duration", 0.0),
            steps=[SemanticStep.from_dict(s) for s in data.get("steps", [])],
        )

    @classmethod
    def list_all(cls, directory: Path | None = None) -> list["TaskGraph"]:
        d = directory or recordings_dir()
        if not d.exists():
            return []
        results = []
        for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text())
                results.append(cls(
                    name=data["name"], prompt=data["prompt"],
                    recorded_at=data.get("recorded_at", ""),
                    duration=data.get("duration", 0.0),
                    steps=[],
                ))
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------ replay

    def build_replay_prompt(
        self,
        current_cursor: tuple[int, int] | None = None,
    ) -> str:
        """Build a prompt directing the LLM to execute the recorded task.

        current_cursor: live (x, y) of the mouse cursor right now, injected at
        replay time so the agent can dynamically calculate movement distances.
        """
        cursor_note = (
            f"Current cursor position: ({current_cursor[0]}, {current_cursor[1]})\n"
            if current_cursor
            else ""
        )
        lines = [
            f"Task: {self.prompt}",
            "",
            "This is a DESKTOP AUTOMATION task — reproduce the recorded UI interactions on the live screen.",
            cursor_note.rstrip(),
            "",
            "MOVEMENT GUIDANCE:",
            "- The recorded steps include where the cursor was BEFORE each click (cursor_before) and",
            "  the movement delta (dx, dy) used to reach the target. Use these as a reference to",
            "  dynamically calculate how far to move from your CURRENT cursor position.",
            "- Formula: target_x ≈ current_cursor_x + (click_x - cursor_before_x)",
            "           target_y ≈ current_cursor_y + (click_y - cursor_before_y)",
            "- Always take a screenshot and confirm visually before acting — coordinates are hints.",
            "",
            "SCROLL GUIDANCE:",
            "- Recorded scroll steps show position (x, y), horizontal amount (dx), and vertical amount (dy).",
            "- dy > 0 = scroll UP (content moves up), dy < 0 = scroll DOWN.",
            "- dx > 0 = scroll RIGHT, dx < 0 = scroll LEFT.",
            "- Use mouse(action=scroll) at the same screen region, matching the direction and approximate amount.",
            "",
            "RULES:",
            "- Use ONLY: screenshot, mouse, keyboard, ui, app.",
            "- Take a screenshot before every action — verify each step visually.",
            "- If an action has no visible effect, try a different approach (different coordinates, scroll first, etc.).",
            "- Never stop and ask the user. Keep trying until every step is done or clearly impossible.",
            "",
            "Recorded steps:",
        ]

        for i, step in enumerate(self.steps, 1):
            if step.type == "click" and step.target:
                t = step.target
                name = t.label or t.value or t.role or "element"
                role_hint = f" [{t.role}]" if t.role else ""
                # Recorded target coordinates (center of bounds, or raw click)
                if t.bounds:
                    cx = t.bounds[0] + t.bounds[2] // 2
                    cy = t.bounds[1] + t.bounds[3] // 2
                else:
                    cx, cy = t.click_x, t.click_y
                move_hint = ""
                if t.cursor_before_x or t.cursor_before_y:
                    move_hint = (
                        f" [cursor was at ({t.cursor_before_x},{t.cursor_before_y})"
                        f" → moved ({t.movement_dx:+d},{t.movement_dy:+d}) to reach target]"
                    )
                lines.append(
                    f"  {i}. Click {name!r}{role_hint} at recorded ~({cx},{cy}){move_hint}"
                )

            elif step.type == "type":
                lines.append(
                    f"  {i}. Type {step.value!r}"
                )

            elif step.type == "key":
                key_name = step.value.replace("Key.", "")
                lines.append(f"  {i}. Press key: {key_name}")

            elif step.type == "hotkey":
                combo = "+".join(step.keys)
                lines.append(f"  {i}. Keyboard shortcut: {combo}")

            elif step.type == "scroll":
                v_dir = "up" if step.scroll_dy > 0 else "down" if step.scroll_dy < 0 else ""
                h_dir = "right" if step.scroll_dx > 0 else "left" if step.scroll_dx < 0 else ""
                parts = []
                if v_dir:
                    parts.append(f"{abs(step.scroll_dy)} ticks {v_dir}")
                if h_dir:
                    parts.append(f"{abs(step.scroll_dx)} ticks {h_dir}")
                direction_str = " + ".join(parts) or "0 ticks"
                lines.append(
                    f"  {i}. Scroll {direction_str} at recorded position"
                    f" ({step.scroll_x},{step.scroll_y})"
                )

        lines += [
            "",
            "Start: take a screenshot, note current cursor position, then execute each step.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raw capture — in-memory only, never written to disk
# ---------------------------------------------------------------------------


@dataclass
class _RawEvent:
    t: float
    type: str
    data: dict[str, Any]
    ax_info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AX capture — enriches click events during recording
# ---------------------------------------------------------------------------


def _capture_screen_b64_safe() -> str:
    """Capture a full-screen screenshot as base64. Returns '' on failure."""
    try:
        from openvibe.computer.capture import capture_screen_b64
        b64, _, _ = capture_screen_b64()
        return b64
    except Exception:
        return ""


def _capture_ax_at_point(x: int, y: int) -> dict:
    """Return AX element info at (x, y). Empty dict on failure."""
    try:
        from ApplicationServices import (  # type: ignore[import]
            AXUIElementCreateSystemWide,
            AXUIElementCopyElementAtPosition,
            AXUIElementCopyAttributeValue,
        )
        system = AXUIElementCreateSystemWide()
        err, element = AXUIElementCopyElementAtPosition(system, float(x), float(y), None)
        if err != 0 or element is None:
            return {}

        info: dict = {}
        for attr, key in [
            ("AXRole", "role"), ("AXTitle", "label"),
            ("AXValue", "value"), ("AXDescription", "description"),
        ]:
            try:
                err2, val = AXUIElementCopyAttributeValue(element, attr, None)
                if err2 == 0 and val:
                    s = str(val)
                    if key == "role":
                        s = s.replace("AX", "").lower()
                    info[key] = s
            except Exception:
                pass

        try:
            err3, pos = AXUIElementCopyAttributeValue(element, "AXPosition", None)
            err4, sz = AXUIElementCopyAttributeValue(element, "AXSize", None)
            if err3 == 0 and err4 == 0 and pos and sz:
                info["bounds"] = [int(pos.x), int(pos.y), int(sz.width), int(sz.height)]
        except Exception:
            pass

        return info
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Compile helpers
# ---------------------------------------------------------------------------


# Modifier keys that form keyboard shortcuts when combined with another key.
_NON_SHIFT_MODS = frozenset({
    "Key.cmd", "Key.cmd_l", "Key.cmd_r",
    "Key.ctrl", "Key.ctrl_l", "Key.ctrl_r",
    "Key.alt", "Key.alt_l", "Key.alt_r",
})
_ALL_MODS = _NON_SHIFT_MODS | frozenset({
    "Key.shift", "Key.shift_l", "Key.shift_r",
})


def _normalize_mod(key: str) -> str:
    k = key.lower()
    if "cmd" in k or "command" in k: return "cmd"
    if "ctrl" in k or "control" in k: return "ctrl"
    if "alt" in k or "option" in k: return "alt"
    if "shift" in k: return "shift"
    return key.replace("Key.", "").lower()


def _normalize_action_key(key: str) -> str:
    if key.startswith("Key."):
        return key[4:].lower()
    return key.lower()


def _extract_shortcuts(events: list[_RawEvent]) -> tuple[dict[int, list[str]], set[int]]:
    """Detect modifier + key combinations (Cmd+C, Ctrl+Z, Cmd+Shift+Z, etc.).

    Returns (shortcuts, skip_indices):
      shortcuts    — maps the index of the triggering key_press → ordered combo list
      skip_indices — all event indices that belong to a shortcut (modifier presses,
                     releases, and the action key) so text-run detection skips them.

    Rule: a shortcut is fired whenever a non-modifier key is pressed while at least
    one non-shift modifier (cmd, ctrl, alt) is currently held.  Shift-only combos
    (e.g. Shift+A) are plain text and are NOT treated as shortcuts.
    """
    shortcuts: dict[int, list[str]] = {}
    skip_indices: set[int] = set()

    # held_mods: modifier_key_str → event_index where it was pressed
    held_mods: dict[str, int] = {}

    for i, ev in enumerate(events):
        key = ev.data.get("key", "")

        if ev.type == "key_press":
            if key in _ALL_MODS:
                held_mods[key] = i
            elif held_mods:
                # Check if any non-shift modifier is held
                non_shift = {k: v for k, v in held_mods.items() if k in _NON_SHIFT_MODS}
                if non_shift:
                    # Build ordered combo: modifiers in canonical order + action key
                    order = ["ctrl", "alt", "cmd", "shift"]
                    present = sorted(
                        {_normalize_mod(m) for m in held_mods},
                        key=lambda m: order.index(m) if m in order else 99,
                    )
                    combo = present + [_normalize_action_key(key)]
                    # Anchor to the earliest modifier event
                    start_idx = min(held_mods.values())
                    shortcuts[start_idx] = combo
                    skip_indices.add(i)
                    for mod_idx in held_mods.values():
                        skip_indices.add(mod_idx)

        elif ev.type == "key_release":
            held_mods.pop(key, None)
            # Mark releases of shortcut modifiers as skip too (handled lazily below)

    # Also skip all key_release events that correspond to skipped key_press events
    # by scanning and matching — simplest is to mark releases of modifier keys when
    # the press was already in skip_indices.
    for i, ev in enumerate(events):
        if ev.type == "key_release":
            key = ev.data.get("key", "")
            if key in _ALL_MODS:
                # Find the most recent skipped press for this modifier
                for j in range(i - 1, -1, -1):
                    ej = events[j]
                    if ej.type == "key_press" and ej.data.get("key") == key and j in skip_indices:
                        skip_indices.add(i)
                        break

    return shortcuts, skip_indices


def _extract_text_runs(
    events: list[_RawEvent],
    exclude_indices: set[int] | None = None,
) -> tuple[dict[int, str], set[int]]:
    """Group consecutive printable key events into text runs (backspaces resolved).

    exclude_indices: indices already claimed by shortcuts — skip them entirely.
    Returns (text_runs, skip_indices): text_runs maps start-index → resolved string;
    skip_indices holds every non-start index that belongs to a run.
    """
    text_runs: dict[int, str] = {}
    skip_indices: set[int] = set()
    exclude = exclude_indices or set()

    def _is_printable(k: str) -> bool:
        return len(k) == 1 and k.isprintable()

    i = 0
    while i < len(events):
        if i in exclude:
            i += 1
            continue
        ev = events[i]
        if ev.type != "key_press":
            i += 1
            continue
        key = ev.data.get("key", "")
        if not _is_printable(key):
            i += 1
            continue

        text = key
        start = i
        j = i + 1

        while j < len(events):
            if j in exclude:
                j += 1; continue
            nev = events[j]
            nkey = nev.data.get("key", "")
            if nev.type == "move":
                skip_indices.add(j); j += 1; continue
            if nev.type == "key_press":
                if _is_printable(nkey):
                    text += nkey; skip_indices.add(j); j += 1; continue
                if nkey == "Key.backspace":
                    text = text[:-1]; skip_indices.add(j); j += 1; continue
                break
            if nev.type == "key_release":
                if _is_printable(nkey) or nkey == "Key.backspace":
                    skip_indices.add(j); j += 1; continue
                break
            break

        text_runs[start] = text
        i = j

    return text_runs, skip_indices


def _trim_trailing_input(events: list[_RawEvent], duration: float) -> list[_RawEvent]:
    """Strip only the final burst of keyboard/move events that are the /learn stop command.

    Only strips events in the last 3 seconds of recording that are exclusively
    key/move/mouse-release — if there's a click press in that window we stop
    trimming immediately so real task clicks are never removed.
    """
    if not events:
        return events
    cutoff = duration - 3.0  # only examine the last 3 seconds
    i = len(events) - 1
    while i >= 0:
        ev = events[i]
        if ev.t < cutoff:
            break  # too far back — stop trimming
        if ev.type in ("key_press", "key_release", "move"):
            i -= 1
        elif ev.type == "click" and not ev.data.get("pressed"):
            i -= 1
        else:
            break  # hit a real action (click press) — stop
    return events[: i + 1]


def _coalesce_scroll_steps(steps: list[SemanticStep]) -> list[SemanticStep]:
    """Merge consecutive scroll steps at approximately the same position."""
    result: list[SemanticStep] = []
    i = 0
    while i < len(steps):
        s = steps[i]
        if s.type != "scroll":
            result.append(s)
            i += 1
            continue
        # Accumulate consecutive scrolls within 100px of the same origin
        tdx, tdy = s.scroll_dx, s.scroll_dy
        j = i + 1
        while j < len(steps):
            ns = steps[j]
            if ns.type != "scroll":
                break
            if abs(ns.scroll_x - s.scroll_x) > 100 or abs(ns.scroll_y - s.scroll_y) > 100:
                break
            tdx += ns.scroll_dx
            tdy += ns.scroll_dy
            j += 1
        result.append(SemanticStep(
            type="scroll", target=None, value="",
            scroll_x=s.scroll_x, scroll_y=s.scroll_y,
            scroll_dx=tdx, scroll_dy=tdy,
        ))
        i = j
    return result


def _compile(events: list[_RawEvent], name: str, prompt: str,
             recorded_at: str, duration: float) -> TaskGraph:
    """Convert raw in-memory events to a TaskGraph.

    Produces SemanticSteps for:
      click   — mouse press, enriched with AX element info + cursor-before position
      type    — resolved text run (backspaces applied)
      key     — special key press (enter, escape, tab, arrow keys, etc.)
      hotkey  — modifier + key combination (Cmd+C, Ctrl+Z, Cmd+Shift+Z, …)
      scroll  — vertical/horizontal scroll (coalesced into single steps per burst)
    """
    # Phase 1: extract shortcuts first, then text runs excluding shortcut indices
    shortcuts, shortcut_skip = _extract_shortcuts(events)
    text_runs, text_skip = _extract_text_runs(events, exclude_indices=shortcut_skip)

    # Combined skip set
    all_skip = shortcut_skip | text_skip

    steps: list[SemanticStep] = []

    # Track last known cursor position for movement delta calculation
    last_cursor: tuple[int, int] = (0, 0)

    for i, ev in enumerate(events):
        # ── Shortcut anchor: check BEFORE all_skip so the anchor modifier
        #    event (which is in skip) still emits a hotkey step. ───────────
        if i in shortcuts:
            steps.append(SemanticStep(
                type="hotkey", target=None, value="",
                keys=shortcuts[i],
            ))
            continue

        if i in all_skip:
            # Update cursor tracking from move events even if skipped
            if ev.type == "move":
                last_cursor = (ev.data["x"], ev.data["y"])
            continue

        # ── Mouse move: update cursor tracking only ──────────────────────
        if ev.type == "move":
            last_cursor = (ev.data["x"], ev.data["y"])
            continue

        # ── Click press ──────────────────────────────────────────────────
        if ev.type == "click" and ev.data.get("pressed"):
            ax = ev.ax_info or {}
            cx, cy = ev.data["x"], ev.data["y"]
            descriptor = UIElementDescriptor(
                role=ax.get("role", ""),
                label=ax.get("label", ax.get("value", "")),
                value=ax.get("value", ""),
                bounds=ax.get("bounds", [cx, cy, 4, 4]),
                click_x=cx,
                click_y=cy,
                screenshot_b64=ax.get("screenshot_b64", ""),
                cursor_before_x=last_cursor[0],
                cursor_before_y=last_cursor[1],
                movement_dx=cx - last_cursor[0],
                movement_dy=cy - last_cursor[1],
            )
            steps.append(SemanticStep(type="click", target=descriptor, value=""))
            last_cursor = (cx, cy)

        # ── Click release: update cursor ─────────────────────────────────
        elif ev.type == "click" and not ev.data.get("pressed"):
            last_cursor = (ev.data["x"], ev.data["y"])

        # ── Scroll ───────────────────────────────────────────────────────
        elif ev.type == "scroll":
            steps.append(SemanticStep(
                type="scroll", target=None, value="",
                scroll_x=ev.data["x"], scroll_y=ev.data["y"],
                scroll_dx=ev.data.get("dx", 0), scroll_dy=ev.data.get("dy", 0),
            ))

        # ── Text run start ───────────────────────────────────────────────
        elif i in text_runs:
            text = text_runs[i]
            if text:  # skip empty runs (all backspaced away)
                steps.append(SemanticStep(type="type", target=None, value=text))

        # ── Special key (Key.enter, Key.escape, arrow keys, etc.) ────────
        elif ev.type == "key_press":
            key_s = ev.data.get("key", "")
            if key_s.startswith("Key.") and key_s not in _ALL_MODS:
                steps.append(SemanticStep(type="key", target=None, value=key_s))

    # Coalesce consecutive scroll bursts into single steps
    steps = _coalesce_scroll_steps(steps)

    return TaskGraph(
        name=name, prompt=prompt,
        recorded_at=recorded_at, duration=duration,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Key serialization
# ---------------------------------------------------------------------------


def _serialize_key(key: Any) -> str | None:
    try:
        from pynput.keyboard import Key, KeyCode
        if isinstance(key, Key):
            return f"Key.{key.name}"
        if isinstance(key, KeyCode):
            if key.char is not None:
                return key.char
            if key.vk is not None:
                return f"KeyCode.{key.vk}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Recorder — pynput event capture
# ---------------------------------------------------------------------------


class Recorder:
    """Captures mouse and keyboard events; compiles to TaskGraph on stop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[_RawEvent] = []
        self._start_time: float = 0.0
        self._prompt: str = ""
        self._name: str = ""
        self._recording: bool = False
        self._mouse_listener = None
        self._keyboard_listener = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, prompt: str, name: str) -> None:
        if self._recording:
            raise RuntimeError("Already recording. Call stop() first.")
        try:
            from pynput import mouse as pm, keyboard as pk
        except ImportError as exc:
            raise ImportError("pynput is required: pip install pynput") from exc

        with self._lock:
            self._events = []
            self._prompt = prompt
            self._name = name

        self._start_time = time.time()
        self._recording = True

        self._mouse_listener = pm.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

        self._keyboard_listener = pk.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._keyboard_listener.daemon = True
        self._keyboard_listener.start()

    def stop(self) -> TaskGraph:
        """Stop recording and return the compiled TaskGraph."""
        if not self._recording:
            raise RuntimeError("Not recording. Call start() first.")
        self._recording = False
        duration = time.time() - self._start_time
        recorded_at = datetime.datetime.now().isoformat()

        for listener in (self._mouse_listener, self._keyboard_listener):
            if listener:
                try:
                    listener.stop()
                except Exception:
                    pass

        with self._lock:
            events = _trim_trailing_input(list(self._events), duration)

        return _compile(events, self._name, self._prompt, recorded_at, duration)

    def _elapsed(self) -> float:
        return time.time() - self._start_time

    def _record(self, type_: str, data: dict, ax_info: dict | None = None) -> None:
        if not self._recording:
            return
        with self._lock:
            self._events.append(_RawEvent(
                t=self._elapsed(), type=type_, data=data, ax_info=ax_info or {}
            ))

    def _on_move(self, x: int, y: int) -> None:
        self._record("move", {"x": x, "y": y})

    def _on_click(self, x: int, y: int, button: Any, pressed: bool) -> None:
        ax_info: dict = {}
        if pressed:
            ax_info = _capture_ax_at_point(x, y)
            ax_info["screenshot_b64"] = _capture_screen_b64_safe()
        self._record("click",
                     {"x": x, "y": y, "button": button.name, "pressed": pressed},
                     ax_info)

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._record("scroll", {"x": x, "y": y, "dx": dx, "dy": dy})

    def _on_key_press(self, key: Any) -> None:
        s = _serialize_key(key)
        if s is not None:
            self._record("key_press", {"key": s})

    def _on_key_release(self, key: Any) -> None:
        s = _serialize_key(key)
        if s is not None:
            self._record("key_release", {"key": s})


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_recorder = Recorder()


def get_recorder() -> Recorder:
    return _recorder
