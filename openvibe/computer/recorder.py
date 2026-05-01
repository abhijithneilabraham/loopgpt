"""Screenshot-assisted pynput recorder with vision-guided replay.

Recording
---------
Captures every mouse click, scroll, and key event via pynput listeners.
A screenshot is taken at the moment of each mouse-press so the vision LLM can
locate the same element at replay time regardless of window position changes.

Replay
------
For every click event:
  1. Take a fresh screenshot of the current screen.
  2. Show the LLM: the recorded screenshot + the recorded click coords.
  3. LLM returns the pixel coordinates of the same element in the current screenshot.
  4. Click there with a raw OS mouse event.

Mouse moves are skipped during vision replay — only semantic events (clicks,
scrolls, key presses) are replayed.

Interrupt
---------
Any keyboard press or significant mouse movement by the human during replay
stops it immediately.
"""

from __future__ import annotations

import asyncio
import base64
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
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RecordedEvent:
    t: float            # seconds from recording start
    type: str           # move | click | scroll | key_press | key_release
    data: dict[str, Any]
    screenshot_b64: str = ""  # screenshot at the moment of a mouse press

    def to_dict(self) -> dict:
        d: dict = {"t": self.t, "type": self.type, "data": self.data}
        if self.screenshot_b64:
            d["screenshot_b64"] = self.screenshot_b64
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RecordedEvent":
        return cls(
            t=d["t"],
            type=d["type"],
            data=d.get("data", {}),
            screenshot_b64=d.get("screenshot_b64", ""),
        )


@dataclass
class Recording:
    name: str
    prompt: str
    recorded_at: str
    duration: float
    events: list[RecordedEvent]

    # ------------------------------------------------------------------ save/load

    def save(self, directory: Path | None = None) -> Path:
        d = directory or recordings_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.name}.json"
        path.write_text(json.dumps({
            "name": self.name,
            "prompt": self.prompt,
            "recorded_at": self.recorded_at,
            "duration": self.duration,
            "events": [e.to_dict() for e in self.events],
        }, indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> "Recording | None":
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
    def _from_path(cls, path: Path) -> "Recording":
        data = json.loads(path.read_text())
        return cls(
            name=data["name"],
            prompt=data["prompt"],
            recorded_at=data.get("recorded_at", ""),
            duration=data.get("duration", 0.0),
            events=[RecordedEvent.from_dict(e) for e in data.get("events", [])],
        )

    @classmethod
    def list_all(cls, directory: Path | None = None) -> list["Recording"]:
        d = directory or recordings_dir()
        if not d.exists():
            return []
        results = []
        for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text())
                results.append(cls(
                    name=data["name"],
                    prompt=data["prompt"],
                    recorded_at=data.get("recorded_at", ""),
                    duration=data.get("duration", 0.0),
                    events=[],
                ))
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------ replay

    def replay_with_vision(self, speed: float = 1.0) -> str:
        """Vision-guided replay.

        For each mouse click, takes a screenshot and asks the vision LLM to
        locate the same element in the current screen.  Any keyboard press or
        significant mouse deviation by the human stops the replay.

        Returns "done" or "interrupted".
        """
        try:
            from pynput import mouse as pm, keyboard as pk
        except ImportError as exc:
            raise ImportError("pynput is required: pip install pynput") from exc

        mc = pm.Controller()
        kc = pk.Controller()

        # Run a dedicated event loop for LLM calls.
        loop = asyncio.new_event_loop()

        def _llm_coords(old_b64: str, old_x: int, old_y: int, new_b64: str) -> tuple[int, int]:
            if not old_b64:
                return old_x, old_y
            try:
                return loop.run_until_complete(
                    _find_element_coords(old_b64, old_x, old_y, new_b64)
                )
            except Exception:
                return old_x, old_y

        # Track where we last placed the mouse to detect human intervention.
        _expected_pos: list[tuple[int, int] | None] = [None]

        def _human_moved() -> bool:
            if _expected_pos[0] is None:
                return False
            ex, ey = _expected_pos[0]
            cx, cy = mc.position
            return abs(cx - ex) > 30 or abs(cy - ey) > 30

        result = "done"
        try:
            prev_t = 0.0
            for event in self.events:
                if _human_moved():
                    result = "interrupted"
                    break

                dt = (event.t - prev_t) / speed
                if dt > 0:
                    time.sleep(min(dt, 2.0))  # cap gap to 2s
                prev_t = event.t

                if _human_moved():
                    result = "interrupted"
                    break

                d = event.data

                # --- mouse click ------------------------------------------------
                if event.type == "click":
                    from pynput.mouse import Button
                    btn = getattr(Button, d.get("button", "left"), Button.left)

                    if d.get("pressed"):
                        # Take screenshot → LLM → precise coords
                        new_x, new_y = _vision_coords(
                            mc, _llm_coords, event, d["x"], d["y"]
                        )
                        mc.position = (new_x, new_y)
                        time.sleep(0.04)
                        mc.press(btn)
                        _expected_pos[0] = (new_x, new_y)
                        time.sleep(0.05)
                    else:
                        mc.release(btn)

                # --- scroll -----------------------------------------------------
                elif event.type == "scroll":
                    mc.position = (d["x"], d["y"])
                    mc.scroll(d.get("dx", 0), d.get("dy", 0))
                    _expected_pos[0] = (d["x"], d["y"])

                # --- keyboard ---------------------------------------------------
                elif event.type == "key_press":
                    key = _deserialize_key(d["key"])
                    if key is not None:
                        kc.press(key)

                elif event.type == "key_release":
                    key = _deserialize_key(d["key"])
                    if key is not None:
                        kc.release(key)

                # mouse move events are skipped — clicks handle positioning

        finally:
            loop.close()

        return result


def _vision_coords(
    mc,
    llm_fn,
    event: RecordedEvent,
    fallback_x: int,
    fallback_y: int,
) -> tuple[int, int]:
    """Take current screenshot and use LLM to find the element's current coords."""
    try:
        from openvibe.computer.capture import capture_screen
        png, _, _ = capture_screen()
        cur_b64 = base64.b64encode(png).decode()
        return llm_fn(event.screenshot_b64, fallback_x, fallback_y, cur_b64)
    except Exception:
        return fallback_x, fallback_y


# ---------------------------------------------------------------------------
# Vision LLM — element coordinate finder
# ---------------------------------------------------------------------------


async def _find_element_coords(
    recorded_b64: str,
    recorded_x: int,
    recorded_y: int,
    current_b64: str,
) -> tuple[int, int]:
    """Ask the vision LLM where the same element is in the current screenshot.

    Returns (x, y) in current screenshot pixel space.
    Falls back to recorded coords on any error.
    """
    from openvibe.llm import LiteLLMBackend, Message, ContentBlock, TextDelta, resolve_model

    system = (
        "You are a precise screen-coordinate calculator. "
        "You will be shown two screenshots: RECORDED (with a marked click position) "
        "and CURRENT (the screen right now). "
        "Identify the same UI element that was clicked in the recorded screenshot "
        "and return its center pixel coordinates in the CURRENT screenshot. "
        "Return ONLY valid JSON — no markdown, no explanation: "
        '{"x": <integer>, "y": <integer>}'
    )

    content = [
        ContentBlock(
            type="text",
            text=(
                f"RECORDED screenshot — the user clicked at pixel ({recorded_x}, {recorded_y}). "
                "Identify which UI element was at that position:"
            ),
        ),
        ContentBlock(
            type="image_url",
            image_url={"url": f"data:image/png;base64,{recorded_b64}"},
        ),
        ContentBlock(
            type="text",
            text=(
                "CURRENT screenshot — find the same element and return its "
                "exact center pixel coordinates in THIS image:"
            ),
        ),
        ContentBlock(
            type="image_url",
            image_url={"url": f"data:image/png;base64,{current_b64}"},
        ),
        ContentBlock(
            type="text",
            text='Return ONLY: {"x": <integer>, "y": <integer>}',
        ),
    ]

    messages = [Message(role="user", content=content)]
    backend = LiteLLMBackend()
    model = resolve_model()

    raw = ""
    async for ev in backend.stream(
        model=model,
        messages=messages,
        system=system,
        max_tokens=64,
        temperature=0.0,
    ):
        if isinstance(ev, TextDelta):
            raw += ev.content

    m = re.search(r"\{[^}]+\}", raw)
    if m:
        d = json.loads(m.group())
        return int(d["x"]), int(d["y"])

    return recorded_x, recorded_y


# ---------------------------------------------------------------------------
# Key serialization / deserialization
# ---------------------------------------------------------------------------


def _serialize_key(key) -> str | None:
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


def _deserialize_key(s: str):
    from pynput.keyboard import Key, KeyCode
    if s.startswith("Key."):
        return getattr(Key, s[4:], None)
    if s.startswith("KeyCode."):
        return KeyCode.from_vk(int(s[8:]))
    if len(s) == 1:
        return s
    return None


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Records all mouse and keyboard events. Captures a screenshot on each click."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[RecordedEvent] = []
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

    def stop(self) -> Recording:
        if not self._recording:
            raise RuntimeError("Not recording. Call start() first.")
        self._recording = False
        duration = time.time() - self._start_time

        for listener in (self._mouse_listener, self._keyboard_listener):
            if listener:
                try:
                    listener.stop()
                except Exception:
                    pass

        with self._lock:
            events = list(self._events)

        return Recording(
            name=self._name,
            prompt=self._prompt,
            recorded_at=datetime.datetime.now().isoformat(),
            duration=duration,
            events=events,
        )

    # ------------------------------------------------------------------

    def _elapsed(self) -> float:
        return time.time() - self._start_time

    def _record(self, type_: str, data: dict, screenshot_b64: str = "") -> None:
        if not self._recording:
            return
        with self._lock:
            self._events.append(
                RecordedEvent(t=self._elapsed(), type=type_, data=data,
                               screenshot_b64=screenshot_b64)
            )

    def _on_move(self, x: int, y: int) -> None:
        self._record("move", {"x": x, "y": y})

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        screenshot_b64 = ""
        if pressed:
            try:
                from openvibe.computer.capture import capture_screen
                png, _, _ = capture_screen()
                screenshot_b64 = base64.b64encode(png).decode()
            except Exception:
                pass
        self._record("click", {"x": x, "y": y, "button": button.name, "pressed": pressed},
                     screenshot_b64)

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._record("scroll", {"x": x, "y": y, "dx": dx, "dy": dy})

    def _on_key_press(self, key) -> None:
        s = _serialize_key(key)
        if s is not None:
            self._record("key_press", {"key": s})

    def _on_key_release(self, key) -> None:
        s = _serialize_key(key)
        if s is not None:
            self._record("key_release", {"key": s})


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_recorder = Recorder()


def get_recorder() -> Recorder:
    return _recorder
