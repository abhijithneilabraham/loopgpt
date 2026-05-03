"""Unified cross-platform input control using pynput.

pynput provides consistent, well-tested mouse and keyboard control across
macOS (Quartz), Linux (X11 via Xlib or Wayland via evdev), and Windows
(SendInput).  All functions are synchronous; call them from a thread pool.

Usage::

    from openvibe.computer.input import mouse_click, keyboard_type, keyboard_hotkey

    mouse_click(500, 300)
    keyboard_type("Hello World")
    keyboard_hotkey(["ctrl", "c"])
"""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Sequence


_PLATFORM = platform.system()  # "Darwin" | "Linux" | "Windows"


# ---------------------------------------------------------------------------
# Lazy pynput import
# ---------------------------------------------------------------------------


def _pynput_mouse():
    try:
        from pynput import mouse
        return mouse
    except ImportError as exc:
        raise ImportError(
            "pynput is required for computer use: pip install pynput"
        ) from exc


def _pynput_keyboard():
    try:
        from pynput import keyboard
        return keyboard
    except ImportError as exc:
        raise ImportError(
            "pynput is required for computer use: pip install pynput"
        ) from exc


# ---------------------------------------------------------------------------
# Screen size (used for Retina scaling)
# ---------------------------------------------------------------------------


def screen_size() -> tuple[int, int]:
    """Return ``(width, height)`` of the primary monitor in logical pixels."""
    try:
        import mss
        with mss.mss() as sct:
            m = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            return m["width"], m["height"]
    except ImportError:
        pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


# ---------------------------------------------------------------------------
# Accessibility permission check (macOS only)
# ---------------------------------------------------------------------------


def check_accessibility() -> None:
    """Raise RuntimeError with a clear message if macOS Accessibility is denied.

    pynput raises ``PermissionError`` on the first actual input call when
    Accessibility is denied — this check surfaces the problem early with
    instructions rather than a cryptic PermissionError.
    """
    if _PLATFORM != "Darwin":
        return
    try:
        r = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first process'],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0 and "not allowed" in (r.stderr + r.stdout).lower():
            raise RuntimeError(
                "macOS Accessibility permission is required for mouse/keyboard control.\n"
                "System Settings → Privacy & Security → Accessibility → add your terminal."
            )
    except subprocess.TimeoutExpired:
        pass  # if osascript times out, proceed and let pynput surface any real error


# ---------------------------------------------------------------------------
# Clipboard helpers (Unicode-safe typing)
# ---------------------------------------------------------------------------


def _copy_to_clipboard(text: str) -> None:
    """Copy *text* to the system clipboard."""
    if _PLATFORM == "Darwin":
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)

    elif _PLATFORM == "Linux":
        for cmd in [
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]:
            try:
                subprocess.run(
                    cmd,
                    input=text.encode("utf-8"),
                    check=True,
                    capture_output=True,
                    timeout=5,
                )
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        raise RuntimeError(
            "No clipboard tool found on Linux. Install xclip:\n"
            "  sudo apt install xclip   # Debian/Ubuntu\n"
            "  sudo dnf install xclip   # Fedora"
        )

    else:  # Windows
        try:
            import pyperclip
        except ImportError:
            from openvibe.computer.deps import ensure_import
            pyperclip = ensure_import("pyperclip")
        pyperclip.copy(text)


def _paste_from_clipboard() -> None:
    """Press the platform paste hotkey (Cmd+V or Ctrl+V)."""
    kb = _pynput_keyboard()
    ctrl = kb.Controller()
    modifier = kb.Key.cmd if _PLATFORM == "Darwin" else kb.Key.ctrl
    with ctrl.pressed(modifier):
        ctrl.press("v")
        ctrl.release("v")


# ---------------------------------------------------------------------------
# Mouse
# ---------------------------------------------------------------------------


def _smooth_move(
    controller,
    x: int,
    y: int,
    duration: float,
) -> None:
    """Animate mouse from current position to (x, y) over *duration* seconds."""
    if duration <= 0.01:
        controller.position = (x, y)
        return
    start_x, start_y = controller.position
    steps = max(int(duration * 60), 2)
    for i in range(steps + 1):
        t = i / steps
        ease = t * t * (3.0 - 2.0 * t)  # cubic Hermite smooth-step
        cx = int(start_x + (x - start_x) * ease)
        cy = int(start_y + (y - start_y) * ease)
        controller.position = (cx, cy)
        if i < steps:
            time.sleep(duration / steps)


def get_mouse_position() -> tuple[int, int]:
    """Return the current cursor position in logical screen coordinates."""
    mouse = _pynput_mouse()
    m = mouse.Controller()
    pos = m.position
    return (int(pos[0]), int(pos[1]))


def mouse_move(x: int, y: int, duration: float = 0.25, settle_ms: int = 200) -> str:
    """Move the mouse pointer to (x, y)."""
    check_accessibility()
    mouse = _pynput_mouse()
    m = mouse.Controller()
    _smooth_move(m, x, y, duration)
    time.sleep(settle_ms / 1000.0)
    return f"Moved mouse to ({x}, {y})."


def mouse_click(
    x: int,
    y: int,
    button: str = "left",
    count: int = 1,
    duration: float = 0.1,
    settle_ms: int = 500,
) -> str:
    """Click at (x, y). button: 'left', 'right', 'middle'."""
    check_accessibility()
    mouse = _pynput_mouse()
    btn_map = {
        "left": mouse.Button.left,
        "right": mouse.Button.right,
        "middle": mouse.Button.middle,
    }
    btn = btn_map.get(button.lower(), mouse.Button.left)
    m = mouse.Controller()
    _smooth_move(m, x, y, duration)
    time.sleep(0.02)
    m.click(btn, count)
    time.sleep(settle_ms / 1000.0)
    if count == 2:
        label = "Double-clicked"
    elif button == "right":
        label = "Right-clicked"
    elif button == "middle":
        label = "Middle-clicked"
    else:
        label = "Left-clicked"
    return f"{label} at ({x}, {y})."


def mouse_scroll(
    x: int,
    y: int,
    dx: int = 0,
    dy: int = 3,
    settle_ms: int = 300,
) -> str:
    """Scroll at (x, y). Positive dy = up, negative = down."""
    check_accessibility()
    mouse = _pynput_mouse()
    m = mouse.Controller()
    m.position = (x, y)
    time.sleep(0.05)
    m.scroll(dx, dy)
    time.sleep(settle_ms / 1000.0)
    direction = "up" if dy > 0 else "down"
    return f"Scrolled {direction} {abs(dy)} ticks at ({x}, {y})."


def mouse_drag(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration: float = 0.5,
    settle_ms: int = 500,
) -> str:
    """Drag from (x1, y1) to (x2, y2) with left button held."""
    check_accessibility()
    mouse = _pynput_mouse()
    m = mouse.Controller()
    _smooth_move(m, x1, y1, 0.1)
    time.sleep(0.05)
    m.press(mouse.Button.left)
    time.sleep(0.05)
    _smooth_move(m, x2, y2, duration)
    time.sleep(0.05)
    m.release(mouse.Button.left)
    time.sleep(settle_ms / 1000.0)
    return f"Dragged from ({x1}, {y1}) to ({x2}, {y2})."


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------


# Maps user-facing key names → pynput Key attribute names
_KEY_ALIASES: dict[str, str] = {
    "enter": "enter", "return": "enter",
    "escape": "esc", "esc": "esc",
    "tab": "tab",
    "space": "space",
    "backspace": "backspace",
    "delete": "delete",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end",
    "pageup": "page_up", "page_up": "page_up",
    "pagedown": "page_down", "page_down": "page_down",
    **{f"f{i}": f"f{i}" for i in range(1, 13)},
    "ctrl": "ctrl_l", "control": "ctrl_l",
    "ctrl_l": "ctrl_l", "ctrl_r": "ctrl_r",
    "shift": "shift", "shift_l": "shift_l", "shift_r": "shift_r",
    "alt": "alt_l", "alt_l": "alt_l", "alt_r": "alt_r",
    "option": "alt_l",
    "cmd": "cmd", "command": "cmd",
    "win": "cmd",
    "caps_lock": "caps_lock",
    "insert": "insert",
    "num_lock": "num_lock",
    "scroll_lock": "scroll_lock",
    "pause": "pause",
    "print_screen": "print_screen",
    "media_play_pause": "media_play_pause",
    "media_volume_up": "media_volume_up",
    "media_volume_down": "media_volume_down",
    "media_volume_mute": "media_volume_mute",
    "media_next": "media_next",
    "media_previous": "media_previous",
}


def _resolve_key(name: str):
    """Resolve a key name to a pynput Key enum value or character."""
    kb = _pynput_keyboard()
    mapped = _KEY_ALIASES.get(name.lower(), name.lower())
    key_obj = getattr(kb.Key, mapped, None)
    if key_obj is not None:
        return key_obj
    if len(name) == 1:
        return name
    raise ValueError(
        f"Unknown key: {name!r}. Use names like 'enter', 'escape', 'ctrl', 'f5', "
        f"or a single character. Full list: {sorted(_KEY_ALIASES)}"
    )


def keyboard_type(text: str, settle_ms: int = 300) -> str:
    """Type *text* using clipboard paste (Unicode-safe on all platforms)."""
    check_accessibility()
    _copy_to_clipboard(text)
    time.sleep(0.12)
    _paste_from_clipboard()
    time.sleep(settle_ms / 1000.0)
    preview = text[:40] + ("…" if len(text) > 40 else "")
    return f"Typed {len(text)} chars: {preview!r}"


def keyboard_press(key_name: str, settle_ms: int = 200) -> str:
    """Press and release a single key."""
    check_accessibility()
    kb = _pynput_keyboard()
    controller = kb.Controller()
    key = _resolve_key(key_name)
    controller.press(key)
    time.sleep(0.04)
    controller.release(key)
    time.sleep(settle_ms / 1000.0)
    return f"Pressed: {key_name!r}"


def keyboard_hotkey(keys: Sequence[str], settle_ms: int = 200) -> str:
    """Press a key combination — all keys down in order, then released in reverse."""
    if not keys:
        raise ValueError("keys list must not be empty.")
    check_accessibility()
    kb = _pynput_keyboard()
    controller = kb.Controller()
    resolved = [_resolve_key(k) for k in keys]
    for key in resolved:
        controller.press(key)
        time.sleep(0.02)
    for key in reversed(resolved):
        controller.release(key)
        time.sleep(0.02)
    time.sleep(settle_ms / 1000.0)
    return f"Hotkey: {'+'.join(keys)}"
