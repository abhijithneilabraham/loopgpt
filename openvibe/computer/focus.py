"""OS-level focus and active window detection.

Returns a compact description of what currently has input focus —
the active application, window title, and focused UI element.

On macOS: AppleScript + AXUIElement (no extra deps required).
On Linux: xdotool (must be installed: sudo apt install xdotool).
On Windows: ctypes GetForegroundWindow + WMIC.

All functions are synchronous — call from a thread pool when needed.

Example::

    from openvibe.computer.focus import get_focus_context
    print(get_focus_context())
    # "App: Chrome | Window: GitHub – Pull Requests | Element: AXTextField (address bar)"
"""

from __future__ import annotations

import platform
import subprocess

_PLATFORM = platform.system()  # "Darwin" | "Linux" | "Windows"

_MACOS_SCRIPT = """\
set parts to {}
tell application "System Events"
    set frontProc to first application process whose frontmost is true
    set end of parts to "App: " & (name of frontProc)
    try
        set win to window 1 of frontProc
        set end of parts to "Window: " & (name of win)
    end try
    try
        set focEl to value of attribute "AXFocusedUIElement" of frontProc
        set elRole to value of attribute "AXRole" of focEl
        set label to elRole
        try
            set d to value of attribute "AXDescription" of focEl
            if d is not "" then set label to elRole & " (" & d & ")"
        end try
        if label is elRole then try
            set t to value of attribute "AXTitle" of focEl
            if t is not "" then set label to elRole & " (" & t & ")"
        end try
        set end of parts to "Element: " & label
    end try
end tell
set AppleScript's text item delimiters to " | "
set output to parts as string
set AppleScript's text item delimiters to ""
return output
"""


def get_focus_context() -> str:
    """Return a one-line description of the current input focus.

    Examples::

        "App: Chrome | Window: Google – New Tab | Element: AXTextField (address bar)"
        "App: Terminal | Window: bash"
        "(focus detection unavailable)"
    """
    try:
        if _PLATFORM == "Darwin":
            return _focus_macos()
        if _PLATFORM == "Linux":
            return _focus_linux()
        if _PLATFORM == "Windows":
            return _focus_windows()
        return f"(focus detection not supported on {_PLATFORM})"
    except Exception as exc:
        return f"(focus error: {exc})"


def _focus_macos() -> str:
    r = subprocess.run(
        ["osascript", "-e", _MACOS_SCRIPT],
        capture_output=True, text=True, timeout=3,
    )
    out = (r.stdout or "").strip()
    if not out:
        return "(no focus info)"
    return out


def _focus_linux() -> str:
    try:
        win_id = subprocess.check_output(
            ["xdotool", "getactivewindow"], timeout=2, text=True,
        ).strip()
        title = subprocess.check_output(
            ["xdotool", "getwindowname", win_id], timeout=2, text=True,
        ).strip()
        pid = subprocess.check_output(
            ["xdotool", "getwindowpid", win_id], timeout=2, text=True,
        ).strip()
        try:
            app = subprocess.check_output(
                ["ps", "-p", pid, "-o", "comm="], timeout=2, text=True,
            ).strip()
        except Exception:
            app = f"pid:{pid}"
        return f"App: {app} | Window: {title}"
    except FileNotFoundError:
        return "(xdotool not installed — sudo apt install xdotool)"
    except subprocess.TimeoutExpired:
        return "(focus query timed out)"


def _focus_windows() -> str:
    try:
        import ctypes
        import ctypes.wintypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        pid = ctypes.wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            app = subprocess.check_output(
                ["wmic", "process", "where",
                 f"ProcessId={pid.value}", "get", "Name", "/value"],
                timeout=2, text=True,
            ).strip().split("=")[-1]
        except Exception:
            app = f"pid:{pid.value}"
        return f"App: {app} | Window: {title}"
    except Exception as exc:
        return f"(windows focus error: {exc})"
