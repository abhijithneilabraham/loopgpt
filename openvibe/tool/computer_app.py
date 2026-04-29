"""AppTool — open, close, focus, and list running applications.

Platform support
----------------
macOS   — ``open -a`` / AppleScript via osascript
Linux   — ``xdg-open`` / ``wmctrl`` / ``xdotool``
Windows — ``start`` shell command / pygetwindow

pygetwindow provides cross-platform window title listing and focus on all
three platforms when available (pip install pygetwindow).  Platform-native
commands are used as primary methods because they're more reliable for
actually launching and quitting apps; pygetwindow is used for listing and
focusing where it adds real value.

All actions are gated by the session sandbox allow-list.
"""

from __future__ import annotations

import asyncio
import platform
import subprocess
import time
from typing import Literal

from pydantic import Field

from openvibe.tool.base import Tool, ToolContext, ToolResult

_PLATFORM = platform.system()  # "Darwin" | "Linux" | "Windows"


class AppTool(Tool):
    """Open, close, focus, or list desktop applications."""

    name = "app"
    description = (
        "Interact with desktop applications: open an app by name, close it, "
        "bring it to the foreground, or list all currently running windows. "
        "Also supports waiting for an app to appear and checking if it is running."
    )

    class Params(Tool.Params):
        action: Literal["open", "close", "focus", "list", "is_running"] = Field(
            description=(
                "App action:\n"
                "  open       — launch an application by name or path\n"
                "  close      — quit a running application\n"
                "  focus      — bring a window to the foreground\n"
                "  list       — list all open windows / running applications\n"
                "  is_running — check if an application is currently running"
            )
        )
        name: str | None = Field(
            default=None,
            description=(
                "App name (e.g. 'Terminal', 'Google Chrome', 'VS Code') or full path. "
                "Required for open/close/focus/is_running."
            ),
        )
        wait_seconds: float = Field(
            default=2.0,
            description="Seconds to wait after opening for the app to become ready (default 2).",
        )

    async def execute(self, ctx: ToolContext, params: "AppTool.Params") -> ToolResult:
        from openvibe.computer.sandbox import ActionType, get_sandbox

        app_arg = params.name or "(list)"
        sandbox = get_sandbox(ctx.session_id)
        if not sandbox.is_pre_approved("app"):
            await ctx.check_permission(
                tool="app",
                argument=f"{params.action} {app_arg}",
                description=f"App control: {params.action} '{app_arg}'",
            )

        if params.action in ("open", "close", "focus") and params.name:
            if not sandbox.is_app_allowed(params.name):
                return ToolResult(
                    title="App action denied",
                    output=(
                        f"'{params.name}' is not in the allow-list for this session. "
                        f"Allowed: {sandbox.allowed_apps or ['(all)']}"
                    ),
                    error=True,
                )

        action_map = {
            "open": ActionType.APP_OPEN,
            "close": ActionType.APP_CLOSE,
            "focus": ActionType.APP_FOCUS,
            "list": ActionType.APP_LIST,
            "is_running": ActionType.APP_LIST,
        }

        try:
            loop = asyncio.get_event_loop()
            result_msg = await loop.run_in_executor(None, self._do_action, params)
        except Exception as exc:
            await sandbox.record_action(
                action_map.get(params.action, ActionType.APP_OPEN),
                params={"action": params.action, "name": params.name},
                error=str(exc),
            )
            return ToolResult(
                title="App error",
                output=f"{params.action} '{params.name}' failed: {exc}",
                error=True,
            )

        await sandbox.record_action(
            action_map.get(params.action, ActionType.APP_OPEN),
            params={"action": params.action, "name": params.name},
            result=result_msg[:200],
        )
        return ToolResult(
            title=f"App: {params.action} '{params.name or ''}'",
            output=result_msg,
        )

    @staticmethod
    def _do_action(params: "AppTool.Params") -> str:
        if params.action == "list":
            return _list_windows()
        if params.action == "is_running":
            if not params.name:
                raise ValueError("name is required for is_running.")
            return _is_running(params.name)
        if not params.name:
            raise ValueError(f"name is required for action='{params.action}'.")
        if params.action == "open":
            return _open_app(params.name, params.wait_seconds)
        if params.action == "close":
            return _close_app(params.name)
        if params.action == "focus":
            return _focus_app(params.name)
        raise ValueError(f"Unknown app action: {params.action!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15, **kwargs)


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


def _is_running(name: str) -> str:
    if _PLATFORM == "Darwin":
        r = _run(["pgrep", "-x", "-i", name])
        if r.returncode == 0:
            return f"'{name}' is running (PID: {r.stdout.strip()})."
        return f"'{name}' is not running."

    if _PLATFORM == "Linux":
        r = _run(["pgrep", "-f", name])
        return f"'{name}' is running." if r.returncode == 0 else f"'{name}' is not running."

    if _PLATFORM == "Windows":
        r = _run(["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"])
        return (
            f"'{name}' is running."
            if name.lower() in r.stdout.lower()
            else f"'{name}' is not running."
        )
    raise RuntimeError(f"Unsupported platform: {_PLATFORM}")


# ---------------------------------------------------------------------------
# open
# ---------------------------------------------------------------------------


def _open_app(name: str, wait_seconds: float = 2.0) -> str:
    if _PLATFORM == "Darwin":
        r = _run(["open", "-a", name])
        if r.returncode != 0:
            r2 = _run(["open", name])
            if r2.returncode != 0:
                raise RuntimeError(r.stderr.strip() or r2.stderr.strip())
        time.sleep(wait_seconds)
        _run(["osascript", "-e", f'tell application "{name}" to activate'])
        return f"Opened '{name}' on macOS."

    if _PLATFORM == "Linux":
        try:
            subprocess.Popen([name], start_new_session=True)
        except FileNotFoundError:
            subprocess.Popen(["xdg-open", name], start_new_session=True)
        time.sleep(wait_seconds)
        return f"Opened '{name}' on Linux."

    if _PLATFORM == "Windows":
        subprocess.Popen(["start", "", name], shell=True, start_new_session=True)
        time.sleep(wait_seconds)
        return f"Opened '{name}' on Windows."

    raise RuntimeError(f"Unsupported platform: {_PLATFORM}")


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def _close_app(name: str) -> str:
    if _PLATFORM == "Darwin":
        r = _run(["osascript", "-e", f'tell application "{name}" to quit'])
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        return f"Quit '{name}'."

    if _PLATFORM == "Linux":
        r = _run(["pkill", "-f", name])
        if r.returncode not in (0, 1):
            raise RuntimeError(r.stderr.strip())
        return f"Sent SIGTERM to processes matching '{name}'."

    if _PLATFORM == "Windows":
        r = _run(["taskkill", "/IM", name, "/F"])
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        return f"Terminated '{name}'."

    raise RuntimeError(f"Unsupported platform: {_PLATFORM}")


# ---------------------------------------------------------------------------
# focus
# ---------------------------------------------------------------------------


def _focus_app(name: str) -> str:
    # Try pygetwindow first (cross-platform, reliable title matching)
    try:
        import pygetwindow as gw  # type: ignore[import-not-found]
        wins = gw.getWindowsWithTitle(name)
        if not wins:
            # Case-insensitive partial match
            wins = [w for w in gw.getAllWindows() if name.lower() in w.title.lower()]
        if wins:
            win = wins[0]
            try:
                win.activate()
            except Exception:
                win.restore()
                win.activate()
            return f"Focused '{win.title}'."
    except ImportError:
        pass
    except Exception:
        pass

    # Platform-native fallbacks
    if _PLATFORM == "Darwin":
        r = _run(["osascript", "-e", f'tell application "{name}" to activate'])
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        return f"Focused '{name}'."

    if _PLATFORM == "Linux":
        r = _run(["wmctrl", "-a", name])
        if r.returncode == 0:
            return f"Focused window '{name}'."
        r2 = _run(["xdotool", "search", "--name", name, "windowactivate"])
        if r2.returncode == 0:
            return f"Focused window '{name}'."
        raise RuntimeError(
            f"Could not focus '{name}'. "
            "Install wmctrl: sudo apt install wmctrl"
        )

    if _PLATFORM == "Windows":
        raise RuntimeError(
            f"Could not focus '{name}'. "
            "Install pygetwindow: pip install pygetwindow"
        )

    raise RuntimeError(f"Unsupported platform: {_PLATFORM}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _list_windows() -> str:
    # pygetwindow provides a clean cross-platform window list
    try:
        import pygetwindow as gw  # type: ignore[import-not-found]
        titles = sorted({w.title.strip() for w in gw.getAllWindows() if w.title.strip()})
        if titles:
            return f"Open windows ({len(titles)}):\n" + "\n".join(
                f"  • {t}" for t in titles
            )
    except ImportError:
        pass
    except Exception:
        pass

    # Platform-native fallbacks
    if _PLATFORM == "Darwin":
        script = (
            'tell application "System Events" to get the name of every process '
            'whose background only is false'
        )
        r = _run(["osascript", "-e", script])
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        names = sorted({n.strip() for n in r.stdout.strip().split(",") if n.strip()})
        return "Running applications:\n" + "\n".join(f"  • {n}" for n in names)

    if _PLATFORM == "Linux":
        r = _run(["wmctrl", "-l"])
        if r.returncode == 0:
            lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
            return f"Open windows ({len(lines)}):\n" + "\n".join(
                f"  • {ln}" for ln in lines
            )
        r2 = _run(["ps", "-eo", "comm="])
        if r2.returncode == 0:
            procs = sorted(set(r2.stdout.strip().splitlines()))
            return "Running processes:\n" + "\n".join(f"  • {p}" for p in procs[:50])
        raise RuntimeError("Could not list windows: install wmctrl (sudo apt install wmctrl).")

    if _PLATFORM == "Windows":
        r = _run(["tasklist", "/FO", "CSV", "/NH"])
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()[:30]
            return "Running processes:\n" + "\n".join(f"  • {l}" for l in lines)
        raise RuntimeError("Could not list processes on Windows.")

    raise RuntimeError(f"Unsupported platform: {_PLATFORM}")
