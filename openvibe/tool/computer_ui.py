"""UITool — accessibility-tree UI interaction (no pixel coords needed).

macOS: atomacos → AppleScript fallback.
Linux: pyatspi → xdotool fallback.
Windows: pywinauto.
"""

from __future__ import annotations

import platform
import subprocess
import time
from typing import Any, Literal

from pydantic import Field

from openvibe.tool.base import Tool, ToolContext, ToolResult

_PLATFORM = platform.system()  # "Darwin" | "Linux" | "Windows"


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class UITool(Tool):
    """Interact with desktop UI elements by name — no pixel coordinates needed.

    Prefer this over the mouse tool whenever the target element has a visible
    label or title.  Fall back to screenshot + mouse only for unlabelled
    canvas areas (games, video players, custom renderers).
    """

    name = "ui"
    description = (
        "Interact with UI elements by accessible name on any OS — no coordinates. "
        "ALWAYS try this before the mouse tool. Actions:\n"
        "  get_tree   — list all accessible elements in the app window\n"
        "  click      — click a button/element by its title\n"
        "  click_menu — click a menu item, e.g. File → Save\n"
        "  type       — type text into the focused element (clipboard-paste, Unicode-safe)\n"
        "  press_key  — press a key or chord: key='return', modifiers=['command']\n"
        "  get_value  — read the current text/value of a named element"
    )

    class Params(Tool.Params):
        action: Literal[
            "get_tree", "click", "click_menu", "type", "press_key", "get_value"
        ] = Field(description="UI action to perform.")
        app: str = Field(
            description=(
                "Application name / process name.\n"
                "macOS: localized name as shown in Activity Monitor, e.g. 'TextEdit'.\n"
                "Linux: process name or window title substring, e.g. 'gedit'.\n"
                "Windows: executable or window title, e.g. 'Notepad'."
            )
        )
        title: str | None = Field(
            default=None,
            description="Title or label of the UI element (for click, get_value).",
        )
        role: str | None = Field(
            default=None,
            description=(
                "Element role to narrow the search (optional). "
                "macOS: 'button', 'text field', 'text area', 'checkbox'. "
                "Linux: 'push button', 'entry', 'check box'. "
                "Windows: 'Button', 'Edit', 'CheckBox'."
            ),
        )
        text: str | None = Field(
            default=None,
            description="Text to type (for action='type').",
        )
        menu: str | None = Field(
            default=None,
            description="Menu bar menu name for click_menu, e.g. 'File', 'Edit'.",
        )
        menu_item: str | None = Field(
            default=None,
            description="Menu item name for click_menu, e.g. 'Save', 'Copy'.",
        )
        key: str | None = Field(
            default=None,
            description=(
                "Key for press_key: 'return', 'escape', 'tab', 'space', 'delete', "
                "'up', 'down', 'left', 'right', or a single character."
            ),
        )
        modifiers: list[str] = Field(
            default_factory=list,
            description=(
                "Modifier keys for press_key: 'command' (macOS), 'ctrl', 'shift', 'alt'. "
                "E.g. ['command'] for Cmd+key."
            ),
        )
        window_index: int = Field(
            default=1,
            description="Window index (1 = frontmost). Used by get_tree, click, get_value.",
        )

    async def execute(self, ctx: ToolContext, params: "UITool.Params") -> ToolResult:
        import asyncio
        from openvibe.computer.sandbox import ActionType, get_sandbox

        await ctx.check_permission(
            tool="ui",
            argument=f"{params.action} in {params.app}",
            description=f"UI accessibility: {params.action} in '{params.app}'",
        )

        sandbox = get_sandbox(ctx.session_id)
        loop = asyncio.get_event_loop()

        try:
            result_msg = await loop.run_in_executor(None, self._do_action, params)
        except (RuntimeError, ValueError, ImportError) as exc:
            await sandbox.record_action(
                ActionType.MOUSE_CLICK,
                params={"action": params.action, "app": params.app, "title": params.title},
                error=str(exc),
            )
            return ToolResult(
                title=f"UI error: {params.action} in {params.app}",
                output=str(exc),
                error=True,
            )
        except Exception as exc:
            return ToolResult(
                title=f"UI error: {params.action}",
                output=str(exc),
                error=True,
            )

        await sandbox.record_action(
            ActionType.MOUSE_CLICK,
            params={"action": params.action, "app": params.app, "title": params.title},
            result=result_msg[:200],
        )
        label = params.title or params.menu_item or params.key or ""
        return ToolResult(
            title=f"UI: {params.action} '{label}' in {params.app}",
            output=result_msg,
        )

    @staticmethod
    def _do_action(params: "UITool.Params") -> str:
        if _PLATFORM == "Darwin":
            return _macos_dispatch(params)
        if _PLATFORM == "Linux":
            return _linux_dispatch(params)
        if _PLATFORM == "Windows":
            return _windows_dispatch(params)
        raise RuntimeError(
            f"UITool: unsupported platform '{_PLATFORM}'. "
            "Use screenshot + mouse tools instead."
        )


# macOS backend — atomacos → AppleScript fallback


def _try_atomacos():
    """Return the atomacos module or None if not installed."""
    try:
        import atomacos
        return atomacos
    except ImportError:
        return None


def _macos_dispatch(params: "UITool.Params") -> str:
    ax = _try_atomacos()
    if ax is not None:
        return _atomacos_dispatch(ax, params)
    return _applescript_dispatch(params)


# ---- atomacos path ----------------------------------------------------------


def _ax_find_app(ax, name: str):
    """Return atomacos app ref for name, or raise RuntimeError."""
    errors: list[str] = []

    try:
        return ax.getAppRefByLocalizedName(name)
    except Exception as e:
        errors.append(f"localized name: {e}")

    try:  # bundle ID
        return ax.getAppRefByBundleId(name)
    except Exception as e:
        errors.append(f"bundle id: {e}")

    raise RuntimeError(
        f"Could not find app '{name}' via macOS Accessibility. "
        "Make sure the app is running and Accessibility permission is granted "
        "(System Settings → Privacy & Security → Accessibility). "
        f"Attempted: {'; '.join(errors)}"
    )


def _ax_walk(elem: Any, max_depth: int = 12):
    """Yield all AX elements depth-first up to *max_depth*."""
    yield elem, 0

    def _recurse(node, depth):
        if depth >= max_depth:
            return
        try:
            children = node.AXChildren or []
        except Exception:
            return
        for child in children:
            yield child, depth
            yield from _recurse(child, depth + 1)

    yield from _recurse(elem, 1)


def _ax_get_label(elem: Any) -> str:
    """Return the best human-readable label for an AX element."""
    for attr in ("AXTitle", "AXDescription", "AXPlaceholderValue", "AXValue", "AXHelp"):
        try:
            val = getattr(elem, attr, None)
            if val and isinstance(val, str):
                return val
        except Exception:
            pass
    return ""


def _ax_get_role(elem: Any) -> str:
    try:
        r = elem.AXRole or ""
        return r.replace("AX", "").lower()
    except Exception:
        return ""


def _ax_find_elem(
    root: Any,
    title: str | None,
    role: str | None,
) -> Any | None:
    """Find the first AX element matching title and/or role."""
    title_l = title.lower() if title else None
    role_l = role.lower() if role else None

    for elem, _ in _ax_walk(root):
        try:
            elem_role = _ax_get_role(elem)
            elem_label = _ax_get_label(elem).lower()

            role_ok = not role_l or role_l in elem_role
            title_ok = not title_l or title_l in elem_label

            if role_ok and title_ok and (title_l or role_l):
                return elem
        except Exception:
            continue
    return None


def _atomacos_dispatch(ax: Any, params: "UITool.Params") -> str:
    if params.action == "get_tree":
        return _ax_get_tree(ax, params.app, params.window_index)
    if params.action == "click":
        return _ax_click(ax, params.app, params.title, params.role, params.window_index)
    if params.action == "click_menu":
        return _ax_click_menu(ax, params.app, params.menu, params.menu_item)
    if params.action == "type":
        return _ax_type(ax, params.app, params.text)
    if params.action == "press_key":
        return _ax_press_key(ax, params.app, params.key, params.modifiers)
    if params.action == "get_value":
        return _ax_get_value(ax, params.app, params.title, params.role, params.window_index)
    raise ValueError(f"Unknown action: {params.action!r}")


def _ax_get_tree(ax: Any, app_name: str, window_index: int = 1) -> str:
    app = _ax_find_app(ax, app_name)
    try:
        windows = app.windows()
    except Exception:
        windows = []

    if not windows:
        return (
            f"No windows found for '{app_name}'. "
            "App may use custom rendering. Fall back to screenshot + mouse."
        )
    idx = min(window_index - 1, len(windows) - 1)
    win = windows[idx]
    win_title = _ax_get_label(win) or "(untitled)"
    lines = [f"Window {window_index}: {win_title!r}"]
    seen = 0
    for elem, depth in _ax_walk(win, max_depth=10):
        if seen >= 300:
            lines.append("  … (truncated — use a more specific app/window)")
            break
        role = _ax_get_role(elem)
        label = _ax_get_label(elem)
        if label and role:
            lines.append(f"{'  ' * min(depth, 4)}[{role}] {label}")
            seen += 1
    return "\n".join(lines)


def _ax_click(
    ax: Any,
    app_name: str,
    title: str | None,
    role: str | None,
    window_index: int = 1,
) -> str:
    if not title and not role:
        raise ValueError("Provide at least 'title' or 'role' for click.")
    app = _ax_find_app(ax, app_name)
    windows = app.windows()
    if not windows:
        raise RuntimeError(f"No windows open in '{app_name}'.")
    win = windows[min(window_index - 1, len(windows) - 1)]
    elem = _ax_find_elem(win, title, role)
    if elem is None:
        raise RuntimeError(
            f"Element title='{title}' role='{role}' not found in '{app_name}'. "
            "Run get_tree to see available elements."
        )
    elem_label = _ax_get_label(elem) or title or role or "element"
    elem_role = _ax_get_role(elem)

    # Try AXPress first (works for native buttons, checkboxes, etc.)
    pressed = False
    press_exc = None
    try:
        elem.Press()
        pressed = True
    except Exception as exc:
        press_exc = exc

    if pressed:
        return f"Clicked [{elem_role}] '{elem_label}' in {app_name}."

    # Fallback: click at the element's center coordinates.
    # This works for browser text fields, web inputs, and any element where
    # AXPress is not implemented (web content exposed via accessibility).
    try:
        pos = elem.AXPosition   # CGPoint (x, y) in screen logical coords
        sz = elem.AXSize        # CGSize (width, height)
        cx = int(pos.x + sz.width / 2)
        cy = int(pos.y + sz.height / 2)
        from openvibe.computer.input import mouse_click
        mouse_click(cx, cy, "left", 1, 0.1, 300)
        return (
            f"Clicked [{elem_role}] '{elem_label}' in {app_name} "
            f"at ({cx},{cy}) via coordinates."
        )
    except Exception as coord_exc:
        raise RuntimeError(
            f"Could not click element '{elem_label}': "
            f"Press failed ({press_exc}); coordinate fallback failed ({coord_exc})"
        ) from coord_exc


def _ax_click_menu(
    ax: Any,
    app_name: str,
    menu: str | None,
    menu_item: str | None,
) -> str:
    if not menu or not menu_item:
        raise ValueError("Both 'menu' and 'menu_item' are required for click_menu.")
    app = _ax_find_app(ax, app_name)
    try:
        menu_bar = app.menuBar()
        menu_elem = _ax_find_elem(menu_bar, menu, "menu")
        if menu_elem is None:
            raise RuntimeError(f"Menu '{menu}' not found in '{app_name}'.")
        menu_elem.Press()
        time.sleep(0.15)
        item_elem = _ax_find_elem(menu_elem, menu_item, "menu item")
        if item_elem is None:
            raise RuntimeError(f"Menu item '{menu_item}' not found under '{menu}'.")
        item_elem.Press()
        return f"Clicked {menu} → {menu_item} in {app_name}."
    except (RuntimeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Menu click failed: {exc}") from exc


def _ax_type(ax: Any, app_name: str, text: str | None) -> str:
    if not text:
        raise ValueError("'text' is required for action='type'.")
    app = _ax_find_app(ax, app_name)
    app.activate()
    time.sleep(0.2)
    from openvibe.computer.input import keyboard_type
    return keyboard_type(text)


def _ax_press_key(
    ax: Any,
    app_name: str,
    key: str | None,
    modifiers: list[str],
) -> str:
    if not key:
        raise ValueError("'key' is required for action='press_key'.")
    app = _ax_find_app(ax, app_name)
    app.activate()
    time.sleep(0.1)
    if modifiers:
        from openvibe.computer.input import keyboard_hotkey
        return keyboard_hotkey(modifiers + [key])
    from openvibe.computer.input import keyboard_press
    return keyboard_press(key)


def _ax_get_value(
    ax: Any,
    app_name: str,
    title: str | None,
    role: str | None,
    window_index: int = 1,
) -> str:
    if not title and not role:
        raise ValueError("Provide 'title' or 'role' for get_value.")
    app = _ax_find_app(ax, app_name)
    windows = app.windows()
    if not windows:
        raise RuntimeError(f"No windows open in '{app_name}'.")
    win = windows[min(window_index - 1, len(windows) - 1)]
    elem = _ax_find_elem(win, title, role)
    if elem is None:
        raise RuntimeError(f"Element title='{title}' role='{role}' not found.")
    # AXValue (text fields) → AXTitle (buttons) → AXDescription
    for attr in ("AXValue", "AXTitle", "AXDescription"):
        try:
            val = getattr(elem, attr, None)
            if val is not None:
                return str(val)
        except Exception:
            pass
    return "(no value)"


# ---- AppleScript fallback (when atomacos is not installed) ------------------


def _osascript(script: str, timeout: int = 15) -> str:
    r = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip())
    return r.stdout.strip()


def _applescript_dispatch(params: "UITool.Params") -> str:
    """AppleScript fallback — used when atomacos is not installed."""
    if params.action == "get_tree":
        return _as_get_tree(params.app, params.window_index)
    if params.action == "click":
        return _as_click(params.app, params.title, params.role, params.window_index)
    if params.action == "click_menu":
        return _as_click_menu(params.app, params.menu, params.menu_item)
    if params.action == "type":
        return _as_type(params.app, params.text)
    if params.action == "press_key":
        return _as_press_key(params.app, params.key, params.modifiers)
    if params.action == "get_value":
        return _as_get_value(params.app, params.title, params.role, params.window_index)
    raise ValueError(f"Unknown action: {params.action!r}")


def _as_get_tree(app: str, window_index: int = 1) -> str:
    script = f'''
tell application "System Events"
    tell process "{app}"
        set win to window {window_index}
        set winTitle to ""
        try
            set winTitle to title of win
        end try
        set output to "Window {window_index}: \\"" & winTitle & "\\"\\n"
        set allElems to entire contents of win
        repeat with elem in allElems
            try
                set r to role of elem
                set t to ""
                try
                    set t to title of elem
                end try
                if t is "" then
                    try
                        set t to value of elem as text
                    on error
                        set t to ""
                    end try
                end if
                if t is not "" and t is not missing value then
                    set output to output & "  [" & r & "] " & t & "\\n"
                end if
            end try
        end repeat
        return output
    end tell
end tell
'''
    result = _osascript(script)
    if not result.strip():
        return (
            f"No accessible elements in '{app}' window {window_index}. "
            "Install atomacos for better results: pip install atomacos"
        )
    return result


def _as_click(
    app: str, title: str | None, role: str | None, window_index: int = 1
) -> str:
    if not title and not role:
        raise ValueError("Provide at least 'title' or 'role' for click.")
    if title:
        esc = title.replace('"', '\\"')
        script = f'''
tell application "System Events"
    tell process "{app}"
        set allElems to entire contents of window {window_index}
        repeat with elem in allElems
            try
                if title of elem is "{esc}" then
                    click elem
                    return "Clicked [" & (role of elem) & "] \\"{esc}\\" in {app}."
                end if
            end try
        end repeat
        error "No element titled \\"{esc}\\" in {app}."
    end tell
end tell
'''
    else:
        script = f'''
tell application "System Events"
    tell process "{app}"
        click first {role} of window {window_index}
        return "Clicked first [{role}] in {app}."
    end tell
end tell
'''
    return _osascript(script)


def _as_click_menu(app: str, menu: str | None, menu_item: str | None) -> str:
    if not menu or not menu_item:
        raise ValueError("Both 'menu' and 'menu_item' are required.")
    em, ei = menu.replace('"', '\\"'), menu_item.replace('"', '\\"')
    script = f'''
tell application "System Events"
    tell process "{app}"
        click menu item "{ei}" of menu "{em}" of menu bar item "{em}" of menu bar 1
        return "Clicked {menu} → {menu_item} in {app}."
    end tell
end tell
'''
    return _osascript(script)


def _as_type(app: str, text: str | None) -> str:
    if not text:
        raise ValueError("'text' is required for action='type'.")
    _osascript(f'tell application "{app}" to activate')
    time.sleep(0.2)
    from openvibe.computer.input import keyboard_type
    return keyboard_type(text)


_AS_KEY_MAP: dict[str, tuple[str, bool]] = {
    "return": ("return", False), "enter": ("return", False),
    "escape": ("escape", False), "esc": ("escape", False),
    "tab": ("tab", False), "space": ("space", False),
    "delete": ("delete", False), "backspace": ("delete", False),
    "up": ("126", True), "down": ("125", True),
    "left": ("123", True), "right": ("124", True),
    "home": ("115", True), "end": ("119", True),
    "pageup": ("116", True), "pagedown": ("121", True),
    **{f"f{i}": (str(c), True) for i, c in zip(
        range(1, 13), [122,120,99,118,96,97,98,100,101,109,103,111]
    )},
}
_AS_MOD_MAP = {
    "command": "command down", "cmd": "command down",
    "shift": "shift down", "option": "option down",
    "alt": "option down", "control": "control down", "ctrl": "control down",
}


def _as_press_key(app: str, key: str | None, modifiers: list[str]) -> str:
    if not key:
        raise ValueError("'key' is required for action='press_key'.")
    _osascript(f'tell application "{app}" to activate')
    time.sleep(0.1)
    from openvibe.computer.input import keyboard_hotkey, keyboard_press
    if modifiers:
        return keyboard_hotkey(modifiers + [key])
    return keyboard_press(key)


def _as_get_value(
    app: str, title: str | None, role: str | None, window_index: int = 1
) -> str:
    if title:
        esc = title.replace('"', '\\"')
        script = f'''
tell application "System Events"
    tell process "{app}"
        set allElems to entire contents of window {window_index}
        repeat with elem in allElems
            try
                if title of elem is "{esc}" then
                    return value of elem as text
                end if
            end try
        end repeat
        error "No element titled \\"{esc}\\"."
    end tell
end tell
'''
    elif role:
        script = f'''
tell application "System Events"
    tell process "{app}"
        return value of first {role} of window {window_index} as text
    end tell
end tell
'''
    else:
        raise ValueError("Provide 'title' or 'role' for get_value.")
    return _osascript(script)


# Linux backend — pyatspi → xdotool fallback


def _linux_dispatch(params: "UITool.Params") -> str:
    try:
        import pyatspi  # noqa: F401
        return _atspi_dispatch(params)
    except ImportError:
        return _xdotool_dispatch(params)




def _atspi_find_app(name: str):
    import pyatspi
    desktop = pyatspi.Registry.getDesktop(0)
    name_lower = name.lower()
    for app in desktop:
        if app and app.name and app.name.lower() == name_lower:
            return app
    for app in desktop:
        if app and app.name and name_lower in app.name.lower():
            return app
    available = [a.name for a in desktop if a and a.name]
    raise RuntimeError(
        f"App '{name}' not found in AT-SPI tree. "
        f"Running: {', '.join(available) or 'none'}. "
        "Ensure AT-SPI2 accessibility is enabled "
        "(export GTK_MODULES=gail:atk-bridge for older GTK apps)."
    )


def _atspi_walk(node: Any, depth: int = 0, max_depth: int = 10):
    if depth > max_depth:
        return
    yield node
    try:
        for i in range(node.childCount):
            try:
                yield from _atspi_walk(node.getChildAtIndex(i), depth + 1, max_depth)
            except Exception:
                pass
    except Exception:
        pass


def _atspi_find(app_node: Any, title: str | None, role: str | None) -> Any:
    title_lower = title.lower() if title else None
    role_lower = role.lower() if role else None
    for node in _atspi_walk(app_node):
        try:
            name = (node.name or "").lower()
            node_role = (node.getLocalizedRoleName() or "").lower()
            name_ok = not title_lower or title_lower in name
            role_ok = not role_lower or role_lower in node_role
            if name_ok and role_ok and (title_lower or role_lower):
                if node.name or node.getLocalizedRoleName():
                    return node
        except Exception:
            pass
    return None


def _atspi_dispatch(params: "UITool.Params") -> str:
    if params.action == "get_tree":
        return _atspi_get_tree(params.app, params.window_index)
    if params.action == "click":
        return _atspi_click(params.app, params.title, params.role)
    if params.action == "click_menu":
        return _atspi_click_menu(params.app, params.menu, params.menu_item)
    if params.action == "type":
        return _linux_type(params.app, params.text)
    if params.action == "press_key":
        return _linux_press_key(params.app, params.key, params.modifiers)
    if params.action == "get_value":
        return _atspi_get_value(params.app, params.title, params.role)
    raise ValueError(f"Unknown action: {params.action!r}")


def _atspi_get_tree(app_name: str, window_index: int = 1) -> str:
    app = _atspi_find_app(app_name)
    lines = [f"App: {app.name}"]
    seen = 0
    for node in _atspi_walk(app, max_depth=6):
        if seen >= 120:
            lines.append("  … (truncated)")
            break
        try:
            name = node.name or ""
            role = node.getLocalizedRoleName() or ""
            if name or role:
                lines.append(f"  [{role}] {name}")
                seen += 1
        except Exception:
            pass
    return "\n".join(lines) if len(lines) > 1 else (
        f"No accessible elements in '{app_name}'. "
        "App may use custom rendering. Fall back to mouse tool."
    )


def _atspi_click(app_name: str, title: str | None, role: str | None) -> str:
    import pyatspi
    app = _atspi_find_app(app_name)
    node = _atspi_find(app, title, role)
    if node is None:
        raise RuntimeError(
            f"No element title='{title}' role='{role}' in '{app_name}'. "
            "Run get_tree to see available elements."
        )
    try:
        action = node.queryAction()
        names = [action.getName(i).lower() for i in range(action.nActions)]
        for pref in ("click", "press", "activate", "toggle"):
            if pref in names:
                action.doAction(names.index(pref))
                return f"Clicked [{node.getLocalizedRoleName()}] '{node.name}' in {app_name}."
        if action.nActions > 0:
            action.doAction(0)
            return f"Activated [{node.getLocalizedRoleName()}] '{node.name}' in {app_name}."
    except Exception:
        pass
    try:
        bbox = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
        cx, cy = bbox.x + bbox.width // 2, bbox.y + bbox.height // 2
        subprocess.run(["xdotool", "mousemove", "--sync", str(cx), str(cy)], check=True)
        subprocess.run(["xdotool", "click", "1"], check=True)
        return f"Clicked at ({cx},{cy}) [{node.getLocalizedRoleName()}] '{node.name}'."
    except Exception as exc:
        raise RuntimeError(f"Could not click '{title}': {exc}") from exc


def _atspi_click_menu(app_name: str, menu: str | None, menu_item: str | None) -> str:
    import pyatspi
    if not menu or not menu_item:
        raise ValueError("Both 'menu' and 'menu_item' are required.")
    app = _atspi_find_app(app_name)
    menu_bar = _atspi_find(app, None, "menu bar") or _atspi_find(app, None, "menubar")
    if menu_bar is None:
        raise RuntimeError(f"No menu bar found in '{app_name}'.")
    menu_node = _atspi_find(menu_bar, menu, "menu")
    if menu_node is None:
        raise RuntimeError(f"Menu '{menu}' not found.")
    try:
        menu_node.queryAction().doAction(0)
        time.sleep(0.2)
    except Exception:
        pass
    item_node = _atspi_find(menu_node, menu_item, "menu item")
    if item_node is None:
        raise RuntimeError(f"Menu item '{menu_item}' not found under '{menu}'.")
    try:
        item_node.queryAction().doAction(0)
    except Exception:
        bbox = item_node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
        cx, cy = bbox.x + bbox.width // 2, bbox.y + bbox.height // 2
        subprocess.run(["xdotool", "mousemove", "--sync", str(cx), str(cy)], check=True)
        subprocess.run(["xdotool", "click", "1"], check=True)
    return f"Clicked {menu} → {menu_item} in {app_name}."


def _atspi_get_value(app_name: str, title: str | None, role: str | None) -> str:
    app = _atspi_find_app(app_name)
    node = _atspi_find(app, title, role)
    if node is None:
        raise RuntimeError(f"No element title='{title}' role='{role}' in '{app_name}'.")
    try:
        return str(node.queryValue().currentValue)
    except Exception:
        pass
    try:
        return node.queryText().getText(0, -1)
    except Exception:
        pass
    return node.name or "(no value)"




def _has_xdotool() -> bool:
    return subprocess.run(["which", "xdotool"], capture_output=True).returncode == 0


def _xdotool_dispatch(params: "UITool.Params") -> str:
    if params.action == "type":
        return _linux_type(params.app, params.text)
    if params.action == "press_key":
        return _linux_press_key(params.app, params.key, params.modifiers)
    if not _has_xdotool():
        raise ImportError(
            f"pyatspi is required for '{params.action}' on Linux: "
            "pip install pyatspi  (or: sudo apt install python3-pyatspi)\n"
            "xdotool also not found: sudo apt install xdotool"
        )
    raise ImportError(
        f"action='{params.action}' requires pyatspi for element discovery. "
        "pip install pyatspi  (or: sudo apt install python3-pyatspi)"
    )


def _linux_focus(app_name: str) -> None:
    for cmd in [
        ["xdotool", "search", "--name", app_name, "windowactivate", "--sync"],
        ["xdotool", "search", "--class", app_name, "windowactivate", "--sync"],
    ]:
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            time.sleep(0.2)
            return


def _linux_type(app_name: str, text: str | None) -> str:
    if not text:
        raise ValueError("'text' is required for action='type'.")
    _linux_focus(app_name)
    from openvibe.computer.input import keyboard_type
    return keyboard_type(text)


_LINUX_KEY_MAP = {
    "return": "Return", "enter": "Return",
    "escape": "Escape", "esc": "Escape",
    "tab": "Tab", "space": "space",
    "delete": "Delete", "backspace": "BackSpace",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End",
    "pageup": "Prior", "pagedown": "Next",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}
_LINUX_MOD_MAP = {
    "command": "ctrl", "cmd": "ctrl",
    "ctrl": "ctrl", "control": "ctrl",
    "shift": "shift",
    "alt": "alt", "option": "alt",
}


def _linux_press_key(app_name: str, key: str | None, modifiers: list[str]) -> str:
    if not key:
        raise ValueError("'key' is required for action='press_key'.")
    if not _has_xdotool():
        raise RuntimeError("xdotool required: sudo apt install xdotool")
    _linux_focus(app_name)
    mapped_key = _LINUX_KEY_MAP.get(key.lower(), key)
    mapped_mods = [_LINUX_MOD_MAP.get(m.lower(), m) for m in modifiers]
    combo = "+".join(mapped_mods + [mapped_key]) if mapped_mods else mapped_key
    subprocess.run(["xdotool", "key", "--clearmodifiers", combo], check=True)
    display = ("+".join(modifiers) + "+" if modifiers else "") + key
    return f"Pressed {display} in {app_name}."


# Windows backend — pywinauto


_WIN_ROLE_MAP = {
    "button": "Button", "btn": "Button",
    "edit": "Edit", "text field": "Edit", "input": "Edit",
    "text": "Text", "label": "Text",
    "checkbox": "CheckBox", "check box": "CheckBox",
    "combobox": "ComboBox", "combo box": "ComboBox",
    "listitem": "ListItem", "list item": "ListItem",
    "list": "List",
    "tab": "TabItem", "tab item": "TabItem",
    "tree item": "TreeItem",
    "menu item": "MenuItem",
    "toolbar": "ToolBar",
}


def _win_connect(app_name: str):
    try:
        from pywinauto import Application
    except ImportError as exc:
        raise RuntimeError(
            "pywinauto is required for UI automation on Windows: pip install pywinauto"
        ) from exc
    for backend in ("uia", "win32"):
        for kwargs in (
            {"path": app_name},
            {"title_re": f".*{app_name}.*"},
            {"title": app_name},
        ):
            try:
                return Application(backend=backend).connect(**kwargs, timeout=3)
            except Exception:
                pass
    raise RuntimeError(
        f"Could not connect to '{app_name}' on Windows. "
        "Make sure the app is running."
    )


def _windows_dispatch(params: "UITool.Params") -> str:
    if params.action == "get_tree":
        return _win_get_tree(params.app, params.window_index)
    if params.action == "click":
        return _win_click(params.app, params.title, params.role, params.window_index)
    if params.action == "click_menu":
        return _win_click_menu(params.app, params.menu, params.menu_item)
    if params.action == "type":
        return _win_type(params.app, params.text)
    if params.action == "press_key":
        return _win_press_key(params.app, params.key, params.modifiers)
    if params.action == "get_value":
        return _win_get_value(params.app, params.title, params.role, params.window_index)
    raise ValueError(f"Unknown action: {params.action!r}")


def _win_get_tree(app_name: str, window_index: int = 1) -> str:
    app = _win_connect(app_name)
    dlg = app.top_window()
    lines = [f"Window: {dlg.window_text()}"]
    seen = 0
    for ctrl in dlg.descendants():
        if seen >= 120:
            lines.append("  … (truncated)")
            break
        try:
            title = ctrl.window_text().strip()
            role = ctrl.friendly_class_name()
            if title:
                lines.append(f"  [{role}] {title}")
                seen += 1
        except Exception:
            pass
    return "\n".join(lines) if len(lines) > 1 else (
        f"No accessible elements in '{app_name}'. Fall back to mouse tool."
    )


def _win_click(
    app_name: str, title: str | None, role: str | None, window_index: int = 1
) -> str:
    app = _win_connect(app_name)
    dlg = app.top_window()
    win_role = _WIN_ROLE_MAP.get((role or "").lower())
    try:
        if title and win_role:
            ctrl = dlg.child_window(title=title, control_type=win_role)
        elif title:
            ctrl = dlg.child_window(title=title)
        elif win_role:
            ctrl = dlg.child_window(control_type=win_role)
        else:
            raise ValueError("Provide 'title' or 'role' for click.")
        ctrl.click_input()
        return f"Clicked '{title or role}' in {app_name}."
    except Exception as exc:
        raise RuntimeError(
            f"Could not click '{title or role}' in '{app_name}'. "
            f"Run get_tree to see elements. Error: {exc}"
        ) from exc


def _win_click_menu(app_name: str, menu: str | None, menu_item: str | None) -> str:
    if not menu or not menu_item:
        raise ValueError("Both 'menu' and 'menu_item' are required.")
    app = _win_connect(app_name)
    try:
        app.top_window().menu_select(f"{menu}->{menu_item}")
        return f"Clicked {menu} → {menu_item} in {app_name}."
    except Exception as exc:
        raise RuntimeError(f"Menu click failed: {exc}") from exc


def _win_type(app_name: str, text: str | None) -> str:
    if not text:
        raise ValueError("'text' is required for action='type'.")
    app = _win_connect(app_name)
    app.top_window().set_focus()
    from openvibe.computer.input import keyboard_type
    return keyboard_type(text)


_WIN_MOD_PREFIX = {
    "command": "^", "cmd": "^",
    "ctrl": "^", "control": "^",
    "shift": "+",
    "alt": "%", "option": "%",
}
_WIN_KEY_MAP = {
    "return": "{ENTER}", "enter": "{ENTER}",
    "escape": "{ESC}", "esc": "{ESC}",
    "tab": "{TAB}", "space": " ",
    "delete": "{DELETE}", "backspace": "{BACKSPACE}",
    "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}",
    "home": "{HOME}", "end": "{END}",
    "pageup": "{PGUP}", "pagedown": "{PGDN}",
    **{f"f{i}": f"{{F{i}}}" for i in range(1, 13)},
}


def _win_press_key(app_name: str, key: str | None, modifiers: list[str]) -> str:
    if not key:
        raise ValueError("'key' is required for action='press_key'.")
    app = _win_connect(app_name)
    prefix = "".join(_WIN_MOD_PREFIX.get(m.lower(), "") for m in modifiers)
    key_str = _WIN_KEY_MAP.get(key.lower(), key if len(key) == 1 else f"{{{key.upper()}}}")
    app.top_window().type_keys(f"{prefix}{key_str}")
    display = ("+".join(modifiers) + "+" if modifiers else "") + key
    return f"Pressed {display} in {app_name}."


def _win_get_value(
    app_name: str, title: str | None, role: str | None, window_index: int = 1
) -> str:
    if not title and not role:
        raise ValueError("Provide 'title' or 'role' for get_value.")
    app = _win_connect(app_name)
    dlg = app.top_window()
    win_role = _WIN_ROLE_MAP.get((role or "").lower())
    try:
        if title and win_role:
            ctrl = dlg.child_window(title=title, control_type=win_role)
        elif title:
            ctrl = dlg.child_window(title=title)
        else:
            ctrl = dlg.child_window(control_type=win_role)
        return ctrl.window_text()
    except Exception as exc:
        raise RuntimeError(
            f"Could not get value of '{title or role}' in '{app_name}': {exc}"
        ) from exc
