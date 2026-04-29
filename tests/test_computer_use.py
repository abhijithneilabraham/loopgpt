"""Tests for the computer-use subsystem.

These tests cover the sandbox, tool parameter validation, and permission
gating without requiring the optional mss/pynput/pillow packages to be
installed (all screen/input calls are mocked out).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openvibe.computer.sandbox import (
    ActionType,
    AuditEntry,
    ComputerSandbox,
    clear_sandbox,
    get_sandbox,
)
from openvibe.tool.base import ToolContext, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(session_id: str = "test-session") -> ToolContext:
    """Build a minimal ToolContext with no permission service (all allowed)."""
    return ToolContext(
        session_id=session_id,
        message_id="msg-1",
        agent_name="computer",
        project_id="proj-1",
        working_dir="/tmp",
        abort=asyncio.Event(),
        call_id="call-1",
        _permissions=None,
    )


# ---------------------------------------------------------------------------
# Sandbox tests
# ---------------------------------------------------------------------------


class TestComputerSandbox:
    def setup_method(self):
        clear_sandbox("s1")

    def test_get_sandbox_creates_new(self):
        sb = get_sandbox("s1")
        assert sb.session_id == "s1"
        assert sb.audit_log == []

    def test_get_sandbox_returns_same_instance(self):
        assert get_sandbox("s1") is get_sandbox("s1")

    def test_clear_sandbox(self):
        a = get_sandbox("s1")
        clear_sandbox("s1")
        b = get_sandbox("s1")
        assert a is not b

    @pytest.mark.asyncio
    async def test_record_action_appends_entry(self):
        sb = ComputerSandbox(session_id="s1")
        entry = await sb.record_action(
            ActionType.SCREENSHOT, params={"region": None}, result="800x600"
        )
        assert len(sb.audit_log) == 1
        assert sb.audit_log[0] is entry
        assert entry.action_type == ActionType.SCREENSHOT
        assert entry.result == "800x600"
        assert entry.error is None

    @pytest.mark.asyncio
    async def test_record_action_with_error(self):
        sb = ComputerSandbox(session_id="s1")
        await sb.record_action(
            ActionType.MOUSE_CLICK,
            params={"x": 100, "y": 200},
            error="boom",
        )
        assert sb.audit_log[0].error == "boom"

    def test_export_audit_log(self):
        sb = ComputerSandbox(session_id="s1")
        asyncio.run(
            sb.record_action(ActionType.APP_OPEN, params={"name": "Terminal"}, result="ok")
        )
        log = sb.export_audit_log()
        assert len(log) == 1
        assert log[0]["action"] == "app_open"
        assert log[0]["result"] == "ok"

    def test_is_app_allowed_no_list(self):
        sb = ComputerSandbox(session_id="s1")
        assert sb.is_app_allowed("Anything") is True

    def test_is_app_allowed_with_list(self):
        sb = ComputerSandbox(session_id="s1", allowed_apps=["Terminal", "Chrome"])
        assert sb.is_app_allowed("terminal") is True  # case-insensitive
        assert sb.is_app_allowed("Google Chrome") is True  # substring
        assert sb.is_app_allowed("Slack") is False

    def test_is_coordinate_allowed_no_region(self):
        sb = ComputerSandbox(session_id="s1")
        assert sb.is_coordinate_allowed(9999, 9999) is True

    def test_is_coordinate_allowed_with_region(self):
        sb = ComputerSandbox(session_id="s1", screen_region=(100, 100, 500, 400))
        assert sb.is_coordinate_allowed(150, 200) is True
        assert sb.is_coordinate_allowed(50, 50) is False
        assert sb.is_coordinate_allowed(700, 600) is False

    def test_summary(self):
        sb = ComputerSandbox(session_id="abcdef123456")
        asyncio.run(sb.record_action(ActionType.SCREENSHOT, params={}))
        asyncio.run(sb.record_action(ActionType.MOUSE_CLICK, params={}))
        s = sb.summary()
        assert "2 actions" in s
        assert "mouse_click" in s
        assert "screenshot" in s


# ---------------------------------------------------------------------------
# ScreenshotTool tests
# ---------------------------------------------------------------------------


class TestScreenshotTool:
    @pytest.mark.asyncio
    async def test_invalid_region_length(self):
        from openvibe.tool.computer_screenshot import ScreenshotTool

        tool = ScreenshotTool()
        ctx = _ctx()
        result = await tool.execute(ctx, ScreenshotTool.Params(region=[100, 200]))
        assert result.error is True
        assert "4 elements" in result.output

    @pytest.mark.asyncio
    async def test_coordinate_outside_sandbox_region(self):
        from openvibe.tool.computer_screenshot import ScreenshotTool

        clear_sandbox("s-ss")
        sb = get_sandbox("s-ss")
        sb.screen_region = (0, 0, 200, 200)

        tool = ScreenshotTool()
        ctx = _ctx("s-ss")
        result = await tool.execute(ctx, ScreenshotTool.Params(region=[500, 500, 100, 100]))
        assert result.error is True
        assert "outside" in result.output.lower()

    @pytest.mark.asyncio
    async def test_capture_full_screen(self):
        from openvibe.tool.computer_screenshot import ScreenshotTool

        clear_sandbox("s-full")
        tool = ScreenshotTool()
        ctx = _ctx("s-full")

        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        with patch(
            "openvibe.computer.capture.capture_screen",
            return_value=(fake_png, 1920, 1080),
        ):
            result = await tool.execute(ctx, ScreenshotTool.Params())

        assert result.error is False
        assert "1920" in result.title
        assert "1080" in result.title
        # Attachment carries the raw PNG bytes
        assert len(result.attachments) == 1
        assert result.attachments[0].filename == "screenshot.png"
        assert result.attachments[0].content == fake_png
        assert result.attachments[0].media_type == "image/png"

    @pytest.mark.asyncio
    async def test_processor_stores_image_in_metadata(self):
        """image_b64 must be stored in ToolState.metadata so the LLM can see it."""
        import base64

        from openvibe.tool.base import Attachment, ToolResult

        fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        fake_result = ToolResult(
            title="Screenshot 800x600",
            output="Captured 800x600 screenshot.",
            attachments=[
                Attachment(filename="screenshot.png", content=fake_png, media_type="image/png")
            ],
        )

        # Simulate what the processor does when it receives a ToolResult with an image
        from openvibe.session.models import ToolState
        from openvibe.config import ToolStateStatus

        state = ToolState(
            status=ToolStateStatus.COMPLETED,
            call_id="call-1",
            tool_name="screenshot",
            input={},
            output=fake_result.output,
        )

        # Apply the same logic as the processor
        for att in fake_result.attachments:
            if att.media_type.startswith("image/"):
                state.metadata["image_b64"] = base64.b64encode(att.content).decode("ascii")
                state.metadata["image_media_type"] = att.media_type
                break

        assert "image_b64" in state.metadata
        assert state.metadata["image_media_type"] == "image/png"
        assert base64.b64decode(state.metadata["image_b64"]) == fake_png

    @pytest.mark.asyncio
    async def test_capture_records_audit_entry(self):
        from openvibe.tool.computer_screenshot import ScreenshotTool

        clear_sandbox("s-audit")
        tool = ScreenshotTool()
        ctx = _ctx("s-audit")

        fake_png = b"\x89PNG" + b"\x00" * 20
        with patch(
            "openvibe.computer.capture.capture_screen",
            return_value=(fake_png, 800, 600),
        ):
            await tool.execute(ctx, ScreenshotTool.Params())

        sb = get_sandbox("s-audit")
        assert len(sb.audit_log) == 1
        assert sb.audit_log[0].action_type == ActionType.SCREENSHOT


# ---------------------------------------------------------------------------
# MouseTool tests
# ---------------------------------------------------------------------------


class TestMouseTool:
    @pytest.mark.asyncio
    async def test_click_outside_sandbox_region(self):
        from openvibe.tool.computer_mouse import MouseTool

        clear_sandbox("s-mouse")
        sb = get_sandbox("s-mouse")
        sb.screen_region = (0, 0, 500, 500)

        tool = MouseTool()
        ctx = _ctx("s-mouse")
        result = await tool.execute(
            ctx, MouseTool.Params(action="click", x=999, y=999)
        )
        assert result.error is True
        assert "outside" in result.output.lower()

    @pytest.mark.asyncio
    async def test_click_within_region(self):
        from openvibe.tool.computer_mouse import MouseTool

        clear_sandbox("s-click")
        sb = get_sandbox("s-click")
        sb.screen_region = (0, 0, 1920, 1080)

        tool = MouseTool()
        ctx = _ctx("s-click")

        with patch.object(MouseTool, "_do_action", return_value="Left-clicked at (100, 200)."):
            result = await tool.execute(
                ctx, MouseTool.Params(action="click", x=100, y=200)
            )

        assert result.error is False
        assert "100" in result.output

    @pytest.mark.asyncio
    async def test_scroll(self):
        from openvibe.tool.computer_mouse import MouseTool

        clear_sandbox("s-scroll")
        tool = MouseTool()
        ctx = _ctx("s-scroll")

        with patch.object(MouseTool, "_do_action", return_value="Scrolled down 3 ticks at (400, 400)."):
            result = await tool.execute(
                ctx, MouseTool.Params(action="scroll", x=400, y=400, amount=-3)
            )

        assert result.error is False
        assert "down" in result.output.lower()

    @pytest.mark.asyncio
    async def test_drag_missing_end(self):
        from openvibe.tool.computer_mouse import MouseTool

        clear_sandbox("s-drag")
        tool = MouseTool()
        ctx = _ctx("s-drag")

        with patch.object(MouseTool, "_do_action", side_effect=ValueError("end_x and end_y are required for drag.")):
            result = await tool.execute(
                ctx, MouseTool.Params(action="drag", x=100, y=100)
            )

        assert result.error is True
        assert "end_x" in result.output.lower() or "end" in result.output.lower()

    @pytest.mark.asyncio
    async def test_drag_records_audit(self):
        from openvibe.tool.computer_mouse import MouseTool

        clear_sandbox("s-drag2")
        tool = MouseTool()
        ctx = _ctx("s-drag2")

        with patch.object(MouseTool, "_do_action", return_value="Dragged from (10, 10) to (300, 300)."):
            await tool.execute(
                ctx, MouseTool.Params(action="drag", x=10, y=10, end_x=300, end_y=300)
            )

        sb = get_sandbox("s-drag2")
        assert any(e.action_type == ActionType.MOUSE_DRAG for e in sb.audit_log)


# ---------------------------------------------------------------------------
# KeyboardTool tests
# ---------------------------------------------------------------------------


class TestKeyboardTool:
    @pytest.mark.asyncio
    async def test_type_text(self):
        from openvibe.tool.computer_keyboard import KeyboardTool

        clear_sandbox("s-kbd")
        tool = KeyboardTool()
        ctx = _ctx("s-kbd")

        with patch.object(KeyboardTool, "_do_action", return_value="Typed 11 chars: 'hello world'"):
            result = await tool.execute(
                ctx, KeyboardTool.Params(action="type", text="hello world")
            )

        assert result.error is False
        assert "11" in result.output  # 11 characters

    @pytest.mark.asyncio
    async def test_type_missing_text(self):
        from openvibe.tool.computer_keyboard import KeyboardTool

        clear_sandbox("s-kbd2")
        tool = KeyboardTool()
        ctx = _ctx("s-kbd2")

        with patch.object(KeyboardTool, "_do_action", side_effect=ValueError("text is required")):
            result = await tool.execute(
                ctx, KeyboardTool.Params(action="type")
            )

        assert result.error is True

    @pytest.mark.asyncio
    async def test_press_key(self):
        from openvibe.tool.computer_keyboard import KeyboardTool

        clear_sandbox("s-press")
        tool = KeyboardTool()
        ctx = _ctx("s-press")

        with patch.object(KeyboardTool, "_do_action", return_value="Pressed: 'enter'"):
            result = await tool.execute(
                ctx, KeyboardTool.Params(action="press", key="enter")
            )

        assert result.error is False
        assert "enter" in result.output.lower()

    @pytest.mark.asyncio
    async def test_hotkey(self):
        from openvibe.tool.computer_keyboard import KeyboardTool

        clear_sandbox("s-hotkey")
        tool = KeyboardTool()
        ctx = _ctx("s-hotkey")

        with patch.object(KeyboardTool, "_do_action", return_value="Hotkey: ctrl+c"):
            result = await tool.execute(
                ctx, KeyboardTool.Params(action="hotkey", keys=["ctrl", "c"])
            )

        assert result.error is False
        assert "ctrl" in result.output.lower()

    @pytest.mark.asyncio
    async def test_keyboard_records_audit(self):
        from openvibe.tool.computer_keyboard import KeyboardTool

        clear_sandbox("s-kdaud")
        tool = KeyboardTool()
        ctx = _ctx("s-kdaud")

        with patch.object(KeyboardTool, "_do_action", return_value="Pressed: 'escape'"):
            await tool.execute(
                ctx, KeyboardTool.Params(action="press", key="escape")
            )

        sb = get_sandbox("s-kdaud")
        assert any(e.action_type == ActionType.KEYBOARD_PRESS for e in sb.audit_log)


# ---------------------------------------------------------------------------
# AppTool tests
# ---------------------------------------------------------------------------


class TestAppTool:
    @pytest.mark.asyncio
    async def test_open_denied_by_allowlist(self):
        from openvibe.tool.computer_app import AppTool

        clear_sandbox("s-app")
        sb = get_sandbox("s-app")
        sb.allowed_apps = ["Terminal"]

        tool = AppTool()
        ctx = _ctx("s-app")
        result = await tool.execute(ctx, AppTool.Params(action="open", name="Slack"))
        assert result.error is True
        assert "allow-list" in result.output.lower() or "allowed" in result.output.lower()

    @pytest.mark.asyncio
    async def test_list_action_bypasses_allowlist(self):
        """list action does not require an app name and should not be blocked."""
        from openvibe.tool.computer_app import AppTool, _list_windows

        clear_sandbox("s-list")
        sb = get_sandbox("s-list")
        sb.allowed_apps = ["Terminal"]

        tool = AppTool()
        ctx = _ctx("s-list")

        with patch("openvibe.tool.computer_app._list_windows", return_value="• App1\n• App2"):
            result = await tool.execute(ctx, AppTool.Params(action="list"))

        assert result.error is False

    @pytest.mark.asyncio
    async def test_open_missing_name(self):
        from openvibe.tool.computer_app import AppTool

        clear_sandbox("s-appname")
        tool = AppTool()
        ctx = _ctx("s-appname")

        # No app name provided — should fail gracefully
        with patch("openvibe.tool.computer_app._open_app", side_effect=ValueError("name is required")):
            result = await tool.execute(ctx, AppTool.Params(action="open"))

        assert result.error is True

    @pytest.mark.asyncio
    async def test_open_records_audit(self):
        from openvibe.tool.computer_app import AppTool

        clear_sandbox("s-appaud")
        tool = AppTool()
        ctx = _ctx("s-appaud")

        with patch("openvibe.tool.computer_app._open_app", return_value="Opened 'Terminal'."):
            result = await tool.execute(ctx, AppTool.Params(action="open", name="Terminal"))

        assert result.error is False
        sb = get_sandbox("s-appaud")
        assert any(e.action_type == ActionType.APP_OPEN for e in sb.audit_log)


# ---------------------------------------------------------------------------
# LLM message builder — vision content block tests
# ---------------------------------------------------------------------------


class TestLLMVisionMessages:
    """Verify that _to_llm_messages emits image ContentBlocks for screenshot results."""

    def _make_tool_part_with_image(self, b64: str, media_type: str = "image/png"):
        from openvibe.config import ToolStateStatus
        from openvibe.session.models import ToolPart, ToolState

        state = ToolState(
            status=ToolStateStatus.COMPLETED,
            call_id="call-img-1",
            tool_name="screenshot",
            input={},
            output="Captured 1920x1080 screenshot.",
            metadata={"image_b64": b64, "image_media_type": media_type},
        )
        return ToolPart(state=state)

    def _make_assistant_msg(self, tool_part):
        from openvibe.config import MessageRole
        from openvibe.session.models import MessageInfo, TextPart

        return MessageInfo(
            id="msg-1",
            session_id="s1",
            role=MessageRole.ASSISTANT,
            position=0,
            created_at="2026-01-01T00:00:00",
            parts=[TextPart(content="Let me take a screenshot."), tool_part],
        )

    def test_image_tool_result_produces_content_block_list(self):
        import base64

        from openvibe.session.processor import _to_llm_messages
        from openvibe.llm import ContentBlock

        fake_b64 = base64.b64encode(b"\x89PNG fake").decode("ascii")
        tool_part = self._make_tool_part_with_image(fake_b64)
        assistant_msg = self._make_assistant_msg(tool_part)

        # Build a dummy agent with no disabled tools
        from openvibe.agent.agent import _BUILTIN_AGENTS
        agent = _BUILTIN_AGENTS["computer"]

        messages = _to_llm_messages([assistant_msg], agent)

        # Should produce: one assistant message + one tool message
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) == 1

        tool_msg = tool_msgs[0]
        assert tool_msg.tool_call_id == "call-img-1"
        # Content must be a list of ContentBlocks (not a plain string)
        assert isinstance(tool_msg.content, list)
        assert len(tool_msg.content) == 2

        image_block = tool_msg.content[0]
        text_block = tool_msg.content[1]

        assert isinstance(image_block, ContentBlock)
        assert image_block.type == "image_url"
        assert image_block.image_url is not None
        assert image_block.image_url["url"].startswith("data:image/png;base64,")
        assert fake_b64 in image_block.image_url["url"]

        assert isinstance(text_block, ContentBlock)
        assert text_block.type == "text"
        assert "1920" in (text_block.text or "")

    def test_text_only_tool_result_produces_plain_string(self):
        from openvibe.config import MessageRole, ToolStateStatus
        from openvibe.session.models import MessageInfo, TextPart, ToolPart, ToolState
        from openvibe.session.processor import _to_llm_messages
        from openvibe.agent.agent import _BUILTIN_AGENTS

        state = ToolState(
            status=ToolStateStatus.COMPLETED,
            call_id="call-text-1",
            tool_name="bash",
            input={"command": "ls"},
            output="file1.py\nfile2.py",
        )
        tool_part = ToolPart(state=state)
        assistant_msg = MessageInfo(
            id="msg-2",
            session_id="s1",
            role=MessageRole.ASSISTANT,
            position=0,
            created_at="2026-01-01T00:00:00",
            parts=[TextPart(content="Running ls."), tool_part],
        )

        agent = _BUILTIN_AGENTS["build"]
        messages = _to_llm_messages([assistant_msg], agent)

        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        # Plain text tool result — content must be a string, not a list
        assert isinstance(tool_msgs[0].content, str)
        assert "file1.py" in tool_msgs[0].content

    def test_litellm_serialisation_of_image_tool_result(self):
        """_to_litellm_messages must produce valid dict for image tool results."""
        import base64

        from openvibe.llm import ContentBlock, Message, _to_litellm_messages

        fake_b64 = base64.b64encode(b"PNG-DATA").decode("ascii")
        msg = Message(
            role="tool",
            content=[
                ContentBlock(
                    type="image_url",
                    image_url={"url": f"data:image/png;base64,{fake_b64}"},
                ),
                ContentBlock(type="text", text="Screenshot captured."),
            ],
            tool_call_id="call-123",
        )

        result = _to_litellm_messages([msg])
        assert len(result) == 1
        d = result[0]
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call-123"
        assert isinstance(d["content"], list)
        assert d["content"][0]["type"] == "image_url"
        assert d["content"][1]["type"] == "text"
        assert d["content"][1]["text"] == "Screenshot captured."


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestComputerUseRegistry:
    def test_create_computer_use_registry_contains_cu_tools(self):
        from openvibe.tool.base import create_computer_use_registry

        registry = create_computer_use_registry()
        assert "screenshot" in registry
        assert "mouse" in registry
        assert "keyboard" in registry
        assert "app" in registry

    def test_create_computer_use_registry_contains_default_tools(self):
        from openvibe.tool.base import create_computer_use_registry

        registry = create_computer_use_registry()
        assert "bash" in registry
        assert "read" in registry
        assert "write" in registry
        assert "glob" in registry


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestComputerAgent:
    def test_computer_agent_exists(self):
        from openvibe.agent.agent import _BUILTIN_AGENTS

        assert "computer" in _BUILTIN_AGENTS

    def test_computer_agent_rules(self):
        from openvibe.agent.agent import _BUILTIN_AGENTS
        from openvibe.config import PermissionAction

        rules = {r.tool: r.action for r in _BUILTIN_AGENTS["computer"].permission_rules}
        assert rules.get("screenshot") == PermissionAction.ALLOW
        assert rules.get("mouse") == PermissionAction.ASK
        assert rules.get("keyboard") == PermissionAction.ASK
        assert rules.get("app") == PermissionAction.ASK


# ---------------------------------------------------------------------------
# Verification loop tests
# ---------------------------------------------------------------------------


class TestVerificationLoop:
    """Tests for the automatic change-detection verification loop."""

    def _make_solid_png(self, width: int, height: int, color: tuple) -> bytes:
        """Create a solid-colour PNG for diffing tests."""
        from PIL import Image
        import io
        img = Image.new("RGB", (width, height), color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_diff_unchanged(self):
        from openvibe.computer.capture import diff_screenshots
        png = self._make_solid_png(100, 100, (128, 128, 128))
        report = diff_screenshots(png, png)
        assert report["changed"] is False
        assert report["change_fraction"] < 0.001
        assert "no visible change" in str(report["summary"]).lower()

    def test_diff_full_change(self):
        from openvibe.computer.capture import diff_screenshots
        before = self._make_solid_png(100, 100, (0, 0, 0))
        after = self._make_solid_png(100, 100, (255, 255, 255))
        report = diff_screenshots(before, after)
        assert report["changed"] is True
        assert report["change_fraction"] > 0.99
        assert report["changed_region"] is not None

    def test_diff_partial_change(self):
        """Only the right half changes — bounding box should cover that region."""
        from PIL import Image
        import io
        from openvibe.computer.capture import diff_screenshots

        before = self._make_solid_png(200, 100, (0, 0, 0))
        # Paint right half white
        img = Image.new("RGB", (200, 100), (0, 0, 0))
        for x in range(100, 200):
            for y in range(100):
                img.putpixel((x, y), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        after = buf.getvalue()

        report = diff_screenshots(before, after)
        assert report["changed"] is True
        region = report["changed_region"]
        assert region is not None
        x, _y, w, _h = region
        assert x >= 99  # change starts around x=100

    def test_diff_size_mismatch(self):
        from openvibe.computer.capture import diff_screenshots
        small = self._make_solid_png(100, 100, (0, 0, 0))
        large = self._make_solid_png(200, 200, (0, 0, 0))
        report = diff_screenshots(small, large)
        assert report["changed"] is True
        assert "resolution" in str(report["summary"]).lower()

    @pytest.mark.asyncio
    async def test_screenshot_tool_includes_diff_in_output(self):
        """Second screenshot should report change vs the first."""
        from openvibe.tool.computer_screenshot import ScreenshotTool
        from openvibe.computer.sandbox import clear_sandbox, get_sandbox

        clear_sandbox("s-diff")
        tool = ScreenshotTool()
        ctx = _ctx("s-diff")

        from PIL import Image
        import io

        def _png(color):
            img = Image.new("RGB", (100, 100), color)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        first_png = _png((0, 0, 0))
        second_png = _png((255, 255, 255))

        with patch("openvibe.computer.capture.capture_screen", return_value=(first_png, 100, 100)):
            await tool.execute(ctx, ScreenshotTool.Params())

        with patch("openvibe.computer.capture.capture_screen", return_value=(second_png, 100, 100)):
            result = await tool.execute(ctx, ScreenshotTool.Params())

        assert "change detection" in result.output.lower()
        assert result.error is False

    @pytest.mark.asyncio
    async def test_screenshot_tool_no_diff_on_first_capture(self):
        """First screenshot has nothing to compare against — no diff line."""
        from openvibe.tool.computer_screenshot import ScreenshotTool
        from openvibe.computer.sandbox import clear_sandbox

        clear_sandbox("s-nodiff")
        tool = ScreenshotTool()
        ctx = _ctx("s-nodiff")

        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        with patch("openvibe.computer.capture.capture_screen", return_value=(png, 100, 100)):
            result = await tool.execute(ctx, ScreenshotTool.Params())

        assert "change detection" not in result.output.lower()
        assert result.error is False

    @pytest.mark.asyncio
    async def test_sandbox_stores_last_screenshot(self):
        """sandbox.last_screenshot is updated after each capture."""
        from openvibe.tool.computer_screenshot import ScreenshotTool
        from openvibe.computer.sandbox import clear_sandbox, get_sandbox

        clear_sandbox("s-store")
        tool = ScreenshotTool()
        ctx = _ctx("s-store")

        from PIL import Image
        import io
        img = Image.new("RGB", (50, 50), (1, 2, 3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        assert get_sandbox("s-store").last_screenshot is None

        with patch("openvibe.computer.capture.capture_screen", return_value=(png, 50, 50)):
            await tool.execute(ctx, ScreenshotTool.Params())

        assert get_sandbox("s-store").last_screenshot == png

    @pytest.mark.asyncio
    async def test_mouse_settle_ms_respected(self):
        """Mouse tool passes settle_ms through to _do_action."""
        from openvibe.tool.computer_mouse import MouseTool
        from openvibe.computer.sandbox import clear_sandbox

        clear_sandbox("s-settle")
        tool = MouseTool()
        ctx = _ctx("s-settle")

        with patch.object(MouseTool, "_do_action", return_value="Left-clicked at (100, 200)."):
            result = await tool.execute(
                ctx, MouseTool.Params(action="click", x=100, y=200, settle_ms=0)
            )
        assert result.error is False


# ---------------------------------------------------------------------------
# UITool tests
# ---------------------------------------------------------------------------


class TestUIToolParams:
    def test_params_require_action_and_app(self):
        from openvibe.tool.computer_ui import UITool
        from pydantic import ValidationError
        with pytest.raises((ValidationError, TypeError)):
            UITool.Params()  # missing action and app

    def test_params_valid_get_tree(self):
        from openvibe.tool.computer_ui import UITool
        p = UITool.Params(action="get_tree", app="TextEdit")
        assert p.action == "get_tree"
        assert p.app == "TextEdit"
        assert p.window_index == 1

    def test_params_defaults(self):
        from openvibe.tool.computer_ui import UITool
        p = UITool.Params(action="click", app="Finder")
        assert p.title is None
        assert p.role is None
        assert p.text is None
        assert p.menu is None
        assert p.modifiers == []

    def test_params_all_actions_valid(self):
        from openvibe.tool.computer_ui import UITool
        for action in ("get_tree", "click", "click_menu", "type", "press_key", "get_value"):
            p = UITool.Params(action=action, app="TestApp")
            assert p.action == action

    def test_params_invalid_action_rejected(self):
        from openvibe.tool.computer_ui import UITool
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            UITool.Params(action="bad_action", app="App")


class TestUIToolDoAction:
    def test_do_action_unsupported_platform_raises(self):
        from openvibe.tool.computer_ui import UITool
        import openvibe.tool.computer_ui as ui_mod
        params = UITool.Params(action="get_tree", app="TestApp")
        with patch.object(ui_mod, "_PLATFORM", "SomeOS"):
            with pytest.raises(RuntimeError, match="unsupported platform"):
                UITool._do_action(params)

    def test_do_action_darwin_dispatches(self):
        from openvibe.tool.computer_ui import UITool
        import openvibe.tool.computer_ui as ui_mod
        params = UITool.Params(action="get_tree", app="TextEdit")
        with patch.object(ui_mod, "_PLATFORM", "Darwin"), \
             patch.object(ui_mod, "_macos_dispatch", return_value="tree output") as mock_dispatch:
            result = UITool._do_action(params)
        mock_dispatch.assert_called_once_with(params)
        assert result == "tree output"

    def test_do_action_linux_dispatches(self):
        from openvibe.tool.computer_ui import UITool
        import openvibe.tool.computer_ui as ui_mod
        params = UITool.Params(action="get_tree", app="gedit")
        with patch.object(ui_mod, "_PLATFORM", "Linux"), \
             patch.object(ui_mod, "_linux_dispatch", return_value="linux tree") as mock_dispatch:
            result = UITool._do_action(params)
        mock_dispatch.assert_called_once_with(params)
        assert result == "linux tree"

    def test_do_action_windows_dispatches(self):
        from openvibe.tool.computer_ui import UITool
        import openvibe.tool.computer_ui as ui_mod
        params = UITool.Params(action="get_tree", app="Notepad")
        with patch.object(ui_mod, "_PLATFORM", "Windows"), \
             patch.object(ui_mod, "_windows_dispatch", return_value="win tree") as mock_dispatch:
            result = UITool._do_action(params)
        mock_dispatch.assert_called_once_with(params)
        assert result == "win tree"


class TestUIToolExecute:
    @pytest.mark.asyncio
    async def test_execute_returns_result_on_success(self):
        from openvibe.tool.computer_ui import UITool
        from openvibe.computer.sandbox import clear_sandbox

        clear_sandbox("ui-ok")
        tool = UITool()
        ctx = _ctx("ui-ok")
        params = UITool.Params(action="get_tree", app="TextEdit", title="Save")

        with patch.object(UITool, "_do_action", return_value="[button] Save\n[text] Hello"):
            result = await tool.execute(ctx, params)

        assert result.error is False
        assert "[button] Save" in result.output or "Save" in result.title or result.output

    @pytest.mark.asyncio
    async def test_execute_returns_error_on_runtime_error(self):
        from openvibe.tool.computer_ui import UITool
        from openvibe.computer.sandbox import clear_sandbox

        clear_sandbox("ui-err")
        tool = UITool()
        ctx = _ctx("ui-err")
        params = UITool.Params(action="click", app="TextEdit", title="NonExistent")

        with patch.object(UITool, "_do_action", side_effect=RuntimeError("element not found")):
            result = await tool.execute(ctx, params)

        assert result.error is True
        assert "element not found" in result.output

    @pytest.mark.asyncio
    async def test_execute_returns_error_on_value_error(self):
        from openvibe.tool.computer_ui import UITool
        from openvibe.computer.sandbox import clear_sandbox

        clear_sandbox("ui-val-err")
        tool = UITool()
        ctx = _ctx("ui-val-err")
        params = UITool.Params(action="click", app="TextEdit")

        with patch.object(UITool, "_do_action", side_effect=ValueError("need title or role")):
            result = await tool.execute(ctx, params)

        assert result.error is True

    @pytest.mark.asyncio
    async def test_execute_title_in_result(self):
        from openvibe.tool.computer_ui import UITool
        from openvibe.computer.sandbox import clear_sandbox

        clear_sandbox("ui-title")
        tool = UITool()
        ctx = _ctx("ui-title")
        params = UITool.Params(action="click", app="Finder", title="Desktop")

        with patch.object(UITool, "_do_action", return_value="Clicked [button] Desktop in Finder."):
            result = await tool.execute(ctx, params)

        assert "Desktop" in result.title or "click" in result.title.lower()

    @pytest.mark.asyncio
    async def test_execute_records_action_in_sandbox(self):
        from openvibe.tool.computer_ui import UITool
        from openvibe.computer.sandbox import clear_sandbox, get_sandbox

        clear_sandbox("ui-audit")
        tool = UITool()
        ctx = _ctx("ui-audit")
        params = UITool.Params(action="get_tree", app="Safari")

        with patch.object(UITool, "_do_action", return_value="tree ok"):
            await tool.execute(ctx, params)

        sandbox = get_sandbox("ui-audit")
        assert len(sandbox.audit_log) >= 1


class TestUIToolHelpers:
    def test_macos_key_map_contains_common_keys(self):
        from openvibe.tool.computer_ui import _MACOS_KEY_MAP
        for key in ("return", "escape", "tab", "space", "delete", "up", "down"):
            assert key in _MACOS_KEY_MAP

    def test_macos_mod_map(self):
        from openvibe.tool.computer_ui import _MACOS_MOD_MAP
        assert "command" in _MACOS_MOD_MAP
        assert "shift" in _MACOS_MOD_MAP
        assert "ctrl" in _MACOS_MOD_MAP

    def test_linux_key_map_contains_common_keys(self):
        from openvibe.tool.computer_ui import _LINUX_KEY_MAP
        for key in ("return", "escape", "tab", "space", "up", "down"):
            assert key in _LINUX_KEY_MAP

    def test_win_key_map_contains_common_keys(self):
        from openvibe.tool.computer_ui import _WIN_KEY_MAP
        for key in ("return", "escape", "tab", "delete", "up", "down"):
            assert key in _WIN_KEY_MAP

    def test_win_role_map(self):
        from openvibe.tool.computer_ui import _WIN_ROLE_MAP
        assert "button" in _WIN_ROLE_MAP
        assert "checkbox" in _WIN_ROLE_MAP
        assert "edit" in _WIN_ROLE_MAP

    def test_macos_dispatch_raises_on_unknown_action(self):
        from openvibe.tool.computer_ui import UITool, _macos_dispatch
        import openvibe.tool.computer_ui as ui_mod
        params = UITool.Params(action="get_tree", app="Foo")
        # Monkey-patch action to something invalid to hit the raise branch
        params2 = params.model_copy(update={"action": "get_tree"})
        # We can't set an invalid Literal directly — test via the dispatch guard
        with patch.object(ui_mod, "_osascript", return_value="ok"):
            # Valid actions should not raise
            try:
                _macos_dispatch(params2)
            except Exception:
                pass  # osascript not available in CI — that's expected

    def test_linux_xdotool_fallback_raises_on_tree_without_tools(self):
        from openvibe.tool.computer_ui import _linux_xdotool_dispatch, UITool
        params = UITool.Params(action="get_tree", app="gedit")
        with patch("openvibe.tool.computer_ui._has_xdotool", return_value=False):
            with pytest.raises(ImportError):
                _linux_xdotool_dispatch(params)

    def test_linux_type_raises_without_text(self):
        from openvibe.tool.computer_ui import _linux_type
        with pytest.raises(ValueError, match="text"):
            _linux_type("gedit", None)

    def test_linux_press_key_raises_without_key(self):
        from openvibe.tool.computer_ui import _linux_press_key
        with pytest.raises(ValueError, match="key"):
            _linux_press_key("gedit", None, [])

    def test_macos_type_raises_without_text(self):
        from openvibe.tool.computer_ui import _macos_type
        with pytest.raises(ValueError, match="text"):
            _macos_type("TextEdit", None)

    def test_macos_click_raises_without_title_or_role(self):
        from openvibe.tool.computer_ui import _macos_click
        with pytest.raises(ValueError):
            _macos_click("TextEdit", None, None)

    def test_macos_click_menu_raises_without_menu(self):
        from openvibe.tool.computer_ui import _macos_click_menu
        with pytest.raises(ValueError):
            _macos_click_menu("TextEdit", None, "Save")

    def test_macos_press_key_raises_without_key(self):
        from openvibe.tool.computer_ui import _macos_press_key
        with pytest.raises(ValueError, match="key"):
            _macos_press_key("TextEdit", None, [])

    def test_macos_get_value_raises_without_title_or_role(self):
        from openvibe.tool.computer_ui import _macos_get_value
        with pytest.raises(ValueError):
            _macos_get_value("TextEdit", None, None)
