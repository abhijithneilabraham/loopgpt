"""Tool base class, registry, and built-in tool loader."""

from __future__ import annotations

import abc
import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from openvibe.llm import Message
    from openvibe.permission.permission import PermissionService

MAX_OUTPUT_CHARS = 4_000


# ---------------------------------------------------------------------------
# Context and Result
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Runtime context passed to every tool execution."""

    session_id: str
    message_id: str
    agent_name: str
    project_id: str
    working_dir: str
    abort: asyncio.Event = field(default_factory=asyncio.Event)
    call_id: str = ""
    # Permission service is injected so tools can call ctx.check_permission()
    _permissions: "PermissionService | None" = field(default=None, repr=False)
    _rules: list = field(default_factory=list, repr=False)
    _messages: list["Message"] = field(default_factory=list, repr=False)

    async def check_permission(
        self, tool: str, argument: str | None = None, description: str = ""
    ) -> None:
        """Raise PermissionDenied / PermissionRejected if not allowed."""
        if self._permissions:
            await self._permissions.check(
                tool=tool,
                argument=argument,
                rules=self._rules or None,
                project_id=self.project_id,
                session_id=self.session_id,
                description=description,
            )


@dataclass
class Attachment:
    """Binary or text attachment to include alongside tool output."""

    filename: str
    content: bytes
    media_type: str = "text/plain"


@dataclass
class ToolResult:
    """Structured output returned by a tool."""

    title: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[Attachment] = field(default_factory=list)
    error: bool = False
    # When set, the session processor injects this as a new USER message after
    # storing the tool result, causing the LLM to respond to it as a fresh task.
    follow_up_message: str = ""
    # When set alongside follow_up_message, the processor switches to this agent
    # for the follow-up turn (e.g. "computer" for learn/replay tasks).
    follow_up_agent: str = ""

    def __post_init__(self) -> None:
        # Auto-truncate unless explicitly marked otherwise
        if not self.metadata.get("truncated") and len(self.output) > MAX_OUTPUT_CHARS:
            self.output = self.output[:MAX_OUTPUT_CHARS] + "\n… [truncated]"
            self.metadata["truncated"] = True


# ---------------------------------------------------------------------------
# Tool ABC
# ---------------------------------------------------------------------------


class Tool(abc.ABC):
    """Base class for all openvibe tools. Subclasses set name/description and implement execute()."""

    name: str  # must be set by subclass
    description: str  # must be set by subclass

    class Params(BaseModel):
        """Override in subclasses to define tool parameters."""

        model_config = {"extra": "forbid"}

    def parameters_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for this tool's parameters."""
        return self.Params.model_json_schema()

    @abc.abstractmethod
    async def execute(self, ctx: ToolContext, params: "Tool.Params") -> ToolResult:
        """Execute the tool with *params* in *ctx*. Must be overridden."""
        ...

    def cancel(self) -> None:
        """Cancel a running execution. Override in subclasses that support it."""

    async def __call__(
        self, ctx: ToolContext, raw_args: str | dict[str, Any]
    ) -> ToolResult:
        """Parse *raw_args* and call ``execute``. Handles validation errors."""
        if isinstance(raw_args, str):
            try:
                data = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                return ToolResult(
                    title=f"Error: invalid JSON args for {self.name}",
                    output=str(exc),
                    error=True,
                )
        else:
            data = raw_args

        try:
            params = self.Params.model_validate(data)
        except Exception as exc:
            return ToolResult(
                title=f"Error: bad parameters for {self.name}",
                output=str(exc),
                error=True,
            )

        return await self.execute(ctx, params)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Holds all available tools, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def create_default_registry() -> ToolRegistry:
    """Return a registry pre-loaded with all built-in tools."""
    from openvibe.tool.bash import BashTool
    from openvibe.tool.edit import EditTool
    from openvibe.tool.glob_tool import GlobTool
    from openvibe.tool.grep_tool import GrepTool
    from openvibe.tool.read import ReadTool
    from openvibe.tool.todo import TodoReadTool, TodoWriteTool
    from openvibe.tool.browser import WebBrowserTool
    from openvibe.tool.web_fetch import WebFetchTool
    from openvibe.tool.web_search import WebSearchTool
    from openvibe.tool.write import WriteTool
    from openvibe.tool.computer_app import AppTool
    from openvibe.tool.computer_keyboard import KeyboardTool
    from openvibe.tool.computer_mouse import MouseTool
    from openvibe.tool.computer_preflight import PreFlightTool
    from openvibe.tool.computer_screenshot import ScreenshotTool
    from openvibe.tool.computer_ui import UITool
    from openvibe.tool.computer_watch import WatchScreenTool
    from openvibe.tool.computer_learn import LearnTool

    registry = ToolRegistry()
    for tool in [
        BashTool(),
        ReadTool(),
        WriteTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
        WebSearchTool(),
        WebBrowserTool(),
        WebFetchTool(),
        TodoWriteTool(),
        TodoReadTool(),
        PreFlightTool(),
        ScreenshotTool(),
        UITool(),
        MouseTool(),
        KeyboardTool(),
        AppTool(),
        WatchScreenTool(),
        LearnTool(),
    ]:
        registry.register(tool)
    return registry


def create_computer_use_registry() -> ToolRegistry:
    """Alias for create_default_registry — computer-use tools are now built-in."""
    return create_default_registry()
