"""Tests for the @tool decorator (openvibe.tool.base.tool).

Each test resets the global _user_registry so registrations don't bleed
between tests.
"""

from __future__ import annotations

import asyncio

import pytest

from openvibe.tool.base import (
    Tool,
    ToolContext,
    ToolResult,
    create_default_registry,
    tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_user_registry():
    """Wipe the module-level user registry before and after every test."""
    from openvibe.tool import base as _base

    _base._user_registry = None
    yield
    _base._user_registry = None


@pytest.fixture()
def ctx(tmp_path):
    return ToolContext(
        session_id="s",
        message_id="m",
        agent_name="a",
        project_id="p",
        working_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Decoration — name, description, type
# ---------------------------------------------------------------------------


def test_tool_returns_tool_instance():
    @tool
    def my_tool(x: str) -> str:
        """Does something."""
        return x

    assert isinstance(my_tool, Tool)


def test_tool_name_from_function():
    @tool
    def search_docs(query: str) -> str:
        """Search docs."""
        return query

    assert my_tool_name(search_docs) == "search_docs"


def my_tool_name(t):
    return t.name


def test_tool_description_from_docstring():
    @tool
    def my_fn(x: str) -> str:
        """This is the description."""
        return x

    assert my_fn.description == "This is the description."


def test_tool_no_docstring_uses_function_name():
    @tool
    def undocumented(x: str) -> str:
        return x

    assert undocumented.description == "undocumented"


# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------


def test_schema_required_field():
    @tool
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"

    schema = greet.parameters_schema()
    assert "name" in schema["properties"]
    assert "name" in schema["required"]


def test_schema_optional_field_has_default():
    @tool
    def repeat(text: str, times: int = 2) -> str:
        """Repeat text."""
        return text * times

    schema = repeat.parameters_schema()
    assert "times" in schema["properties"]
    assert "times" not in schema.get("required", [])


def test_schema_type_annotation_respected():
    @tool
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    schema = add.parameters_schema()
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "integer"


def test_schema_no_ctx_in_properties():
    @tool
    def uses_ctx(ctx: ToolContext, name: str) -> str:
        """Uses context."""
        return name

    schema = uses_ctx.parameters_schema()
    assert "ctx" not in schema["properties"]


def test_schema_additional_properties_forbidden():
    @tool
    def strict(x: str) -> str:
        """Strict."""
        return x

    assert strict.parameters_schema().get("additionalProperties") is False


# ---------------------------------------------------------------------------
# Execution — sync tools
# ---------------------------------------------------------------------------


def test_sync_tool_str_result(ctx):
    @tool
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"

    result = asyncio.run(greet(ctx, {"name": "World"}))
    assert result.output == "Hello, World!"
    assert result.error is False


def test_sync_tool_title_is_tool_name(ctx):
    @tool
    def my_op(x: str) -> str:
        """An op."""
        return x

    result = asyncio.run(my_op(ctx, {"x": "val"}))
    assert result.title == "my_op"


def test_sync_tool_toolresult_returned_directly(ctx):
    @tool
    def custom(x: str) -> ToolResult:
        """Returns ToolResult directly."""
        return ToolResult(title="custom_title", output="custom_output")

    result = asyncio.run(custom(ctx, {"x": "anything"}))
    assert result.title == "custom_title"
    assert result.output == "custom_output"


def test_sync_tool_default_param_used(ctx):
    @tool
    def repeat(text: str, times: int = 3) -> str:
        """Repeat."""
        return text * times

    result = asyncio.run(repeat(ctx, {"text": "hi "}))
    assert result.output == "hi hi hi "


def test_sync_tool_default_param_overridden(ctx):
    @tool
    def repeat(text: str, times: int = 3) -> str:
        """Repeat."""
        return text * times

    result = asyncio.run(repeat(ctx, {"text": "yo ", "times": 2}))
    assert result.output == "yo yo "


# ---------------------------------------------------------------------------
# Execution — async tools
# ---------------------------------------------------------------------------


def test_async_tool_executes(ctx):
    @tool
    async def async_greet(name: str) -> str:
        """Async hello."""
        return f"Hi, {name}!"

    result = asyncio.run(async_greet(ctx, {"name": "Alice"}))
    assert result.output == "Hi, Alice!"
    assert result.error is False


def test_async_tool_returns_toolresult(ctx):
    @tool
    async def async_op(value: int) -> ToolResult:
        """Async op."""
        return ToolResult(title="async_result", output=str(value * 2))

    result = asyncio.run(async_op(ctx, {"value": 5}))
    assert result.title == "async_result"
    assert result.output == "10"


# ---------------------------------------------------------------------------
# Context injection
# ---------------------------------------------------------------------------


def test_ctx_injected_when_declared(tmp_path):
    received: list[ToolContext] = []

    @tool
    def capture_ctx(ctx: ToolContext, x: str) -> str:
        """Captures context."""
        received.append(ctx)
        return x

    ctx_obj = ToolContext(
        session_id="sid",
        message_id="mid",
        agent_name="ag",
        project_id="proj",
        working_dir=str(tmp_path),
    )
    asyncio.run(capture_ctx(ctx_obj, {"x": "val"}))
    assert len(received) == 1
    assert received[0].session_id == "sid"


def test_ctx_not_injected_when_not_declared(tmp_path):
    """Tool without ctx param must not receive it in kwargs."""

    @tool
    def no_ctx(x: str) -> str:
        """No context."""
        return x

    ctx_obj = ToolContext(
        session_id="s",
        message_id="m",
        agent_name="a",
        project_id="p",
        working_dir=str(tmp_path),
    )
    result = asyncio.run(no_ctx(ctx_obj, {"x": "hello"}))
    assert result.output == "hello"


# ---------------------------------------------------------------------------
# Error handling — bad args
# ---------------------------------------------------------------------------


def test_bad_json_returns_error_result(ctx):
    @tool
    def my_tool(x: str) -> str:
        """A tool."""
        return x

    result = asyncio.run(my_tool(ctx, "not valid json {{{"))
    assert result.error is True
    assert "invalid JSON" in result.title


def test_wrong_key_returns_error_result(ctx):
    @tool
    def my_tool(x: str) -> str:
        """A tool."""
        return x

    result = asyncio.run(my_tool(ctx, {"wrong_key": "value"}))
    assert result.error is True
    assert "bad parameters" in result.title.lower()


def test_wrong_type_returns_error_result(ctx):
    @tool
    def typed_tool(count: int) -> str:
        """Needs int."""
        return str(count)

    # pydantic will coerce "3" -> 3, but a non-numeric string should fail
    result = asyncio.run(typed_tool(ctx, {"count": "not_a_number"}))
    assert result.error is True


def test_extra_field_returns_error_result(ctx):
    @tool
    def strict_tool(x: str) -> str:
        """Strict."""
        return x

    result = asyncio.run(strict_tool(ctx, {"x": "ok", "extra": "bad"}))
    assert result.error is True


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_tool_registered_in_user_registry():
    from openvibe.tool.base import _get_user_registry

    @tool
    def my_registered_tool(x: str) -> str:
        """Registered tool."""
        return x

    assert "my_registered_tool" in _get_user_registry()


def test_tool_appears_in_default_registry():
    @tool
    def custom_search(query: str) -> str:
        """Search."""
        return query

    reg = create_default_registry()
    assert "custom_search" in reg


def test_multiple_tools_all_in_default_registry():
    @tool
    def tool_a(x: str) -> str:
        """A."""
        return x

    @tool
    def tool_b(y: int) -> str:
        """B."""
        return str(y)

    reg = create_default_registry()
    assert "tool_a" in reg
    assert "tool_b" in reg


def test_builtin_tools_still_present_alongside_user_tools():
    @tool
    def extra(x: str) -> str:
        """Extra."""
        return x

    reg = create_default_registry()
    # Built-ins must still be registered
    for builtin in ("bash", "read", "write", "edit", "glob", "grep"):
        assert builtin in reg, f"built-in tool '{builtin}' missing from registry"


def test_tool_retrievable_from_registry(ctx):
    @tool
    def fetchable(msg: str) -> str:
        """Fetchable tool."""
        return msg

    reg = create_default_registry()
    retrieved = reg.get("fetchable")
    assert retrieved is not None
    result = asyncio.run(retrieved(ctx, {"msg": "hi"}))
    assert result.output == "hi"


def test_no_user_tools_registered_registry_still_works():
    # No @tool decorations — default registry must still contain built-ins
    reg = create_default_registry()
    assert "bash" in reg


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


def test_long_output_is_truncated(ctx):
    @tool
    def big_output(x: str) -> str:
        """Returns a very long string."""
        return "x" * 10_000

    result = asyncio.run(big_output(ctx, {"x": "ignored"}))
    assert result.metadata.get("truncated") is True
    assert len(result.output) < 10_000
    assert result.output.endswith("… [truncated]")


def test_short_output_not_truncated(ctx):
    @tool
    def small_output(x: str) -> str:
        """Returns a short string."""
        return "hello"

    result = asyncio.run(small_output(ctx, {"x": "anything"}))
    assert not result.metadata.get("truncated")
    assert result.output == "hello"
