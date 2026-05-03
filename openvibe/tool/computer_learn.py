"""LearnTool — record interactions and replay through the computer use agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from openvibe.tool.base import Tool, ToolContext, ToolResult


class LearnTool(Tool):
    """Record mouse and keyboard interactions, then replay via computer use."""

    name = "learn"
    description = (
        "Record every mouse and keyboard interaction, then replay using computer use. "
        "Recordings saved to ./openvibe_recordings/. "
        "Actions: start | stop | list | replay | delete"
    )

    class Params(Tool.Params):
        action: Literal["start", "stop", "list", "replay", "delete"] = Field(
            description="Action: start | stop | list | replay | delete"
        )
        prompt: str = Field(default="", description="Task description (for start).")
        name: str = Field(default="", description="Recording name.")

    async def execute(self, ctx: ToolContext, params: "LearnTool.Params") -> ToolResult:
        import asyncio
        loop = asyncio.get_event_loop()
        if params.action == "start":
            return await loop.run_in_executor(None, self._start, params)
        if params.action == "stop":
            return await loop.run_in_executor(None, self._stop)
        if params.action == "list":
            return await loop.run_in_executor(None, self._list)
        if params.action == "replay":
            return await self._replay_async(ctx, params)
        if params.action == "delete":
            return await loop.run_in_executor(None, self._delete, params)
        return ToolResult(title="learn", output=f"Unknown action: {params.action}", error=True)

    def _start(self, params: "LearnTool.Params") -> ToolResult:
        from openvibe.computer.recorder import get_recorder, name_from_prompt
        rec = get_recorder()
        if rec.is_recording:
            return ToolResult(title="Already recording",
                              output="Call learn(action='stop') first.", error=True)
        prompt = params.prompt.strip()
        if not prompt:
            return ToolResult(title="learn: start", output="prompt is required.", error=True)
        name = params.name.strip() or name_from_prompt(prompt)
        rec.start(prompt, name)
        return ToolResult(
            title=f"Recording: {name}",
            output=(
                f"Recording started.\n"
                f"  Name:   {name}\n"
                f"  Prompt: {prompt}\n\n"
                "Capturing all mouse and keyboard interactions.\n"
                "Call learn(action='stop') when done."
            ),
        )

    def _stop(self) -> ToolResult:
        from openvibe.computer.recorder import get_recorder
        rec = get_recorder()
        if not rec.is_recording:
            return ToolResult(title="Not recording",
                              output="Call learn(action='start') first.", error=True)
        task_graph = rec.stop()
        path = task_graph.save()
        return ToolResult(
            title=f"Saved: {task_graph.name}",
            output=(
                f"Recorded {len(task_graph.steps)} steps in {task_graph.duration:.1f}s\n"
                f"Saved to: {path}\n\n"
                f"Replay: learn(action='replay', name={task_graph.name!r})"
            ),
        )

    def _list(self) -> ToolResult:
        from openvibe.computer.recorder import TaskGraph
        graphs = TaskGraph.list_all()
        if not graphs:
            return ToolResult(title="No recordings",
                              output="No recordings yet. Use learn(action='start').")
        lines = [f"Recordings ({len(graphs)}):"]
        for g in graphs:
            lines.append(f"\n  [{g.name}]  {g.prompt}")
            lines.append(f"    {g.recorded_at[:19]}  {g.duration:.1f}s  {len(g.steps)} steps")
        return ToolResult(title="Recordings", output="\n".join(lines))

    async def _replay_async(self, ctx: ToolContext, params: "LearnTool.Params") -> ToolResult:
        from openvibe.computer.recorder import TaskGraph
        query = params.name or params.prompt
        if not query:
            return ToolResult(title="learn: replay", output="Provide name.", error=True)
        task_graph = TaskGraph.load(query)
        if task_graph is None:
            return ToolResult(title="Not found",
                              output=f"No recording matching {query!r}.", error=True)
        if not task_graph.steps:
            return ToolResult(title="No steps",
                              output="Recording has no steps. Re-record.", error=True)

        # 3-second grace period so the user can position focus / switch windows.
        import asyncio, base64 as _b64
        await asyncio.sleep(3)

        # Capture current screen state for visual grounding.
        current_png: bytes = b""
        current_b64: str = ""
        try:
            from openvibe.computer.capture import capture_screen
            loop = asyncio.get_event_loop()
            current_png, _, _ = await loop.run_in_executor(None, capture_screen)
            current_b64 = _b64.b64encode(current_png).decode("ascii")
        except Exception:
            pass

        # Capture current cursor position (in logical screen coords) so the
        # replay prompt can include it for dynamic movement calculation.
        current_cursor: tuple[int, int] | None = None
        try:
            from openvibe.computer.input import get_mouse_position
            loop = asyncio.get_event_loop()
            current_cursor = await loop.run_in_executor(None, get_mouse_position)
        except Exception:
            pass

        # Attempt grounded replay if we have recorded screenshots.
        click_steps_with_shots = [
            s for s in task_graph.steps
            if s.type == "click" and s.target and s.target.screenshot_b64
        ]
        if click_steps_with_shots and current_b64:
            prompt = await self._ground_replay(task_graph, current_b64, current_cursor)
        else:
            prompt = task_graph.build_replay_prompt(current_cursor=current_cursor)

        from openvibe.tool.base import Attachment
        attachments = []
        if current_png:
            attachments.append(Attachment(
                filename="current_screen.png",
                content=current_png,
                media_type="image/png",
            ))

        cursor_line = (
            f"Current cursor position at replay start: ({current_cursor[0]}, {current_cursor[1]})\n"
            if current_cursor else ""
        )
        wrapped = (
            f"{prompt}\n\n"
            "---\n"
            f"{cursor_line}"
            "Replay recorded interactions on the desktop. Prefer screenshot, mouse, keyboard, ui, and app "
            "tools for UI interactions.\n"
            "- Take a screenshot before every action to see the current screen.\n"
            "- The screenshot output includes your current cursor position — use it to calculate movements.\n"
            "- For each click step: move from current cursor position using the recorded movement delta as guidance,\n"
            "  then verify visually before clicking.\n"
            "- For each scroll step: move mouse to the recorded scroll position, then scroll the recorded amount.\n"
            "- For each hotkey step: use keyboard(action=hotkey, keys=[...]).\n"
            "- If an action fails or has no visible effect, take another screenshot and adapt.\n"
            "- Never stop to ask the user. Keep going until the task is fully complete."
        )
        return ToolResult(
            title=f"Replay: {task_graph.name}",
            output=f"Loaded {len(task_graph.steps)} steps for '{task_graph.name}'. Starting in 3s...",
            attachments=attachments,
            follow_up_message=wrapped,
            follow_up_agent="computer",
            metadata={"truncated": True},
        )

    async def _ground_replay(
        self,
        task_graph: Any,
        current_b64: str,
        current_cursor: tuple[int, int] | None = None,
    ) -> str:
        """Call the vision LLM with recorded + current screenshots to produce grounded instructions."""
        from openvibe.llm import (
            ContentBlock, Message, StreamDone, TextDelta,
            create_default_backend, resolve_model,
        )

        backend = create_default_backend()
        model = resolve_model()

        cursor_context = (
            f"Current cursor position: ({current_cursor[0]}, {current_cursor[1]})\n\n"
            if current_cursor else ""
        )

        content: list[ContentBlock] = [
            ContentBlock(type="text", text=(
                f"Task: {task_graph.prompt}\n\n"
                f"{cursor_context}"
                "Below are the recorded steps with screenshots taken at each click. "
                "The CURRENT screen is shown last. Your job: produce a step-by-step execution plan "
                "with the actual pixel coordinates from the CURRENT screen for each action.\n\n"
                "For each click step:\n"
                "  1. Look at the recorded screenshot to understand WHAT was clicked.\n"
                "  2. Note the recorded movement delta (cursor_before → click_x,y).\n"
                "  3. Find the same element on the current screen visually.\n"
                "  4. Output its current coordinates.\n"
                "  5. Optionally compute: current_x ≈ cursor_now_x + recorded_dx, "
                "current_y ≈ cursor_now_y + recorded_dy as a starting estimate.\n\n"
                "Output format — for each step write:\n"
                "  CLICK x,y  (reason: element name + visual description)\n"
                "  TYPE 'text'\n"
                "  KEY keyname\n"
                "  HOTKEY ctrl+c  (or whichever combo)\n"
                "  SCROLL x,y dy=N  (N>0=up, N<0=down; dx=M for horizontal)\n\n"
                "Recorded steps:\n"
            )),
        ]

        for i, step in enumerate(task_graph.steps, 1):
            if step.type == "click" and step.target and step.target.screenshot_b64:
                t = step.target
                name = t.label or t.value or t.role or "element"
                move_note = (
                    f", cursor_before=({t.cursor_before_x},{t.cursor_before_y})"
                    f" delta=({t.movement_dx:+d},{t.movement_dy:+d})"
                    if (t.cursor_before_x or t.cursor_before_y)
                    else ""
                )
                content.append(ContentBlock(
                    type="text",
                    text=f"\nStep {i}: Click '{name}' [{t.role}] — recorded at ~({t.click_x},{t.click_y}){move_note}:",
                ))
                content.append(ContentBlock(
                    type="image_url",
                    image_url={"url": f"data:image/png;base64,{step.target.screenshot_b64}"},
                ))
            elif step.type == "click" and step.target:
                t = step.target
                name = t.label or t.value or t.role or "element"
                move_note = (
                    f" [delta ({t.movement_dx:+d},{t.movement_dy:+d}) from cursor_before"
                    f" ({t.cursor_before_x},{t.cursor_before_y})]"
                    if (t.cursor_before_x or t.cursor_before_y)
                    else ""
                )
                content.append(ContentBlock(
                    type="text",
                    text=f"\nStep {i}: Click '{name}' [{t.role}] at ~({t.click_x},{t.click_y}){move_note}",
                ))
            elif step.type == "type":
                content.append(ContentBlock(type="text", text=f"\nStep {i}: Type {step.value!r}"))
            elif step.type == "key":
                content.append(ContentBlock(
                    type="text",
                    text=f"\nStep {i}: Press {step.value.replace('Key.', '')}",
                ))
            elif step.type == "hotkey":
                content.append(ContentBlock(
                    type="text",
                    text=f"\nStep {i}: Hotkey {'+'.join(step.keys)}",
                ))
            elif step.type == "scroll":
                v = f"dy={step.scroll_dy}" if step.scroll_dy else ""
                h = f"dx={step.scroll_dx}" if step.scroll_dx else ""
                amounts = " ".join(filter(None, [v, h]))
                content.append(ContentBlock(
                    type="text",
                    text=f"\nStep {i}: Scroll at ({step.scroll_x},{step.scroll_y}) {amounts}",
                ))

        content += [
            ContentBlock(type="text", text="\n\nCurrent screen (use this to find actual coordinates):"),
            ContentBlock(type="image_url", image_url={"url": f"data:image/png;base64,{current_b64}"}),
            ContentBlock(type="text", text=(
                "\n\nNow output the complete grounded execution plan. "
                "Be precise with pixel coordinates. "
                "For scrolls, preserve direction and approximate amount. "
                "For hotkeys, reproduce the exact key combo. "
                "If an element isn't visible, find the closest visual alternative."
            )),
        ]

        collected: list[str] = []
        try:
            async for event in backend.stream(
                model=model,
                messages=[Message(role="user", content=content)],
                max_tokens=1024,
            ):
                if isinstance(event, TextDelta):
                    collected.append(event.text)
                elif isinstance(event, StreamDone):
                    break
        except Exception:
            pass

        if collected:
            return "".join(collected)
        return task_graph.build_replay_prompt()

    def _delete(self, params: "LearnTool.Params") -> ToolResult:
        from openvibe.computer.recorder import recordings_dir
        query = params.name or params.prompt
        if not query:
            return ToolResult(title="learn: delete", output="Provide name.", error=True)
        d = recordings_dir()
        for path in [d / query, d / f"{query}.json"]:
            if path.exists():
                path.unlink()
                return ToolResult(title="Deleted", output=f"Deleted {path.name}")
        return ToolResult(title="Not found",
                          output=f"No recording named {query!r}.", error=True)
