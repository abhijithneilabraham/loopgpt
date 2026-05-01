"""LearnTool — record all mouse + keyboard interactions and replay them exactly.

Record::

    learn(action="start", prompt="fill the contact form", name="contact_form")
    # do the task
    learn(action="stop")

Replay::

    learn(action="replay", name="contact_form")

Every mouse move, click, scroll, key press, and key release is captured via
pynput and saved to ./openvibe_recordings/<name>.json.  Replay restores them
at the same speed using pynput controllers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openvibe.tool.base import Tool, ToolContext, ToolResult


class LearnTool(Tool):
    """Record all mouse + keyboard interactions, then replay them exactly."""

    name = "learn"
    description = (
        "Record every mouse and keyboard interaction, then replay exactly. "
        "Recordings saved to ./openvibe_recordings/. "
        "Actions: start | stop | list | replay | delete"
    )

    class Params(Tool.Params):
        action: Literal["start", "stop", "list", "replay", "delete"] = Field(
            description="Action: start | stop | list | replay | delete"
        )
        prompt: str = Field(default="", description="Task description (for start).")
        name: str = Field(default="", description="Recording name.")
        speed: float = Field(default=1.0, description="Replay speed multiplier (1.0 = realtime).")

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
            return await loop.run_in_executor(None, self._replay, params)
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
                "Every mouse and keyboard event is being captured.\n"
                "Call learn(action='stop') when done."
            ),
        )

    def _stop(self) -> ToolResult:
        from openvibe.computer.recorder import get_recorder
        rec = get_recorder()
        if not rec.is_recording:
            return ToolResult(title="Not recording",
                              output="Call learn(action='start') first.", error=True)
        recording = rec.stop()
        path = recording.save()
        return ToolResult(
            title=f"Saved: {recording.name}",
            output=(
                f"Recorded {len(recording.events)} events in {recording.duration:.1f}s\n"
                f"Saved to: {path}\n\n"
                f"Replay: learn(action='replay', name={recording.name!r})"
            ),
        )

    def _list(self) -> ToolResult:
        from openvibe.computer.recorder import Recording
        recs = Recording.list_all()
        if not recs:
            return ToolResult(title="No recordings",
                              output="No recordings yet. Use learn(action='start').")
        lines = [f"Recordings ({len(recs)}):"]
        for r in recs:
            lines.append(f"\n  [{r.name}]  {r.prompt}")
            lines.append(f"    {r.recorded_at[:19]}  {r.duration:.1f}s")
        return ToolResult(title="Recordings", output="\n".join(lines))

    def _replay(self, params: "LearnTool.Params") -> ToolResult:
        from openvibe.computer.recorder import Recording
        query = params.name or params.prompt
        if not query:
            return ToolResult(title="learn: replay", output="Provide name.", error=True)
        recording = Recording.load(query)
        if recording is None:
            return ToolResult(title="Not found",
                              output=f"No recording matching {query!r}.", error=True)
        if not recording.events:
            return ToolResult(title="No events",
                              output="Recording has no events. Re-record.", error=True)
        outcome = recording.replay_with_vision(speed=params.speed)
        clicks = sum(1 for e in recording.events if e.type == "click" and e.data.get("pressed"))
        msg = f"Replayed {len(recording.events)} events ({clicks} clicks)."
        if outcome == "interrupted":
            msg += " Stopped early — human input detected."
        return ToolResult(title=f"Replay {outcome}: {recording.name}", output=msg)

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
