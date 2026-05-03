"""ScreenshotTool — capture the current screen as a base-64 PNG for the LLM."""

from __future__ import annotations

import asyncio
import os

from pydantic import Field

from openvibe.tool.base import Attachment, Tool, ToolContext, ToolResult


class ScreenshotTool(Tool):
    """Capture a screenshot of the full screen or a specific region."""

    name = "screenshot"
    description = (
        "Capture a screenshot of the current screen or a sub-region. "
        "Returns the image so you can observe the current UI state before "
        "deciding which action to take next. Call this frequently to verify "
        "that previous actions had the intended effect. "
        "Use save_path to write the PNG directly to disk (e.g. '/Users/you/Desktop/shot.png')."
    )

    class Params(Tool.Params):
        region: list[int] | None = Field(
            default=None,
            description=(
                "Optional screen region as [x, y, width, height] in pixels. "
                "Omit to capture the full monitor."
            ),
        )
        monitor: int = Field(
            default=1,
            description=(
                "1-indexed monitor to capture (1 = primary, 2 = second display, …). "
                "Ignored when region is provided."
            ),
        )
        save_path: str | None = Field(
            default=None,
            description=(
                "Optional absolute path to save the PNG to disk. "
                "Parent directories are created automatically."
            ),
        )

    async def execute(self, ctx: ToolContext, params: "ScreenshotTool.Params") -> ToolResult:  # type: ignore[override]
        from openvibe.computer.capture import capture_screen
        from openvibe.computer.sandbox import ActionType, get_sandbox

        await ctx.check_permission(
            tool="screenshot",
            argument="capture screen",
            description="Take a screenshot of the current screen",
        )

        region: tuple[int, int, int, int] | None = None
        if params.region:
            if len(params.region) != 4:
                return ToolResult(
                    title="Screenshot error",
                    output="region must have exactly 4 elements: [x, y, width, height]",
                    error=True,
                )
            region = (params.region[0], params.region[1], params.region[2], params.region[3])

        sandbox = get_sandbox(ctx.session_id)
        if region and not sandbox.is_coordinate_allowed(region[0], region[1]):
            return ToolResult(
                title="Screenshot denied",
                output="The requested region is outside the permitted screen area.",
                error=True,
            )

        try:
            loop = asyncio.get_event_loop()
            png_bytes, width, height = await loop.run_in_executor(
                None, lambda: capture_screen(region=region, monitor=params.monitor)
            )
        except ImportError as exc:
            return ToolResult(
                title="Screenshot error",
                output=str(exc),
                error=True,
            )
        except Exception as exc:
            await sandbox.record_action(
                ActionType.SCREENSHOT,
                params={"region": params.region},
                error=str(exc),
            )
            return ToolResult(
                title="Screenshot error",
                output=f"Failed to capture screenshot: {exc}",
                error=True,
            )

        # Diff vs previous screenshot for change-detection feedback.
        diff_summary: str | None = None
        if sandbox.last_screenshot is not None:
            try:
                from openvibe.computer.capture import diff_screenshots
                loop = asyncio.get_event_loop()
                diff = await loop.run_in_executor(
                    None, diff_screenshots, sandbox.last_screenshot, png_bytes
                )
                diff_summary = diff["summary"]  # type: ignore[index]
            except Exception:
                pass  # diff is best-effort; never block the screenshot

        sandbox.last_screenshot = png_bytes

        # Compute and store the image→logical-pixel scale so the mouse tool can
        # convert coordinates without requiring image_width/image_height to be
        # re-supplied on every call.
        try:
            from openvibe.computer.input import screen_size as _ss
            _lw, _lh = _ss()
            sandbox.coord_scale = (_lw / width, _lh / height)
        except Exception:
            sandbox.coord_scale = (1.0, 1.0)

        await sandbox.record_action(
            ActionType.SCREENSHOT,
            params={"region": params.region},
            result=f"{width}x{height}" + (f" | {diff_summary}" if diff_summary else ""),
        )

        saved_path: str | None = None
        if params.save_path:
            try:
                dest = os.path.expanduser(params.save_path)
                os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else ".", exist_ok=True)
                with open(dest, "wb") as fh:
                    fh.write(png_bytes)
                saved_path = dest
            except Exception as exc:
                return ToolResult(
                    title="Screenshot save error",
                    output=f"Screenshot captured ({width}×{height}) but could not be saved to {params.save_path!r}: {exc}",
                    attachments=[
                        Attachment(filename="screenshot.png", content=png_bytes, media_type="image/png")
                    ],
                    metadata={"width": width, "height": height},
                    error=True,
                )

        region_desc = f" (region {params.region})" if params.region else ""
        save_desc = f" → saved to {saved_path}" if saved_path else ""

        sx, sy = sandbox.coord_scale
        scale_note = (
            f" (coordinate scale {sx:.3f}×{sy:.3f} to screen)"
            if abs(sx - 1.0) > 0.02 or abs(sy - 1.0) > 0.02
            else ""
        )

        cursor_note = ""
        try:
            from openvibe.computer.input import get_mouse_position
            loop = asyncio.get_event_loop()
            cx, cy = await loop.run_in_executor(None, get_mouse_position)
            # Convert logical cursor position to image-space for the LLM
            inv_sx = 1.0 / sandbox.coord_scale[0] if sandbox.coord_scale[0] else 1.0
            inv_sy = 1.0 / sandbox.coord_scale[1] if sandbox.coord_scale[1] else 1.0
            img_cx = round(cx * inv_sx)
            img_cy = round(cy * inv_sy)
            cursor_note = f" | Cursor at ({img_cx},{img_cy}) in image coords"
        except Exception:
            pass

        output_lines = [
            f"Captured {width}×{height} screenshot{region_desc}{save_desc}{scale_note}{cursor_note}.",
            "Use the pixel coordinates you see in this image directly with the mouse tool — scaling is automatic.",
        ]
        if diff_summary:
            output_lines.append(f"Change detection: {diff_summary}")

        return ToolResult(
            title=f"Screenshot {width}×{height}{region_desc}",
            output="\n".join(output_lines),
            attachments=[
                Attachment(
                    filename="screenshot.png",
                    content=png_bytes,
                    media_type="image/png",
                )
            ],
            metadata={"width": width, "height": height, "truncated": True},
        )
