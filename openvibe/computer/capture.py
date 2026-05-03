"""Screen-capture helpers using mss and Pillow. All functions are synchronous."""

from __future__ import annotations

import base64
import io


# Maximum dimensions before the image is downscaled.
# Anthropic's computer-use model is calibrated for ≤1024×768. Sending larger
# images causes coordinate mismatches because the model returns positions in
# the ~1024-wide space while callers assume the full-size image coordinate space.
_MAX_WIDTH = 1024
_MAX_HEIGHT = 768


def list_monitors() -> list[dict[str, int]]:
    """Return info dicts (left/top/width/height) for each monitor."""
    try:
        import mss
    except ImportError as exc:
        raise ImportError("mss is required: pip install mss pillow") from exc

    with mss.mss() as sct:
        # monitors[0] = virtual bounding box, [1..n] = individual screens
        return [
            {"left": m["left"], "top": m["top"], "width": m["width"], "height": m["height"]}
            for m in sct.monitors[1:]
        ]


def capture_screen(
    region: tuple[int, int, int, int] | None = None,
    monitor: int = 1,
) -> tuple[bytes, int, int]:
    """Capture a screenshot and return (png_bytes, width, height).

    region: (x, y, w, h) in logical pixels; None = full monitor.
    monitor: 1-indexed (1 = primary). Ignored when region is set.
    Raises ImportError if mss/Pillow missing, RuntimeError on data mismatch
    (macOS: grant Screen Recording permission).
    """
    try:
        import mss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "mss is required for computer use: pip install mss pillow"
        ) from exc

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for computer use: pip install mss pillow"
        ) from exc

    with mss.mss() as sct:
        if region is not None:
            x, y, w, h = region
            mon_dict: dict[str, int] = {"left": x, "top": y, "width": w, "height": h}
        else:
            # monitors[0] = virtual bounding box, monitors[1..n] = physical monitors
            mon_idx = max(1, min(monitor, len(sct.monitors) - 1))
            mon_dict = dict(sct.monitors[mon_idx] if len(sct.monitors) > mon_idx else sct.monitors[0])

        sct_img = sct.grab(mon_dict)

        # bytes() materialises mss's memoryview; PIL requires plain bytes.
        # BGRX decoder reorders channels BGRA→RGB correctly.
        raw_bgra = bytes(sct_img.bgra)

        expected = sct_img.width * sct_img.height * 4
        if len(raw_bgra) != expected:
            raise RuntimeError(
                f"Screen capture data size mismatch: received {len(raw_bgra)} bytes, "
                f"expected {expected} ({sct_img.width}×{sct_img.height}×4 BGRA). "
                "On macOS this usually means Screen Recording permission has not been "
                "granted — go to System Settings → Privacy & Security → Screen "
                "Recording and add your terminal application."
            )

        img = Image.frombytes(
            "RGB",
            (sct_img.width, sct_img.height),
            raw_bgra,
            "raw",
            "BGRX",   # read B-G-R-X, produce R-G-B
        )

    # Downscale to fit within _MAX_WIDTH × _MAX_HEIGHT (maintain aspect ratio).
    if img.width > _MAX_WIDTH or img.height > _MAX_HEIGHT:
        scale = min(_MAX_WIDTH / img.width, _MAX_HEIGHT / img.height)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    if not png_bytes:
        raise RuntimeError("PNG encoding produced empty output.")

    return png_bytes, img.width, img.height


def diff_screenshots(
    before_png: bytes,
    after_png: bytes,
    threshold: int = 10,
) -> dict[str, object]:
    """Compare two PNG screenshots and return a change report.

    Parameters
    ----------
    before_png, after_png:
        Raw PNG bytes from :func:`capture_screen`.
    threshold:
        Per-channel intensity delta (0–255) below which a pixel is considered
        unchanged.  Default 10 filters camera/compression noise.

    Returns
    -------
    dict with keys:
        ``changed``        — bool, True if any pixel changed above threshold
        ``change_fraction``— float 0.0–1.0, fraction of pixels that changed
        ``changed_region`` — [x, y, w, h] bounding box of the changed area, or None
        ``summary``        — human-readable string for the LLM
    """
    try:
        from PIL import Image, ImageChops  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for screenshot diffing: pip install pillow"
        ) from exc

    before = Image.open(io.BytesIO(before_png)).convert("RGB")
    after = Image.open(io.BytesIO(after_png)).convert("RGB")

    if before.size != after.size:
        return {
            "changed": True,
            "change_fraction": 1.0,
            "changed_region": None,
            "summary": (
                f"Screen resolution changed from {before.size} to {after.size}."
            ),
        }

    diff = ImageChops.difference(before, after)
    gray = diff.convert("L")

    # Build a binary mask: 255 where change exceeds threshold, 0 elsewhere.
    mask = gray.point(lambda p: 255 if p > threshold else 0)
    from PIL import ImageStat  # type: ignore[import-not-found]
    changed_pixels = int(ImageStat.Stat(mask).sum[0] / 255)
    total = before.width * before.height
    fraction = changed_pixels / total if total > 0 else 0.0

    if fraction < 0.001:
        return {
            "changed": False,
            "change_fraction": fraction,
            "changed_region": None,
            "summary": (
                "No visible change detected — the action may not have had any effect."
            ),
        }

    bbox = mask.getbbox()  # (left, top, right, bottom) of non-zero region
    changed_region = None
    region_str = ""
    if bbox:
        x, y, x2, y2 = bbox
        changed_region = [x, y, x2 - x, y2 - y]
        region_str = f" in region [x={x}, y={y}, {x2-x}×{y2-y}px]"

    summary = f"{fraction:.1%} of pixels changed{region_str}."
    return {
        "changed": True,
        "change_fraction": fraction,
        "changed_region": changed_region,
        "summary": summary,
    }


def capture_screen_b64(
    region: tuple[int, int, int, int] | None = None,
) -> tuple[str, int, int]:
    """Like :func:`capture_screen` but returns a base-64 encoded PNG string."""
    png_bytes, w, h = capture_screen(region)
    return base64.b64encode(png_bytes).decode("ascii"), w, h


def screen_size() -> tuple[int, int]:
    """Return ``(width, height)`` of the primary monitor."""
    try:
        import mss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("mss is required: pip install mss") from exc

    with mss.mss() as sct:
        m = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        return m["width"], m["height"]
