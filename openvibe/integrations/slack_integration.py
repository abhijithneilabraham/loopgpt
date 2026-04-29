"""Slack integration using Socket Mode (no public URL required).

Required package: pip install slack-bolt
Required Slack app scopes:
  - app_mentions:read
  - chat:write
  - im:history, im:read, im:write   (for DMs)
  - channels:history                 (for channel messages)
"""

from __future__ import annotations

import logging

from openvibe.api import OpenVibe
from openvibe.config import SlackIntegrationConfig
from openvibe.integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

_MAX_SLACK_LENGTH = 3_900  # Slack message limit is 4000 chars; leave headroom


def _split_message(text: str) -> list[str]:
    """Split long text into Slack-sized chunks."""
    if len(text) <= _MAX_SLACK_LENGTH:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:_MAX_SLACK_LENGTH])
        text = text[_MAX_SLACK_LENGTH:]
    return chunks


class SlackIntegration(BaseIntegration):
    name = "slack"

    def __init__(self, ov: OpenVibe, cfg: SlackIntegrationConfig) -> None:
        super().__init__(ov, auto_approve=cfg.auto_approve, default_agent=cfg.default_agent)
        self._cfg = cfg

    def _start(self) -> None:
        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler
        except ImportError:
            logger.error(
                "[slack] slack-bolt not installed. Run: pip install slack-bolt"
            )
            return

        if not self._cfg.bot_token or not self._cfg.app_token:
            logger.error(
                "[slack] bot_token and app_token are required. "
                "Set them in openvibe.json under integrations.slack"
            )
            return

        bolt_app = App(token=self._cfg.bot_token)

        @bolt_app.event("app_mention")
        def handle_mention(event: dict, say: object) -> None:  # type: ignore[type-arg]
            self._handle_event(event, say)

        @bolt_app.event("message")
        def handle_dm(event: dict, say: object) -> None:  # type: ignore[type-arg]
            # Only handle DMs (channel_type == "im"), not channel messages
            # (those are caught by app_mention).
            if event.get("channel_type") != "im":
                return
            if event.get("bot_id"):
                return  # ignore own messages
            self._handle_event(event, say)

        self._handler = SocketModeHandler(bolt_app, self._cfg.app_token)
        logger.info("[slack] connecting via Socket Mode…")
        self._handler.start()  # blocks until stopped

    def _stop(self) -> None:
        handler = getattr(self, "_handler", None)
        if handler:
            try:
                handler.close()
            except Exception:
                pass

    def _handle_event(self, event: dict, say: object) -> None:  # type: ignore[type-arg]
        text: str = event.get("text", "").strip()
        # Strip the bot mention prefix (<@UXXXXXXXX> ...)
        import re
        text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
        if not text:
            return

        channel: str = event.get("channel", "unknown")
        user: str = event.get("user", "unknown")
        channel_key = f"slack:{channel}:{user}"

        logger.info("[slack] message from %s in %s: %s", user, channel, text[:80])

        # Post a "thinking…" placeholder
        try:
            initial = say(":hourglass_flowing_sand: Working on it…")  # type: ignore[operator]
            ts = initial.get("ts") if initial else None
        except Exception:
            ts = None

        response_text = self._run_prompt(text, channel_key)

        # Update the placeholder or post new messages
        try:
            from slack_sdk import WebClient
            client = WebClient(token=self._cfg.bot_token)
            chunks = _split_message(response_text)
            if ts:
                # Edit the placeholder with the first chunk
                client.chat_update(channel=channel, ts=ts, text=chunks[0])
                for chunk in chunks[1:]:
                    client.chat_postMessage(channel=channel, text=chunk)
            else:
                for chunk in chunks:
                    say(chunk)  # type: ignore[operator]
        except Exception:
            logger.exception("[slack] failed to post response")
