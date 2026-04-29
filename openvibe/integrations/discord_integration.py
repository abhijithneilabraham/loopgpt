"""Discord bot integration.

Required package: pip install discord.py
Required bot permissions: Send Messages, Read Message History
Required gateway intents: messages, message_content, guild_messages, dm_messages
"""

from __future__ import annotations

import asyncio
import logging
import threading

from openvibe.api import OpenVibe
from openvibe.config import DiscordIntegrationConfig
from openvibe.integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

_MAX_DISCORD_LENGTH = 1_900  # Discord limit is 2000 chars


def _split_message(text: str) -> list[str]:
    if len(text) <= _MAX_DISCORD_LENGTH:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:_MAX_DISCORD_LENGTH])
        text = text[_MAX_DISCORD_LENGTH:]
    return chunks


class DiscordIntegration(BaseIntegration):
    name = "discord"

    def __init__(self, ov: OpenVibe, cfg: DiscordIntegrationConfig) -> None:
        super().__init__(ov, auto_approve=cfg.auto_approve, default_agent=cfg.default_agent)
        self._cfg = cfg
        self._loop: asyncio.AbstractEventLoop | None = None

    def _start(self) -> None:
        try:
            import discord
        except ImportError:
            logger.error(
                "[discord] discord.py not installed. Run: pip install discord.py"
            )
            return

        if not self._cfg.token:
            logger.error(
                "[discord] token is required. "
                "Set it in openvibe.json under integrations.discord"
            )
            return

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready() -> None:
            logger.info("[discord] logged in as %s", client.user)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author == client.user:
                return

            # Respond to DMs or to @mentions in servers
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mention = client.user in message.mentions if client.user else False

            if not (is_dm or is_mention):
                return

            text = message.content
            # Strip bot mention
            if client.user:
                text = text.replace(f"<@{client.user.id}>", "").strip()
                text = text.replace(f"<@!{client.user.id}>", "").strip()
            if not text:
                return

            channel_key = f"discord:{message.channel.id}:{message.author.id}"
            logger.info(
                "[discord] message from %s in %s: %s",
                message.author,
                message.channel,
                text[:80],
            )

            # Send a typing indicator while processing
            async with message.channel.typing():
                # Run the blocking openvibe call in a thread pool
                loop = asyncio.get_event_loop()
                response_text = await loop.run_in_executor(
                    None, lambda: self._run_prompt(text, channel_key)
                )

            for chunk in _split_message(response_text):
                await message.channel.send(chunk)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        logger.info("[discord] connecting…")
        self._loop.run_until_complete(client.start(self._cfg.token))

    def _stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
