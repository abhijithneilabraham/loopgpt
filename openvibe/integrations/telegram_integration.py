"""Telegram bot integration using long polling (no public URL required).

Required package: pip install python-telegram-bot
"""

from __future__ import annotations

import logging

from openvibe.api import OpenVibe
from openvibe.config import TelegramIntegrationConfig
from openvibe.integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

_MAX_TELEGRAM_LENGTH = 4_000  # Telegram limit is 4096 chars


def _split_message(text: str) -> list[str]:
    if len(text) <= _MAX_TELEGRAM_LENGTH:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:_MAX_TELEGRAM_LENGTH])
        text = text[_MAX_TELEGRAM_LENGTH:]
    return chunks


class TelegramIntegration(BaseIntegration):
    name = "telegram"

    def __init__(self, ov: OpenVibe, cfg: TelegramIntegrationConfig) -> None:
        super().__init__(ov, auto_approve=cfg.auto_approve, default_agent=cfg.default_agent)
        self._cfg = cfg
        self._app: object | None = None

    def _start(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
        except ImportError:
            logger.error(
                "[telegram] python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot"
            )
            return

        if not self._cfg.token:
            logger.error(
                "[telegram] token is required. "
                "Set it in openvibe.json under integrations.telegram"
            )
            return

        allowed = set(self._cfg.allowed_chat_ids)

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message or not update.effective_chat:
                return

            chat_id = update.effective_chat.id
            if allowed and chat_id not in allowed:
                logger.warning("[telegram] ignoring message from chat_id=%s", chat_id)
                return

            text = (update.message.text or "").strip()
            if not text:
                return

            user_id = update.effective_user.id if update.effective_user else chat_id
            channel_key = f"telegram:{chat_id}:{user_id}"

            logger.info("[telegram] message from %s: %s", chat_id, text[:80])

            # Show "typing…" indicator
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")

            # Run blocking openvibe call in executor
            import asyncio
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, lambda: self._run_prompt(text, channel_key)
            )

            for chunk in _split_message(response_text):
                await update.message.reply_text(chunk)

        async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if update.message:
                await update.message.reply_text(
                    "Hi! I'm your openvibe assistant. Send me a message and I'll get to work."
                )

        app = (
            Application.builder()
            .token(self._cfg.token)
            .build()
        )
        app.add_handler(CommandHandler("start", handle_start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        self._app = app
        logger.info("[telegram] starting polling…")
        app.run_polling(stop_signals=None)

    def _stop(self) -> None:
        app = self._app
        if app is not None:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(app.stop())  # type: ignore[union-attr]
                loop.close()
            except Exception:
                pass
