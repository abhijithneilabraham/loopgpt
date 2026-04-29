"""Gateway manager — starts all enabled integrations and blocks until interrupted."""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any

from openvibe.api import OpenVibe
from openvibe.config import Config, IntegrationsConfig, load_config

logger = logging.getLogger(__name__)


class GatewayManager:
    """Starts and supervises all enabled openvibe integrations.

    Usage::

        manager = GatewayManager.from_config(project_dir=Path("."))
        manager.run()  # blocks until Ctrl-C
    """

    def __init__(self, ov: OpenVibe, cfg: IntegrationsConfig) -> None:
        self._ov = ov
        self._cfg = cfg
        self._integrations: list[Any] = []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, project_dir: Path | None = None, config: Config | None = None) -> "GatewayManager":
        if config is None:
            config = load_config(project_dir)
        ov = OpenVibe(project_dir=project_dir, config=config)
        ov.start()
        return cls(ov, config.integrations)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all enabled integrations and block until SIGINT/SIGTERM."""
        self._build_integrations()

        if not self._integrations:
            logger.warning(
                "No integrations are enabled. "
                "Configure at least one under 'integrations' in openvibe.json."
            )
            return

        for integration in self._integrations:
            integration.start()

        logger.info("Gateway running with %d integration(s). Press Ctrl-C to stop.", len(self._integrations))

        # Block until interrupted
        stop = [False]

        def _signal_handler(sig: int, frame: Any) -> None:
            stop[0] = True

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            while not stop[0]:
                time.sleep(0.5)
        finally:
            self._shutdown()

    def status(self) -> list[dict[str, str]]:
        """Return status of all configured integrations."""
        result = []
        cfg = self._cfg
        checks = [
            ("slack", cfg.slack.enabled, _slack_status(cfg.slack)),
            ("discord", cfg.discord.enabled, _discord_status(cfg.discord)),
            ("telegram", cfg.telegram.enabled, _telegram_status(cfg.telegram)),
            ("webhook", cfg.webhook.enabled, _webhook_status(cfg.webhook)),
            ("teams", cfg.teams.enabled, _teams_status(cfg.teams)),
        ]
        for name, enabled, notes in checks:
            result.append({"name": name, "enabled": "yes" if enabled else "no", "notes": notes})
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_integrations(self) -> None:
        cfg = self._cfg

        if cfg.slack.enabled:
            from openvibe.integrations.slack_integration import SlackIntegration
            self._integrations.append(SlackIntegration(self._ov, cfg.slack))
            logger.info("Slack integration enabled")

        if cfg.discord.enabled:
            from openvibe.integrations.discord_integration import DiscordIntegration
            self._integrations.append(DiscordIntegration(self._ov, cfg.discord))
            logger.info("Discord integration enabled")

        if cfg.telegram.enabled:
            from openvibe.integrations.telegram_integration import TelegramIntegration
            self._integrations.append(TelegramIntegration(self._ov, cfg.telegram))
            logger.info("Telegram integration enabled")

        if cfg.webhook.enabled:
            from openvibe.integrations.webhook_integration import WebhookIntegration
            self._integrations.append(WebhookIntegration(self._ov, cfg.webhook))
            logger.info("Webhook integration enabled (port %s)", cfg.webhook.port)

        if cfg.teams.enabled:
            from openvibe.integrations.teams_integration import TeamsIntegration
            self._integrations.append(TeamsIntegration(self._ov, cfg.teams))
            logger.info("Teams integration enabled")

    def _shutdown(self) -> None:
        for integration in self._integrations:
            try:
                integration.stop()
            except Exception:
                logger.exception("Error stopping %s", integration.name)
        try:
            self._ov.stop()
        except Exception:
            pass
        logger.info("Gateway stopped.")


# ------------------------------------------------------------------
# Status helpers
# ------------------------------------------------------------------


def _slack_status(cfg: Any) -> str:
    if not cfg.enabled:
        return "disabled"
    issues = []
    if not cfg.bot_token:
        issues.append("bot_token missing")
    if not cfg.app_token:
        issues.append("app_token missing")
    return ", ".join(issues) if issues else "configured"


def _discord_status(cfg: Any) -> str:
    if not cfg.enabled:
        return "disabled"
    return "token missing" if not cfg.token else "configured"


def _telegram_status(cfg: Any) -> str:
    if not cfg.enabled:
        return "disabled"
    return "token missing" if not cfg.token else "configured"


def _webhook_status(cfg: Any) -> str:
    if not cfg.enabled:
        return "disabled"
    return f"port {cfg.port}" + (" (no secret)" if not cfg.secret else " (secret set)")


def _teams_status(cfg: Any) -> str:
    if not cfg.enabled:
        return "disabled"
    issues = []
    if not cfg.app_id:
        issues.append("app_id missing")
    if not cfg.app_password:
        issues.append("app_password missing")
    return ", ".join(issues) if issues else "configured"
