"""Microsoft Teams integration via Azure Bot Framework webhook.

Required package: pip install botframework-connector
Requires a public HTTPS endpoint (or use ngrok/tunnelmole for local dev).

Setup
-----
1. Go to https://dev.botframework.com/ and create a new bot registration
2. Under Configuration, set the Messaging endpoint to:
   https://<your-domain>/teams/messages
3. Copy the Microsoft App ID and generate a client secret (App Password)
4. Add the bot to your Teams workspace via the manifest or App Studio
5. Set integrations.teams.app_id and integrations.teams.app_password in openvibe.json

The Teams integration shares the same webhook port as the webhook integration
if both are enabled, or defaults to port 3978 (the Bot Framework default).
"""

from __future__ import annotations

import logging
from typing import Any

from openvibe.api import OpenVibe
from openvibe.config import TeamsIntegrationConfig
from openvibe.integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 3978


class TeamsIntegration(BaseIntegration):
    name = "teams"

    def __init__(self, ov: OpenVibe, cfg: TeamsIntegrationConfig) -> None:
        super().__init__(ov, auto_approve=cfg.auto_approve, default_agent=cfg.default_agent)
        self._cfg = cfg
        self._server: Any = None

    def _start(self) -> None:
        try:
            import uvicorn
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
        except ImportError:
            logger.error("[teams] fastapi/uvicorn not installed")
            return

        try:
            from botframework.connector.auth import (
                MicrosoftAppCredentials,
                JwtTokenValidation,
                SimpleCredentialProvider,
            )
            from botframework.connector import ConnectorClient
        except ImportError:
            logger.error(
                "[teams] botframework-connector not installed. "
                "Run: pip install botframework-connector"
            )
            return

        if not self._cfg.app_id or not self._cfg.app_password:
            logger.error(
                "[teams] app_id and app_password are required. "
                "Set them in openvibe.json under integrations.teams"
            )
            return

        app_id = self._cfg.app_id
        app_password = self._cfg.app_password
        credentials = MicrosoftAppCredentials(app_id, app_password)
        credential_provider = SimpleCredentialProvider(app_id, app_password)

        api = FastAPI(title="openvibe-teams", docs_url=None, redoc_url=None)

        @api.post("/teams/messages")
        async def messages(request: Request) -> JSONResponse:
            import asyncio

            body = await request.json()
            activity_type = body.get("type", "")

            if activity_type != "message":
                return JSONResponse({"status": "ok"})

            text: str = (body.get("text") or "").strip()
            if not text:
                return JSONResponse({"status": "ok"})

            # Extract conversation reference for replying
            conversation = body.get("conversation", {})
            conv_id: str = conversation.get("id", "default")
            service_url: str = body.get("serviceUrl", "")
            channel_key = f"teams:{conv_id}"

            logger.info("[teams] message in conv %s: %s", conv_id, text[:80])

            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, lambda: self._run_prompt(text, channel_key)
            )

            # Send reply via Bot Framework Connector
            try:
                connector = ConnectorClient(credentials, base_url=service_url)
                reply = {
                    "type": "message",
                    "text": response_text,
                    "conversation": conversation,
                }
                connector.conversations.send_to_conversation(conv_id, reply)
            except Exception:
                logger.exception("[teams] failed to send reply")

            return JSONResponse({"status": "ok"})

        config = uvicorn.Config(api, host="0.0.0.0", port=_DEFAULT_PORT, log_level="warning")
        self._server = uvicorn.Server(config)
        logger.info("[teams] listening on 0.0.0.0:%s/teams/messages", _DEFAULT_PORT)
        self._server.run()

    def _stop(self) -> None:
        if self._server:
            self._server.should_exit = True
