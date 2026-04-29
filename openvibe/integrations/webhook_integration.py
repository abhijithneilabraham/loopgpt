"""Generic inbound HTTP webhook integration.

Any tool that can send an HTTP POST — Zapier, Make, n8n, GitHub Actions,
curl, or custom scripts — can trigger openvibe through this endpoint.

API
---
POST /webhook
Headers:
  Content-Type: application/json
  X-Webhook-Secret: <secret>   (required if secret is configured)
Body (JSON):
  {
    "text": "fix the memory leak in server.py",
    "agent": "build",             // optional, defaults to default_agent
    "session_id": "abc123"        // optional, use existing session
  }
Response (JSON):
  {
    "text": "...",                // full agent response
    "session_id": "abc123"
  }

Streaming
---------
Add `Accept: text/event-stream` to get SSE token-by-token streaming.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import queue
import threading
from typing import Any

from openvibe.api import OpenVibe
from openvibe.config import WebhookIntegrationConfig
from openvibe.integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


class WebhookIntegration(BaseIntegration):
    name = "webhook"

    def __init__(self, ov: OpenVibe, cfg: WebhookIntegrationConfig) -> None:
        super().__init__(ov, auto_approve=cfg.auto_approve, default_agent=cfg.default_agent)
        self._cfg = cfg
        self._server: Any = None

    def _start(self) -> None:
        try:
            import uvicorn
            from fastapi import FastAPI, Header, HTTPException, Request
            from fastapi.responses import JSONResponse, StreamingResponse
        except ImportError:
            logger.error("[webhook] fastapi/uvicorn not installed (they should be)")
            return

        api = FastAPI(title="openvibe webhook", docs_url=None, redoc_url=None)
        cfg = self._cfg

        def _check_secret(secret_header: str | None, secret_param: str | None) -> None:
            if not cfg.secret:
                return
            provided = secret_header or secret_param or ""
            if not hmac.compare_digest(provided, cfg.secret):
                raise HTTPException(status_code=401, detail="Invalid secret")

        @api.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "integration": "openvibe-webhook"}

        @api.post("/webhook")
        async def webhook(
            request: Request,
            x_webhook_secret: str | None = Header(default=None, alias="x-webhook-secret"),
        ) -> JSONResponse | StreamingResponse:
            import asyncio

            body = await request.json()
            secret_param = request.query_params.get("secret")
            _check_secret(x_webhook_secret, secret_param)

            text: str = body.get("text", "").strip()
            if not text:
                raise HTTPException(status_code=400, detail="'text' is required")

            agent: str = body.get("agent", cfg.default_agent)
            session_id: str | None = body.get("session_id")
            channel_key = f"webhook:{session_id or 'default'}"

            accept = request.headers.get("accept", "")
            if "text/event-stream" in accept:
                # SSE streaming
                token_q: queue.Queue[str | None] = queue.Queue()

                def _on_token(t: str) -> None:
                    token_q.put(t)

                def _run() -> None:
                    self._run_prompt(text, channel_key, on_token=_on_token)
                    token_q.put(None)  # sentinel

                threading.Thread(target=_run, daemon=True).start()

                async def _stream():  # type: ignore[return]
                    loop = asyncio.get_event_loop()
                    while True:
                        token = await loop.run_in_executor(None, token_q.get)
                        if token is None:
                            break
                        yield f"data: {token}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(_stream(), media_type="text/event-stream")

            # Regular synchronous response
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, lambda: self._run_prompt(text, channel_key)
            )

            # Return session_id so the caller can continue the conversation
            actual_session = self._sessions.get(channel_key)
            sid = getattr(actual_session, "id", session_id or "")
            return JSONResponse({"text": response_text, "session_id": sid})

        config = uvicorn.Config(
            api,
            host=cfg.host,
            port=cfg.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        logger.info("[webhook] listening on %s:%s", cfg.host, cfg.port)
        self._server.run()

    def _stop(self) -> None:
        if self._server:
            self._server.should_exit = True
