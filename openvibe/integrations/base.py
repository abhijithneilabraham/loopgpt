"""Base classes for openvibe integrations."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

from openvibe.api import OpenVibe, SessionState

logger = logging.getLogger(__name__)


class BaseIntegration(ABC):
    """Base class for all messaging integrations.

    Subclasses implement ``_start()`` / ``_stop()`` which are called from a
    background thread by ``GatewayManager``.  The ``_run_prompt`` helper
    handles all openvibe session management.
    """

    name: str = "base"

    def __init__(self, ov: OpenVibe, auto_approve: bool = True, default_agent: str = "build") -> None:
        self._ov = ov
        self._auto_approve = auto_approve
        self._default_agent = default_agent
        # channel_key → session (kept alive between messages for context)
        self._sessions: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the integration in a background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._safe_start,
            name=f"openvibe-{self.name}",
            daemon=True,
        )
        self._thread.start()
        logger.info("[%s] integration started", self.name)

    def stop(self) -> None:
        """Signal the integration to stop."""
        self._running = False
        self._stop()
        logger.info("[%s] integration stopped", self.name)

    def _safe_start(self) -> None:
        try:
            self._start()
        except Exception:
            logger.exception("[%s] crashed", self.name)

    @abstractmethod
    def _start(self) -> None:
        """Blocking call that runs the integration event loop."""

    def _stop(self) -> None:
        """Override to perform clean shutdown."""

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self, channel_key: str) -> Any:
        """Return (or create) the openvibe session for a channel."""
        with self._lock:
            if channel_key not in self._sessions:
                self._sessions[channel_key] = self._ov.create_session(
                    agent=self._default_agent
                )
            return self._sessions[channel_key]

    # ------------------------------------------------------------------
    # Prompt execution
    # ------------------------------------------------------------------

    def _run_prompt(
        self,
        text: str,
        channel_key: str,
        on_token: Any | None = None,
    ) -> str:
        """Send *text* to the agent and return the full response text.

        Automatically handles permission requests (auto-approves when
        ``auto_approve=True``).
        """
        session = self._get_session(channel_key)
        collected: list[str] = []

        def _tok(t: str) -> None:
            collected.append(t)
            if on_token:
                on_token(t)

        response = session.send(text, on_token=_tok)

        # Handle permission / waiting states
        while response.state == SessionState.WAITING:
            req = response.request
            if self._auto_approve:
                choice = "allow"
            else:
                # In manual mode fall back to allow; subclass can override
                choice = "allow"
                logger.warning(
                    "[%s] permission requested for %s — auto-approving",
                    self.name,
                    req.description if req else "unknown",
                )
            response = session.reply(req.id, choice, on_token=_tok)

        if response.state == SessionState.ERROR and response.error:
            return f"Error: {response.error.message}"

        return "".join(collected) or response.text or ""
