"""ProcessRunner — executes openvibe against a prepared environment.

Runs the openvibe sync API with all permissions auto-approved (simulation
context). Returns the blueprint, response text, and execution stats.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class ProcessRunner:
    """Runs openvibe against a goal in a prepared directory."""

    def __init__(
        self,
        config: object | None = None,
        on_token: Callable[[str], None] | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        self._config = config
        self._on_token = on_token or (lambda _: None)
        self._timeout = timeout_seconds

    def execute(
        self, goal: str, tmp_dir: Path
    ) -> tuple[object | None, str, int, int, float]:
        """Run the agent against *goal* in *tmp_dir*.

        Returns (blueprint, agent_text, tool_calls, dag_branches, elapsed_seconds).
        """
        from openvibe.api import OpenVibe, SessionState
        from openvibe.config import load_config
        from openvibe.session.blueprint import get_blueprint

        config = self._config or load_config(tmp_dir)
        tokens: list[str] = []

        t0 = time.monotonic()
        with OpenVibe(project_dir=tmp_dir, config=config) as ov:
            ov.start()
            session = ov.create_session(agent="build")

            response = session.send(
                goal,
                on_token=lambda t: (tokens.append(t), self._on_token(t)),
            )

            # Auto-approve every permission request (simulation)
            depth = 0
            while response.state == SessionState.WAITING and depth < 30:
                depth += 1
                req = response.request
                response = session.reply(
                    req.id, "allow",
                    on_token=lambda t: (tokens.append(t), self._on_token(t)),
                )

            bp = get_blueprint(session.id)
            elapsed = time.monotonic() - t0

        tool_calls = len(bp.nodes) if bp else 0
        branches = bp.current_branch if bp else 0
        return bp, "".join(tokens), tool_calls, branches, elapsed
