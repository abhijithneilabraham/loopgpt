"""Process Blueprint — the step-by-step execution graph of a session.

Each tool call the agent makes becomes a node in the graph.  The human-readable
"purpose" for each step is extracted from the text the agent emits just before
calling the tool — zero extra tokens.

The blueprint auto-saves to ``<session_dir>/process-blueprint.json`` after
each step.  Load it into a new session to resume exactly where the process
left off.

Graph properties
----------------
- Directed acyclic graph (DAG): each node can have multiple parents/children.
- Branching: when the agent undoes a step and tries another approach, the new
  steps live on a new branch while the old node is marked ``undone``.
- Undo/redo: nodes can be marked ``undone``; the current pointer walks back.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

NodeStatus = Literal["running", "completed", "failed", "undone"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProcessNode:
    """One step in the process graph."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    tool_name: str = ""
    # Brief action label — "searched for supplier portal", "clicked Login button"
    action: str = ""
    # Why this step is being done in the context of the main goal
    purpose: str = ""
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    status: NodeStatus = "running"
    output_summary: str = ""
    branch: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessBlueprint:
    """Complete process graph for one session.

    Export with ``/blueprint export``.  Paste the file contents into a new
    session — the agent will read the completed steps and resume the process
    from the last successful point.
    """

    version: int = 1
    session_id: str = ""
    goal: str = ""
    nodes: dict[str, ProcessNode] = field(default_factory=dict)
    current_node_id: str | None = None
    current_branch: int = 0

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_node(
        self,
        tool_name: str,
        action: str,
        purpose: str,
        parent_id: str | None = None,
        branch: int | None = None,
    ) -> ProcessNode:
        parent_ids = [parent_id] if parent_id else []
        node = ProcessNode(
            tool_name=tool_name,
            action=action,
            purpose=purpose,
            parent_ids=parent_ids,
            branch=branch if branch is not None else self.current_branch,
        )
        self.nodes[node.id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].child_ids.append(node.id)
        self.current_node_id = node.id
        return node

    def complete_node(self, node_id: str, output_summary: str = "") -> None:
        if node := self.nodes.get(node_id):
            node.status = "completed"
            node.output_summary = output_summary[:200]

    def fail_node(self, node_id: str, error: str = "") -> None:
        if node := self.nodes.get(node_id):
            node.status = "failed"
            node.output_summary = error[:200]

    def undo_node(self, node_id: str) -> None:
        """Mark a node undone and start a new branch from its parent."""
        if node := self.nodes.get(node_id):
            node.status = "undone"
            self.current_branch += 1
            self.current_node_id = node.parent_ids[0] if node.parent_ids else None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_export_dict(self) -> dict:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "goal": self.goal,
            "steps": [
                {
                    "id": n.id,
                    "tool": n.tool_name,
                    "action": n.action,
                    "purpose": n.purpose,
                    "status": n.status,
                    "output_summary": n.output_summary,
                    "branch": n.branch,
                    "parent_ids": n.parent_ids,
                    "child_ids": n.child_ids,
                }
                for n in self.nodes.values()
            ],
        }

    def save(self, session_dir: str | Path) -> Path:
        path = Path(session_dir) / "process-blueprint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_export_dict(), indent=2), encoding="utf-8")
        return path

    def resume_context(self) -> str:
        """Minimal summary to prepend to a new session for seamless resumption."""
        completed = [n for n in self.nodes.values() if n.status == "completed"]
        if not completed:
            return ""
        lines = [
            f"Resuming process: {self.goal}",
            "Already completed:",
        ]
        for n in completed[-10:]:  # cap at 10 to save tokens
            lines.append(f"  [{n.tool_name}] {n.action} — {n.purpose}")
        lines.append("Continue from the next step.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Purpose extraction (zero extra tokens)
# ---------------------------------------------------------------------------


def extract_purpose(text: str) -> str:
    """Extract a brief purpose from the LLM's preceding text.

    Uses the last meaningful sentence of whatever text the agent emitted
    before calling the tool.  No extra LLM call needed.
    """
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text.strip())
    # Split on sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    purpose = sentences[-1] if sentences else cleaned
    if len(purpose) > 150:
        purpose = purpose[:147] + "…"
    return purpose


def build_action_label(tool_name: str, args: dict) -> str:
    """Return a brief human-readable label for what the tool is doing."""
    _LABEL_MAP = {
        "bash": ("command", lambda v: f"running `{v[:50]}`"),
        "web_search": ("query", lambda v: f"searching for '{v[:50]}'"),
        "web_fetch": ("url", lambda v: f"fetching {v[:50]}"),
        "file_read": ("path", lambda v: f"reading {v}"),
        "file_write": ("path", lambda v: f"writing {v}"),
        "file_edit": ("path", lambda v: f"editing {v}"),
        "mouse_click": (None, lambda _: "clicking"),
        "mouse_move": (None, lambda _: "moving mouse"),
        "keyboard_type": ("text", lambda v: f"typing '{v[:30]}'"),
        "screenshot": (None, lambda _: "taking screenshot"),
        "app_open": ("name", lambda v: f"opening {v}"),
        "pre_flight": (None, lambda _: "checking permissions"),
    }
    if tool_name in _LABEL_MAP:
        key, fmt = _LABEL_MAP[tool_name]
        val = args.get(key, "") if key else ""
        return fmt(str(val))
    if args:
        first_val = str(next(iter(args.values())))[:40]
        return f"{tool_name}: {first_val}"
    return tool_name


# ---------------------------------------------------------------------------
# Session-scoped registry
# ---------------------------------------------------------------------------

_blueprints: dict[str, ProcessBlueprint] = {}


def get_blueprint(session_id: str) -> ProcessBlueprint | None:
    return _blueprints.get(session_id)


def init_blueprint(session_id: str, goal: str) -> ProcessBlueprint:
    if session_id not in _blueprints:
        _blueprints[session_id] = ProcessBlueprint(session_id=session_id, goal=goal)
    elif goal and not _blueprints[session_id].goal:
        _blueprints[session_id].goal = goal
    return _blueprints[session_id]
