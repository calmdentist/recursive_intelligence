"""SQLite-backed append-only event store with state projection."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class NodeState(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_ON_CHILDREN = "waiting_on_children"
    REVIEWING_CHILDREN = "reviewing_children"
    MERGING = "merging"
    VERIFYING = "verifying"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (NodeState.COMPLETED, NodeState.FAILED, NodeState.CANCELLED)

    @property
    def is_idle(self) -> bool:
        """Node is idle — either terminal or paused (waiting for external input)."""
        return self.is_terminal or self == NodeState.PAUSED


VALID_TRANSITIONS: dict[NodeState, set[NodeState]] = {
    NodeState.QUEUED: {NodeState.PLANNING, NodeState.CANCELLED},
    NodeState.PLANNING: {
        NodeState.EXECUTING,
        NodeState.WAITING_ON_CHILDREN,
        NodeState.PAUSED,
        NodeState.FAILED,
        NodeState.CANCELLED,
    },
    NodeState.EXECUTING: {
        NodeState.COMPLETED,
        NodeState.VERIFYING,
        NodeState.PAUSED,
        NodeState.WAITING_ON_CHILDREN,
        NodeState.FAILED,
        NodeState.CANCELLED,
    },
    NodeState.WAITING_ON_CHILDREN: {
        NodeState.REVIEWING_CHILDREN,
        NodeState.PAUSED,
        NodeState.FAILED,
        NodeState.CANCELLED,
    },
    NodeState.REVIEWING_CHILDREN: {
        NodeState.MERGING,
        NodeState.WAITING_ON_CHILDREN,  # revise loop
        NodeState.FAILED,
        NodeState.CANCELLED,
    },
    NodeState.MERGING: {
        NodeState.COMPLETED,
        NodeState.VERIFYING,
        NodeState.PAUSED,  # multi-pass: pause after merge
        NodeState.FAILED,
        NodeState.CANCELLED,
    },
    NodeState.VERIFYING: {
        NodeState.COMPLETED,
        NodeState.PLANNING,  # retry on test failure
        NodeState.PAUSED,    # persistent root passes verification
        NodeState.FAILED,    # retries exhausted
        NodeState.CANCELLED,
    },
    NodeState.PAUSED: {
        NodeState.PLANNING,  # resume with new input
        NodeState.EXECUTING,  # resume for follow-up work
        NodeState.WAITING_ON_CHILDREN,  # resume after forwarding a child request response
        NodeState.COMPLETED,  # user says "done"
        NodeState.CANCELLED,
    },
    NodeState.COMPLETED: {NodeState.EXECUTING},  # revise loop
    NodeState.FAILED: set(),
    NodeState.CANCELLED: set(),
}


@dataclass
class Event:
    event_id: str
    run_id: str
    node_id: str
    event_type: str
    timestamp: str
    data: dict[str, Any]


@dataclass
class NodeRecord:
    node_id: str
    run_id: str
    parent_id: str | None
    task_spec: str
    state: NodeState
    worktree_path: str | None
    branch_name: str | None
    session_id: str | None
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    repo_root: str
    task: str
    root_node_id: str | None
    created_at: str
    finished_at: str | None
    status: str  # "running", "paused", "completed", "failed"
    pass_count: int = 1
    persistent: bool = False
    test_command: str | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    delivery: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainRecord:
    domain_id: str
    run_id: str
    parent_node_id: str
    child_node_id: str
    domain_name: str
    file_patterns: list[str]
    module_scope: str
    created_at: str
    updated_at: str


class StateStore:
    """Append-only event store with projected node/run state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                repo_root TEXT NOT NULL,
                task TEXT NOT NULL,
                root_node_id TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                pass_count INTEGER NOT NULL DEFAULT 1,
                persistent INTEGER NOT NULL DEFAULT 0,
                test_command TEXT,
                telemetry TEXT NOT NULL DEFAULT '{}',
                delivery_state TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                parent_id TEXT REFERENCES nodes(node_id),
                task_spec TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'queued',
                worktree_path TEXT,
                branch_name TEXT,
                session_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                node_id TEXT NOT NULL REFERENCES nodes(node_id),
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES nodes(node_id),
                adapter TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                transcript_path TEXT,
                cost_json TEXT
            );

            CREATE TABLE IF NOT EXISTS domain_registry (
                domain_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                parent_node_id TEXT NOT NULL REFERENCES nodes(node_id),
                child_node_id TEXT NOT NULL REFERENCES nodes(node_id),
                domain_name TEXT NOT NULL,
                file_patterns TEXT NOT NULL DEFAULT '[]',
                module_scope TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(parent_node_id, domain_name)
            );

            CREATE INDEX IF NOT EXISTS idx_events_node ON events(node_id);
            CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_run ON nodes(run_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_state ON nodes(state);
            CREATE INDEX IF NOT EXISTS idx_domain_parent ON domain_registry(parent_node_id);
            CREATE INDEX IF NOT EXISTS idx_domain_child ON domain_registry(child_node_id);
        """)
        self._ensure_column("runs", "test_command", "TEXT")
        self._ensure_column("runs", "telemetry", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("runs", "delivery_state", "TEXT NOT NULL DEFAULT '{}'")
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self) -> None:
        self._conn.close()

    # --- Runs ---

    def create_run(
        self, repo_root: str, task: str, persistent: bool = False, test_command: str | None = None,
    ) -> RunRecord:
        run_id = _new_id("run")
        now = _now()
        telemetry = _default_run_telemetry()
        delivery = _default_run_delivery()
        self._conn.execute(
            "INSERT INTO runs (run_id, repo_root, task, created_at, status, persistent, test_command, telemetry, delivery_state) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)",
            (
                run_id,
                repo_root,
                task,
                now,
                1 if persistent else 0,
                test_command,
                json.dumps(telemetry),
                json.dumps(delivery),
            ),
        )
        self._conn.commit()
        return RunRecord(
            run_id=run_id, repo_root=repo_root, task=task, root_node_id=None,
            created_at=now, finished_at=None, status="running",
            pass_count=1, persistent=persistent, test_command=test_command,
            telemetry=telemetry, delivery=delivery,
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    def finish_run(self, run_id: str, status: str) -> None:
        now = _now()
        self._conn.execute(
            "UPDATE runs SET finished_at = ?, status = ? WHERE run_id = ?",
            (now, status, run_id),
        )
        self._conn.commit()

    def pause_run(self, run_id: str) -> None:
        now = _now()
        self._conn.execute(
            "UPDATE runs SET status = 'paused' WHERE run_id = ?", (run_id,),
        )
        self._conn.commit()

    def resume_paused_run(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE runs SET status = 'running', pass_count = pass_count + 1 WHERE run_id = ?",
            (run_id,),
        )
        self._conn.commit()

    def set_root_node(self, run_id: str, node_id: str) -> None:
        self._conn.execute(
            "UPDATE runs SET root_node_id = ? WHERE run_id = ?", (node_id, run_id),
        )
        self._conn.commit()

    def update_run_telemetry(self, run_id: str, **fields: int | float) -> None:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        telemetry = dict(_default_run_telemetry())
        telemetry.update(run.telemetry)
        telemetry.update(fields)
        self._conn.execute(
            "UPDATE runs SET telemetry = ? WHERE run_id = ?",
            (json.dumps(telemetry), run_id),
        )
        self._conn.commit()

    def increment_run_telemetry(self, run_id: str, **increments: int | float) -> None:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        telemetry = dict(_default_run_telemetry())
        telemetry.update(run.telemetry)
        for key, value in increments.items():
            current = telemetry.get(key, 0)
            telemetry[key] = current + value
        self._conn.execute(
            "UPDATE runs SET telemetry = ? WHERE run_id = ?",
            (json.dumps(telemetry), run_id),
        )
        self._conn.commit()

    def get_run_delivery(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        return dict(run.delivery)

    def update_run_delivery(self, run_id: str, delivery: dict[str, Any]) -> None:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        merged = _default_run_delivery()
        merged.update(run.delivery)
        merged.update(delivery)
        self._conn.execute(
            "UPDATE runs SET delivery_state = ? WHERE run_id = ?",
            (json.dumps(merged), run_id),
        )
        self._conn.commit()

    def record_preview(
        self,
        run_id: str,
        url: str,
        *,
        label: str = "preview",
        status: str = "ready",
        note: str = "",
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        delivery = dict(run.delivery)
        previews = list(delivery.get("previews", []))
        now = _now()
        preview = {
            "preview_id": _new_id("preview"),
            "label": label,
            "url": url,
            "status": status,
            "note": note,
            "created_at": now,
            "updated_at": now,
        }
        previews.insert(0, preview)
        delivery["previews"] = previews
        self.update_run_delivery(run_id, delivery)
        return preview

    def record_deployment(
        self,
        run_id: str,
        environment: str,
        url: str,
        *,
        status: str = "deployed",
        note: str = "",
        verification_status: str = "unknown",
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        delivery = dict(run.delivery)
        deployments = list(delivery.get("deployments", []))
        now = _now()
        deployment = {
            "deployment_id": _new_id("deploy"),
            "environment": environment,
            "url": url,
            "status": status,
            "verification_status": verification_status,
            "note": note,
            "created_at": now,
            "updated_at": now,
        }
        deployments.insert(0, deployment)
        delivery["deployments"] = deployments
        self.update_run_delivery(run_id, delivery)
        return deployment

    def set_release_status(self, run_id: str, status: str, note: str = "") -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        delivery = dict(run.delivery)
        release = dict(delivery.get("release", {}))
        release.update({
            "status": status,
            "note": note,
            "updated_at": _now(),
        })
        delivery["release"] = release
        self.update_run_delivery(run_id, delivery)
        return release

    def add_delivery_blocker(
        self,
        run_id: str,
        *,
        kind: str,
        summary: str,
        details: str = "",
        action_requested: str = "",
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        delivery = dict(run.delivery)
        blockers = list(delivery.get("blockers", []))
        blocker = {
            "blocker_id": _new_id("delivery"),
            "kind": kind,
            "summary": summary,
            "details": details,
            "action_requested": action_requested,
            "created_at": _now(),
            "resolved_at": None,
            "resolution_note": "",
        }
        blockers.insert(0, blocker)
        delivery["blockers"] = blockers
        self.update_run_delivery(run_id, delivery)
        return blocker

    def resolve_delivery_blocker(self, run_id: str, blocker_id: str, note: str = "") -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        delivery = dict(run.delivery)
        blockers = list(delivery.get("blockers", []))
        for blocker in blockers:
            if blocker.get("blocker_id") != blocker_id:
                continue
            blocker["resolved_at"] = _now()
            blocker["resolution_note"] = note
            delivery["blockers"] = blockers
            self.update_run_delivery(run_id, delivery)
            return blocker
        raise ValueError(f"Delivery blocker {blocker_id} not found")

    # --- Nodes ---

    def create_node(
        self,
        run_id: str,
        task_spec: str,
        parent_id: str | None = None,
        worktree_path: str | None = None,
        branch_name: str | None = None,
    ) -> NodeRecord:
        node_id = _new_id("node")
        now = _now()
        self._conn.execute(
            """INSERT INTO nodes
               (node_id, run_id, parent_id, task_spec, state, worktree_path, branch_name, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)""",
            (node_id, run_id, parent_id, task_spec, worktree_path, branch_name, now, now),
        )
        self._append_event(run_id, node_id, "node_created", {
            "parent_id": parent_id, "task_spec": task_spec,
        })
        self._conn.commit()
        return NodeRecord(
            node_id=node_id, run_id=run_id, parent_id=parent_id, task_spec=task_spec,
            state=NodeState.QUEUED, worktree_path=worktree_path, branch_name=branch_name,
            session_id=None, created_at=now, updated_at=now,
        )

    def get_node(self, node_id: str) -> NodeRecord | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        return _row_to_node(row)

    def get_children(self, node_id: str) -> list[NodeRecord]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE parent_id = ? ORDER BY created_at", (node_id,)
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def get_run_nodes(self, run_id: str) -> list[NodeRecord]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def transition_node(self, node_id: str, new_state: NodeState, data: dict[str, Any] | None = None) -> NodeRecord:
        node = self.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        current = node.state
        if new_state not in VALID_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid transition: {current.value} -> {new_state.value}")

        now = _now()
        self._conn.execute(
            "UPDATE nodes SET state = ?, updated_at = ? WHERE node_id = ?",
            (new_state.value, now, node_id),
        )
        self._append_event(node.run_id, node_id, "state_transition", {
            "from": current.value, "to": new_state.value, **(data or {}),
        })
        self._conn.commit()
        node.state = new_state
        node.updated_at = now
        return node

    def update_node(self, node_id: str, **fields: Any) -> None:
        allowed = {"worktree_path", "branch_name", "session_id", "metadata"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        if "metadata" in updates:
            updates["metadata"] = json.dumps(updates["metadata"])
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [_now(), node_id]
        self._conn.execute(
            f"UPDATE nodes SET {set_clause}, updated_at = ? WHERE node_id = ?", values,
        )
        self._conn.commit()

    def get_nodes_in_state(self, run_id: str, state: NodeState) -> list[NodeRecord]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE run_id = ? AND state = ?", (run_id, state.value)
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    # --- Events ---

    def _append_event(self, run_id: str, node_id: str, event_type: str, data: dict[str, Any]) -> str:
        event_id = _new_id("evt")
        now = _now()
        self._conn.execute(
            "INSERT INTO events (event_id, run_id, node_id, event_type, timestamp, data) VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, run_id, node_id, event_type, now, json.dumps(data)),
        )
        return event_id

    def append_event(self, run_id: str, node_id: str, event_type: str, data: dict[str, Any]) -> str:
        eid = self._append_event(run_id, node_id, event_type, data)
        self._conn.commit()
        return eid

    def get_node_events(self, node_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE node_id = ? ORDER BY timestamp", (node_id,)
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def get_run_events(self, run_id: str) -> list[Event]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY timestamp", (run_id,)
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def get_latest_blocker(self, node_id: str) -> dict[str, Any] | None:
        """Return the latest structured blocker/escalation for a node."""
        request = self.get_latest_request(node_id)
        if request is not None:
            blocker = request["request"]
            return {
                **request,
                "blocker": {
                    "kind": blocker.get("kind", "request"),
                    "details": blocker.get("details", ""),
                    "urgency": blocker.get("urgency", "normal"),
                    "escalation": {
                        "summary": blocker.get("summary", ""),
                        "details": blocker.get("details", ""),
                        "action_requested": blocker.get("action_requested", ""),
                    },
                },
            }
        return None

    def get_latest_request(self, node_id: str) -> dict[str, Any] | None:
        """Return the latest structured upstream request for a node."""
        events = self.get_node_events(node_id)
        for event in reversed(events):
            if event.event_type in {
                "root_request_upstream",
                "request_forwarded_upstream",
                "request_upstream",
                "root_escalation_requested",
                "child_escalation_forwarded",
                "user_escalation_requested",
            }:
                request = event.data.get("request")
                if request:
                    return {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "request": request,
                        "source_child_id": event.data.get("source_child_id"),
                        "source_task_spec": event.data.get("source_task_spec"),
                    }
                blocker = event.data.get("blocker")
                if blocker:
                    return {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "request": {
                            "kind": blocker.get("kind", "request"),
                            "summary": blocker.get("escalation", {}).get("summary", blocker.get("details", "")),
                            "details": blocker.get("details", ""),
                            "action_requested": blocker.get("escalation", {}).get("action_requested", ""),
                            "requires_input": blocker.get("needs_user_input", False),
                            "urgency": blocker.get("urgency", "normal"),
                        },
                        "source_child_id": event.data.get("source_child_id"),
                        "source_task_spec": event.data.get("source_task_spec"),
                    }
            if event.event_type == "execution_result":
                raw = event.data.get("raw", {})
                request = raw.get("request")
                if raw.get("status") == "request_upstream" and request:
                    return {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "request": request,
                        "source_child_id": None,
                        "source_task_spec": None,
                    }
                blocker = raw.get("blocker")
                if raw.get("status") == "blocked" and blocker:
                    return {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "request": {
                            "kind": blocker.get("kind", "request"),
                            "summary": blocker.get("escalation", {}).get("summary", blocker.get("details", "")),
                            "details": blocker.get("details", ""),
                            "action_requested": blocker.get("escalation", {}).get("action_requested", ""),
                            "requires_input": blocker.get("needs_user_input", False),
                            "urgency": blocker.get("urgency", "normal"),
                        },
                        "source_child_id": None,
                        "source_task_spec": None,
                    }
            if event.event_type == "state_transition":
                request = event.data.get("request")
                if request:
                    return {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "request": request,
                        "source_child_id": event.data.get("source_child_id"),
                        "source_task_spec": None,
                    }
                blocker = event.data.get("blocker")
                if blocker:
                    return {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "request": {
                            "kind": blocker.get("kind", "request"),
                            "summary": blocker.get("escalation", {}).get("summary", blocker.get("details", "")),
                            "details": blocker.get("details", ""),
                            "action_requested": blocker.get("escalation", {}).get("action_requested", ""),
                            "requires_input": blocker.get("needs_user_input", False),
                            "urgency": blocker.get("urgency", "normal"),
                        },
                        "source_child_id": event.data.get("source_child_id"),
                        "source_task_spec": None,
                    }
        return None

    def get_run_blockers(self, run_id: str) -> list[dict[str, Any]]:
        """Return active blockers for paused/failed nodes in a run."""
        requests = self.get_run_requests(run_id)
        blockers: list[dict[str, Any]] = []
        for request in requests:
            req = request["request"]
            blockers.append({
                **request,
                "blocker": {
                    "kind": req.get("kind", "request"),
                    "details": req.get("details", ""),
                    "urgency": req.get("urgency", "normal"),
                    "escalation": {
                        "summary": req.get("summary", ""),
                        "details": req.get("details", ""),
                        "action_requested": req.get("action_requested", ""),
                    },
                },
            })
        return blockers

    def get_run_requests(self, run_id: str) -> list[dict[str, Any]]:
        """Return active upstream requests for paused/failed nodes in a run."""
        blockers: list[dict[str, Any]] = []
        for node in self.get_run_nodes(run_id):
            if node.state not in {NodeState.PAUSED, NodeState.FAILED}:
                continue
            latest = self.get_latest_request(node.node_id)
            if latest is None:
                continue
            domain = self.get_domain_by_child(node.node_id)
            blockers.append({
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "task_spec": node.task_spec,
                "state": node.state.value,
                "domain_name": domain.domain_name if domain else None,
                **latest,
            })
        blockers.sort(key=lambda item: item["timestamp"], reverse=True)
        return blockers

    def get_latest_downstream_task(self, node_id: str) -> dict[str, Any] | None:
        """Return the latest unresolved downstream work item for a node."""
        pending: dict[str, Any] | None = None
        for event in self.get_node_events(node_id):
            if event.event_type == "downstream_task":
                pending = {
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "task": event.data,
                }
            elif event.event_type == "reactivation_requested":
                pending = {
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "task": {
                        "kind": "reactivation",
                        "summary": "Follow-up work routed from parent",
                        "task_spec": event.data.get("new_task", ""),
                        "previous_summary": event.data.get("previous_summary", ""),
                        "original_task": event.data.get("original_task", ""),
                    },
                }
            elif event.event_type == "revision_requested":
                pending = {
                    "event_type": event.event_type,
                    "timestamp": event.timestamp,
                    "task": {
                        "kind": "revision",
                        "summary": "Review requested changes",
                        "task_spec": event.data.get("follow_up", ""),
                    },
                }
            elif event.event_type in {"execution_result", "downstream_task_result"}:
                pending = None
        return pending

    def get_run_downstream_tasks(self, run_id: str) -> list[dict[str, Any]]:
        """Return unresolved downstream work items across the run tree."""
        tasks: list[dict[str, Any]] = []
        for node in self.get_run_nodes(run_id):
            latest = self.get_latest_downstream_task(node.node_id)
            if latest is None:
                continue
            domain = self.get_domain_by_child(node.node_id)
            tasks.append({
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "task_spec": node.task_spec,
                "state": node.state.value,
                "domain_name": domain.domain_name if domain else None,
                **latest,
            })
        tasks.sort(key=lambda item: item["timestamp"], reverse=True)
        return tasks

    def get_recent_run_results(self, run_id: str, limit: int = 8) -> list[dict[str, Any]]:
        """Return recent completed result events across the run tree."""
        node_cache: dict[str, NodeRecord | None] = {}
        domain_cache: dict[str, DomainRecord | None] = {}
        results: list[dict[str, Any]] = []

        def _node(node_id: str) -> NodeRecord | None:
            if node_id not in node_cache:
                node_cache[node_id] = self.get_node(node_id)
            return node_cache[node_id]

        def _domain_name(node_id: str) -> str | None:
            if node_id not in domain_cache:
                domain_cache[node_id] = self.get_domain_by_child(node_id)
            domain = domain_cache[node_id]
            return domain.domain_name if domain else None

        for event in self.get_run_events(run_id):
            node = _node(event.node_id)
            if node is None:
                continue

            if event.event_type == "execution_result":
                raw = event.data.get("raw", {})
                status = raw.get("status", "")
                if status in {"request_upstream", "blocked"}:
                    continue
                summary = raw.get("summary") or event.data.get("summary", "")
                if not summary and not status:
                    continue
                results.append({
                    "timestamp": event.timestamp,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "task_spec": node.task_spec,
                    "state": node.state.value,
                    "domain_name": _domain_name(node.node_id),
                    "kind": "execution",
                    "status": status or "completed",
                    "summary": summary or status,
                })
            elif event.event_type == "downstream_task_result":
                results.append({
                    "timestamp": event.timestamp,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "task_spec": node.task_spec,
                    "state": node.state.value,
                    "domain_name": _domain_name(node.node_id),
                    "kind": event.data.get("kind", "task"),
                    "status": event.data.get("status", "completed"),
                    "summary": event.data.get("summary") or event.data.get("reason", ""),
                })
            elif event.event_type == "verification_result":
                passed = bool(event.data.get("passed"))
                results.append({
                    "timestamp": event.timestamp,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "task_spec": node.task_spec,
                    "state": node.state.value,
                    "domain_name": _domain_name(node.node_id),
                    "kind": "verification",
                    "status": "passed" if passed else "failed",
                    "summary": "Verification passed" if passed else "Verification failed",
                })

        results.sort(key=lambda item: item["timestamp"], reverse=True)
        return results[:limit]

    def get_node_recent_results(self, node_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return recent completed result events for a single node."""
        node = self.get_node(node_id)
        if node is None:
            return []
        domain = self.get_domain_by_child(node_id)
        domain_name = domain.domain_name if domain else None
        results: list[dict[str, Any]] = []

        for event in self.get_node_events(node_id):
            if event.event_type == "execution_result":
                raw = event.data.get("raw", {})
                status = raw.get("status", "")
                if status in {"request_upstream", "blocked"}:
                    continue
                summary = raw.get("summary") or event.data.get("summary", "")
                if not summary and not status:
                    continue
                results.append({
                    "timestamp": event.timestamp,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "task_spec": node.task_spec,
                    "state": node.state.value,
                    "domain_name": domain_name,
                    "kind": "execution",
                    "status": status or "completed",
                    "summary": summary or status,
                })
            elif event.event_type == "downstream_task_result":
                results.append({
                    "timestamp": event.timestamp,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "task_spec": node.task_spec,
                    "state": node.state.value,
                    "domain_name": domain_name,
                    "kind": event.data.get("kind", "task"),
                    "status": event.data.get("status", "completed"),
                    "summary": event.data.get("summary") or event.data.get("reason", ""),
                })
            elif event.event_type == "verification_result":
                passed = bool(event.data.get("passed"))
                results.append({
                    "timestamp": event.timestamp,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "task_spec": node.task_spec,
                    "state": node.state.value,
                    "domain_name": domain_name,
                    "kind": "verification",
                    "status": "passed" if passed else "failed",
                    "summary": "Verification passed" if passed else "Verification failed",
                })

        results.sort(key=lambda item: item["timestamp"], reverse=True)
        return results[:limit]

    def get_run_work_board(self, run_id: str, recent_limit: int = 8) -> dict[str, Any]:
        """Return a grouped root work board projection for a run."""
        run = self.get_run(run_id)
        nodes = self.get_run_nodes(run_id)
        completed_count = sum(1 for node in nodes if node.state == NodeState.COMPLETED)
        failed_count = sum(1 for node in nodes if node.state == NodeState.FAILED)
        active_count = sum(1 for node in nodes if not node.state.is_idle)
        inbox = self._attach_board_handles(self.get_run_inbox(run_id), prefix="i")
        downstream_tasks = self._attach_board_handles(self.get_run_downstream_tasks(run_id), prefix="w")
        recent_results = self._attach_board_handles(
            self.get_recent_run_results(run_id, limit=recent_limit),
            prefix="r",
        )
        return {
            "run": run,
            "nodes": nodes,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "active_count": active_count,
            "inbox": inbox,
            "downstream_tasks": downstream_tasks,
            "recent_results": recent_results,
        }

    def get_run_readiness(self, run_id: str, recent_limit: int = 8) -> dict[str, Any]:
        """Return a merge-readiness summary for a run."""
        board = self.get_run_work_board(run_id, recent_limit=max(recent_limit, 8))
        run = board["run"]
        nodes = board["nodes"]
        inbox = board["inbox"]
        downstream_tasks = board["downstream_tasks"]
        failed_nodes = [node for node in nodes if node.state == NodeState.FAILED]
        paused_nodes = [node for node in nodes if node.state == NodeState.PAUSED]
        recent_failures = [
            item for item in board["recent_results"]
            if item.get("status") in {"failed", "error"}
        ][:recent_limit]
        delivery = self.get_run_delivery(run_id) if run is not None else _default_run_delivery()
        release = dict(delivery.get("release", {}))
        delivery_blockers = [
            blocker for blocker in delivery.get("blockers", [])
            if not blocker.get("resolved_at")
        ]

        ownership_violations = 0
        if run is not None:
            ownership_violations = int(run.telemetry.get("ownership_violations_count", 0))

        blockers: list[dict[str, Any]] = []
        if inbox:
            blockers.append({
                "kind": "inbox_requests",
                "count": len(inbox),
                "message": f"{len(inbox)} unresolved root request(s)",
                "action": "/board",
            })
        if downstream_tasks:
            blockers.append({
                "kind": "downstream_tasks",
                "count": len(downstream_tasks),
                "message": f"{len(downstream_tasks)} unresolved downstream task(s)",
                "action": "/board",
            })
        if failed_nodes:
            blockers.append({
                "kind": "failed_nodes",
                "count": len(failed_nodes),
                "message": f"{len(failed_nodes)} failed node(s)",
                "action": "/tree",
            })
        if recent_failures:
            blockers.append({
                "kind": "recent_failures",
                "count": len(recent_failures),
                "message": f"{len(recent_failures)} recent failed result(s)",
                "action": "/inspect",
            })
        if ownership_violations:
            blockers.append({
                "kind": "ownership_violations",
                "count": ownership_violations,
                "message": f"{ownership_violations} ownership violation(s)",
                "action": "/inspect",
            })
        if delivery_blockers:
            blockers.append({
                "kind": "delivery_blockers",
                "count": len(delivery_blockers),
                "message": f"{len(delivery_blockers)} unresolved delivery blocker(s)",
                "action": "/delivery",
            })
        if release.get("status") == "blocked":
            blockers.append({
                "kind": "release_blocked",
                "count": 1,
                "message": "release status is blocked",
                "action": "/delivery",
            })

        ready = (
            run is not None
            and run.status == "completed"
            and not inbox
            and not downstream_tasks
            and not failed_nodes
            and not recent_failures
            and ownership_violations == 0
            and not delivery_blockers
            and release.get("status") != "blocked"
        )
        return {
            "run": run,
            "nodes": nodes,
            "ready": ready,
            "inbox_count": len(inbox),
            "downstream_task_count": len(downstream_tasks),
            "failed_node_count": len(failed_nodes),
            "paused_node_count": len(paused_nodes),
            "ownership_violations_count": ownership_violations,
            "delivery": delivery,
            "delivery_blocker_count": len(delivery_blockers),
            "recent_failures": recent_failures,
            "failed_nodes": failed_nodes,
            "blockers": blockers,
        }

    def get_run_board_item(self, run_id: str, handle: str, recent_limit: int = 8) -> dict[str, Any] | None:
        """Return a board item by short handle such as i1, w2, or r3."""
        handle = handle.strip().lower()
        if not handle:
            return None
        board = self.get_run_work_board(run_id, recent_limit=recent_limit)
        for section in ("inbox", "downstream_tasks", "recent_results"):
            for item in board[section]:
                if item.get("board_handle", "").lower() == handle:
                    return item
        return None

    @staticmethod
    def _attach_board_handles(items: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
        handled: list[dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            enriched = dict(item)
            enriched["board_handle"] = f"{prefix}{idx}"
            handled.append(enriched)
        return handled

    def get_run_inbox(self, run_id: str) -> list[dict[str, Any]]:
        """Return unresolved root-facing requests for a run."""
        requests_by_id: dict[str, dict[str, Any]] = {}
        resolved_ids: set[str] = set()

        for event in self.get_run_events(run_id):
            if event.event_type == "root_request_upstream":
                request = event.data.get("request")
                if not request:
                    continue
                request_id = request.get("request_id")
                if not request_id:
                    continue
                source_child_id = event.data.get("source_child_id")
                domain_name = None
                if source_child_id:
                    domain = self.get_domain_by_child(source_child_id)
                    domain_name = domain.domain_name if domain else None
                source_task_spec = None
                if source_child_id:
                    source_node = self.get_node(source_child_id)
                    source_task_spec = source_node.task_spec if source_node else None
                requests_by_id[request_id] = {
                    "request_id": request_id,
                    "timestamp": event.timestamp,
                    "request": request,
                    "source_child_id": source_child_id,
                    "source_task_spec": source_task_spec,
                    "domain_name": domain_name,
                    "node_id": event.node_id,
                }
            elif event.event_type == "request_resolved":
                request_id = event.data.get("request_id")
                if request_id:
                    resolved_ids.add(request_id)

        inbox = [
            item for request_id, item in requests_by_id.items()
            if request_id not in resolved_ids
        ]
        inbox.sort(key=lambda item: item["timestamp"], reverse=True)
        return inbox

    def get_inbox_request(self, run_id: str, request_id: str) -> dict[str, Any] | None:
        """Return a single unresolved root-facing request by id."""
        for item in self.get_run_inbox(run_id):
            if item["request_id"] == request_id:
                return item
        return None

    # --- Sessions ---

    def create_session(self, session_id: str, node_id: str, adapter: str) -> None:
        now = _now()
        self._conn.execute(
            "INSERT INTO sessions (session_id, node_id, adapter, started_at) VALUES (?, ?, ?, ?)",
            (session_id, node_id, adapter, now),
        )
        self._conn.commit()

    def session_exists(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row is not None

    def finish_session(self, session_id: str, transcript_path: str | None = None, cost_json: str | None = None) -> None:
        now = _now()
        self._conn.execute(
            "UPDATE sessions SET finished_at = ?, transcript_path = ?, cost_json = ? WHERE session_id = ?",
            (now, transcript_path, cost_json, session_id),
        )
        self._conn.commit()

    # --- Domain Registry ---

    def register_domain(
        self,
        run_id: str,
        parent_node_id: str,
        child_node_id: str,
        domain_name: str,
        file_patterns: list[str] | None = None,
        module_scope: str = "",
    ) -> DomainRecord:
        domain_id = _new_id("dom")
        now = _now()
        self._conn.execute(
            """INSERT OR REPLACE INTO domain_registry
               (domain_id, run_id, parent_node_id, child_node_id, domain_name, file_patterns, module_scope, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (domain_id, run_id, parent_node_id, child_node_id, domain_name,
             json.dumps(file_patterns or []), module_scope, now, now),
        )
        self._conn.commit()
        return DomainRecord(
            domain_id=domain_id, run_id=run_id, parent_node_id=parent_node_id,
            child_node_id=child_node_id, domain_name=domain_name,
            file_patterns=file_patterns or [], module_scope=module_scope,
            created_at=now, updated_at=now,
        )

    def get_domains(self, parent_node_id: str) -> list[DomainRecord]:
        rows = self._conn.execute(
            "SELECT * FROM domain_registry WHERE parent_node_id = ? ORDER BY created_at",
            (parent_node_id,),
        ).fetchall()
        return [_row_to_domain(r) for r in rows]

    def get_domain_by_child(self, child_node_id: str) -> DomainRecord | None:
        row = self._conn.execute(
            "SELECT * FROM domain_registry WHERE child_node_id = ?", (child_node_id,),
        ).fetchone()
        return _row_to_domain(row) if row else None

    def update_domain(self, domain_id: str, **fields: Any) -> None:
        allowed = {"file_patterns", "module_scope", "domain_name"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        if "file_patterns" in updates:
            updates["file_patterns"] = json.dumps(updates["file_patterns"])
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [_now(), domain_id]
        self._conn.execute(
            f"UPDATE domain_registry SET {set_clause}, updated_at = ? WHERE domain_id = ?", values,
        )
        self._conn.commit()

    # --- Child spawn idempotency ---

    def child_spawn_key_exists(self, parent_id: str, child_slot: str, task_hash: str) -> bool:
        rows = self._conn.execute(
            """SELECT 1 FROM events
               WHERE node_id = ? AND event_type = 'child_spawned'
               AND json_extract(data, '$.child_slot') = ?
               AND json_extract(data, '$.task_hash') = ?""",
            (parent_id, child_slot, task_hash),
        ).fetchone()
        return rows is not None


# --- Helpers ---

def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_node(row: sqlite3.Row) -> NodeRecord:
    return NodeRecord(
        node_id=row["node_id"], run_id=row["run_id"], parent_id=row["parent_id"],
        task_spec=row["task_spec"], state=NodeState(row["state"]),
        worktree_path=row["worktree_path"], branch_name=row["branch_name"],
        session_id=row["session_id"], created_at=row["created_at"],
        updated_at=row["updated_at"], metadata=json.loads(row["metadata"]),
    )


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    # test_command column may not exist in older databases
    test_command = None
    telemetry: dict[str, Any] = {}
    delivery: dict[str, Any] = {}
    try:
        test_command = row["test_command"]
    except (IndexError, KeyError):
        pass
    try:
        raw_telemetry = row["telemetry"]
        telemetry = json.loads(raw_telemetry) if raw_telemetry else {}
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        telemetry = {}
    try:
        raw_delivery = row["delivery_state"]
        delivery = json.loads(raw_delivery) if raw_delivery else {}
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        delivery = {}
    merged_telemetry = _default_run_telemetry()
    merged_telemetry.update(telemetry)
    merged_delivery = _default_run_delivery()
    merged_delivery.update(delivery)
    return RunRecord(
        run_id=row["run_id"], repo_root=row["repo_root"], task=row["task"],
        root_node_id=row["root_node_id"], created_at=row["created_at"],
        finished_at=row["finished_at"], status=row["status"],
        pass_count=row["pass_count"], persistent=bool(row["persistent"]),
        test_command=test_command, telemetry=merged_telemetry, delivery=merged_delivery,
    )


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        event_id=row["event_id"], run_id=row["run_id"], node_id=row["node_id"],
        event_type=row["event_type"], timestamp=row["timestamp"],
        data=json.loads(row["data"]),
    )


def _row_to_domain(row: sqlite3.Row) -> DomainRecord:
    return DomainRecord(
        domain_id=row["domain_id"], run_id=row["run_id"],
        parent_node_id=row["parent_node_id"], child_node_id=row["child_node_id"],
        domain_name=row["domain_name"], file_patterns=json.loads(row["file_patterns"]),
        module_scope=row["module_scope"], created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _default_run_telemetry() -> dict[str, Any]:
    return {
        "user_interruptions_count": 0,
        "root_escalations_count": 0,
        "root_requests_count": 0,
        "resolved_requests_count": 0,
        "human_inputs_count": 0,
        "ownership_violations_count": 0,
    }


def _default_run_delivery() -> dict[str, Any]:
    return {
        "release": {
            "status": "draft",
            "note": "",
            "updated_at": None,
        },
        "previews": [],
        "deployments": [],
        "blockers": [],
    }
