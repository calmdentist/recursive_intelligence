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
        NodeState.PAUSED,
        NodeState.WAITING_ON_CHILDREN,
        NodeState.FAILED,
        NodeState.CANCELLED,
    },
    NodeState.WAITING_ON_CHILDREN: {
        NodeState.REVIEWING_CHILDREN,
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
        NodeState.PAUSED,  # multi-pass: pause after merge
        NodeState.FAILED,
        NodeState.CANCELLED,
    },
    NodeState.PAUSED: {
        NodeState.PLANNING,  # resume with new input
        NodeState.EXECUTING,  # resume for follow-up work
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
                persistent INTEGER NOT NULL DEFAULT 0
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
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- Runs ---

    def create_run(self, repo_root: str, task: str, persistent: bool = False) -> RunRecord:
        run_id = _new_id("run")
        now = _now()
        self._conn.execute(
            "INSERT INTO runs (run_id, repo_root, task, created_at, status, persistent) VALUES (?, ?, ?, ?, 'running', ?)",
            (run_id, repo_root, task, now, 1 if persistent else 0),
        )
        self._conn.commit()
        return RunRecord(
            run_id=run_id, repo_root=repo_root, task=task, root_node_id=None,
            created_at=now, finished_at=None, status="running",
            pass_count=1, persistent=persistent,
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

    # --- Sessions ---

    def create_session(self, session_id: str, node_id: str, adapter: str) -> None:
        now = _now()
        self._conn.execute(
            "INSERT INTO sessions (session_id, node_id, adapter, started_at) VALUES (?, ?, ?, ?)",
            (session_id, node_id, adapter, now),
        )
        self._conn.commit()

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
    return RunRecord(
        run_id=row["run_id"], repo_root=row["repo_root"], task=row["task"],
        root_node_id=row["root_node_id"], created_at=row["created_at"],
        finished_at=row["finished_at"], status=row["status"],
        pass_count=row["pass_count"], persistent=bool(row["persistent"]),
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
