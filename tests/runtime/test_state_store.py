"""Tests for the SQLite state store."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from recursive_intelligence.runtime.state_store import NodeState, StateStore


@pytest.fixture
def store(tmp_path):
    s = StateStore(tmp_path / "test.db")
    yield s
    s.close()


class TestRuns:
    def test_create_and_get_run(self, store):
        run = store.create_run("/tmp/repo", "fix the bug")
        assert run.run_id.startswith("run-")
        assert run.status == "running"

        fetched = store.get_run(run.run_id)
        assert fetched is not None
        assert fetched.task == "fix the bug"

    def test_finish_run(self, store):
        run = store.create_run("/tmp/repo", "task")
        store.finish_run(run.run_id, "completed")
        fetched = store.get_run(run.run_id)
        assert fetched.status == "completed"
        assert fetched.finished_at is not None

    def test_set_root_node(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.set_root_node(run.run_id, node.node_id)
        fetched = store.get_run(run.run_id)
        assert fetched.root_node_id == node.node_id

    def test_create_run_migrates_legacy_runs_table(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                repo_root TEXT NOT NULL,
                task TEXT NOT NULL,
                root_node_id TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                pass_count INTEGER NOT NULL DEFAULT 1,
                persistent INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

        store = StateStore(db_path)
        try:
            run = store.create_run("/tmp/repo", "task", test_command="pytest")
            fetched = store.get_run(run.run_id)
            assert fetched is not None
            assert fetched.test_command == "pytest"
        finally:
            store.close()


class TestNodes:
    def test_create_node_defaults_to_queued(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        assert node.state == NodeState.QUEUED
        assert node.node_id.startswith("node-")

    def test_get_children(self, store):
        run = store.create_run("/tmp/repo", "task")
        parent = store.create_node(run.run_id, "parent task")
        c1 = store.create_node(run.run_id, "child 1", parent_id=parent.node_id)
        c2 = store.create_node(run.run_id, "child 2", parent_id=parent.node_id)

        children = store.get_children(parent.node_id)
        assert len(children) == 2
        assert {c.node_id for c in children} == {c1.node_id, c2.node_id}

    def test_update_node_fields(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.update_node(node.node_id, worktree_path="/tmp/wt", branch_name="ri/test")
        fetched = store.get_node(node.node_id)
        assert fetched.worktree_path == "/tmp/wt"
        assert fetched.branch_name == "ri/test"


class TestStateTransitions:
    def test_valid_transition_queued_to_planning(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        updated = store.transition_node(node.node_id, NodeState.PLANNING)
        assert updated.state == NodeState.PLANNING

    def test_valid_transition_planning_to_executing(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.transition_node(node.node_id, NodeState.PLANNING)
        updated = store.transition_node(node.node_id, NodeState.EXECUTING)
        assert updated.state == NodeState.EXECUTING

    def test_invalid_transition_raises(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        with pytest.raises(ValueError, match="Invalid transition"):
            store.transition_node(node.node_id, NodeState.MERGING)

    def test_terminal_states_block_transitions(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.transition_node(node.node_id, NodeState.PLANNING)
        store.transition_node(node.node_id, NodeState.EXECUTING)
        store.transition_node(node.node_id, NodeState.COMPLETED)
        with pytest.raises(ValueError, match="Invalid transition"):
            store.transition_node(node.node_id, NodeState.PLANNING)

    def test_full_direct_solve_path(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.transition_node(node.node_id, NodeState.PLANNING)
        store.transition_node(node.node_id, NodeState.EXECUTING)
        store.transition_node(node.node_id, NodeState.COMPLETED)
        assert store.get_node(node.node_id).state == NodeState.COMPLETED

    def test_full_recursive_path(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.transition_node(node.node_id, NodeState.PLANNING)
        store.transition_node(node.node_id, NodeState.WAITING_ON_CHILDREN)
        store.transition_node(node.node_id, NodeState.REVIEWING_CHILDREN)
        store.transition_node(node.node_id, NodeState.MERGING)
        store.transition_node(node.node_id, NodeState.COMPLETED)
        assert store.get_node(node.node_id).state == NodeState.COMPLETED

    def test_revise_loop(self, store):
        """reviewing_children -> waiting_on_children -> reviewing_children."""
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.transition_node(node.node_id, NodeState.PLANNING)
        store.transition_node(node.node_id, NodeState.WAITING_ON_CHILDREN)
        store.transition_node(node.node_id, NodeState.REVIEWING_CHILDREN)
        # Revise: back to waiting
        store.transition_node(node.node_id, NodeState.WAITING_ON_CHILDREN)
        # Second review
        store.transition_node(node.node_id, NodeState.REVIEWING_CHILDREN)
        store.transition_node(node.node_id, NodeState.MERGING)
        store.transition_node(node.node_id, NodeState.COMPLETED)
        assert store.get_node(node.node_id).state == NodeState.COMPLETED


class TestEvents:
    def test_node_creation_emits_event(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        events = store.get_node_events(node.node_id)
        assert len(events) == 1
        assert events[0].event_type == "node_created"

    def test_transition_emits_event(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.transition_node(node.node_id, NodeState.PLANNING)
        events = store.get_node_events(node.node_id)
        assert len(events) == 2
        assert events[1].event_type == "state_transition"
        assert events[1].data["from"] == "queued"
        assert events[1].data["to"] == "planning"

    def test_append_custom_event(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.append_event(run.run_id, node.node_id, "custom", {"key": "value"})
        events = store.get_node_events(node.node_id)
        assert events[-1].event_type == "custom"
        assert events[-1].data["key"] == "value"

    def test_get_run_events(self, store):
        run = store.create_run("/tmp/repo", "task")
        n1 = store.create_node(run.run_id, "t1")
        n2 = store.create_node(run.run_id, "t2")
        events = store.get_run_events(run.run_id)
        assert len(events) == 2  # two node_created events


class TestIdempotency:
    def test_child_spawn_dedupe(self, store):
        run = store.create_run("/tmp/repo", "task")
        parent = store.create_node(run.run_id, "parent")

        assert not store.child_spawn_key_exists(parent.node_id, "slot-0", "abc123")

        store.append_event(run.run_id, parent.node_id, "child_spawned", {
            "child_id": "node-test",
            "child_slot": "slot-0",
            "task_hash": "abc123",
        })

        assert store.child_spawn_key_exists(parent.node_id, "slot-0", "abc123")
        assert not store.child_spawn_key_exists(parent.node_id, "slot-1", "abc123")
