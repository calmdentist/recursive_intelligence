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
        assert run.telemetry["user_interruptions_count"] == 0

        fetched = store.get_run(run.run_id)
        assert fetched is not None
        assert fetched.task == "fix the bug"
        assert fetched.telemetry["root_escalations_count"] == 0

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
            assert fetched.telemetry["human_inputs_count"] == 0
        finally:
            store.close()

    def test_increment_run_telemetry(self, store):
        run = store.create_run("/tmp/repo", "task")
        store.increment_run_telemetry(
            run.run_id,
            user_interruptions_count=1,
            root_escalations_count=2,
        )
        fetched = store.get_run(run.run_id)
        assert fetched.telemetry["user_interruptions_count"] == 1
        assert fetched.telemetry["root_escalations_count"] == 2


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

    def test_get_latest_blocker_from_escalation_event(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.transition_node(node.node_id, NodeState.PLANNING)
        store.transition_node(node.node_id, NodeState.EXECUTING)
        store.transition_node(node.node_id, NodeState.PAUSED, {
            "blocker": {"kind": "missing_credentials", "details": "need key"},
        })
        store.append_event(run.run_id, node.node_id, "root_escalation_requested", {
            "blocker": {"kind": "missing_credentials", "details": "need key"},
        })

        blocker = store.get_latest_blocker(node.node_id)
        assert blocker is not None
        assert blocker["blocker"]["kind"] == "missing_credentials"

    def test_get_latest_request_from_request_event(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "subtask")
        store.transition_node(node.node_id, NodeState.PLANNING)
        store.transition_node(node.node_id, NodeState.EXECUTING)
        store.transition_node(node.node_id, NodeState.PAUSED, {
            "request": {"kind": "clarification", "summary": "Need API choice"},
        })
        store.append_event(run.run_id, node.node_id, "root_request_upstream", {
            "request": {"kind": "clarification", "summary": "Need API choice"},
        })

        request = store.get_latest_request(node.node_id)
        assert request is not None
        assert request["request"]["kind"] == "clarification"
        assert request["request"]["summary"] == "Need API choice"

    def test_get_run_blockers_filters_active_nodes(self, store):
        run = store.create_run("/tmp/repo", "task")
        paused = store.create_node(run.run_id, "paused task")
        done = store.create_node(run.run_id, "done task")

        store.transition_node(paused.node_id, NodeState.PLANNING)
        store.transition_node(paused.node_id, NodeState.EXECUTING)
        store.transition_node(paused.node_id, NodeState.PAUSED, {
            "blocker": {"kind": "needs_input", "details": "question"},
        })

        store.transition_node(done.node_id, NodeState.PLANNING)
        store.transition_node(done.node_id, NodeState.EXECUTING)
        store.transition_node(done.node_id, NodeState.COMPLETED)

        blockers = store.get_run_blockers(run.run_id)
        assert len(blockers) == 1
        assert blockers[0]["node_id"] == paused.node_id

    def test_get_run_requests_filters_active_nodes(self, store):
        run = store.create_run("/tmp/repo", "task")
        paused = store.create_node(run.run_id, "paused task")
        store.transition_node(paused.node_id, NodeState.PLANNING)
        store.transition_node(paused.node_id, NodeState.EXECUTING)
        store.transition_node(paused.node_id, NodeState.PAUSED, {
            "request": {"kind": "capability", "summary": "Need Stripe key"},
        })

        requests = store.get_run_requests(run.run_id)
        assert len(requests) == 1
        assert requests[0]["request"]["kind"] == "capability"

    def test_get_run_inbox_tracks_unresolved_root_requests(self, store):
        run = store.create_run("/tmp/repo", "task")
        root = store.create_node(run.run_id, "root task")
        child = store.create_node(run.run_id, "child task", parent_id=root.node_id)
        store.set_root_node(run.run_id, root.node_id)

        request = {
            "request_id": "req-123",
            "kind": "capability",
            "summary": "Need Stripe API key",
            "action_requested": "Add a Stripe test key",
        }
        store.append_event(run.run_id, root.node_id, "root_request_upstream", {
            "request": request,
            "source_child_id": child.node_id,
        })

        inbox = store.get_run_inbox(run.run_id)
        assert len(inbox) == 1
        assert inbox[0]["request_id"] == "req-123"
        assert inbox[0]["request"]["summary"] == "Need Stripe API key"
        assert inbox[0]["source_child_id"] == child.node_id
        assert inbox[0]["source_task_spec"] == "child task"

    def test_get_run_inbox_hides_resolved_requests(self, store):
        run = store.create_run("/tmp/repo", "task")
        root = store.create_node(run.run_id, "root task")
        store.set_root_node(run.run_id, root.node_id)

        store.append_event(run.run_id, root.node_id, "root_request_upstream", {
            "request": {
                "request_id": "req-123",
                "kind": "clarification",
                "summary": "Need product choice",
            },
        })
        store.append_event(run.run_id, root.node_id, "request_resolved", {
            "request_id": "req-123",
        })

        assert store.get_run_inbox(run.run_id) == []

    def test_get_inbox_request_returns_specific_unresolved_item(self, store):
        run = store.create_run("/tmp/repo", "task")
        root = store.create_node(run.run_id, "root task")
        store.set_root_node(run.run_id, root.node_id)

        store.append_event(run.run_id, root.node_id, "root_request_upstream", {
            "request": {
                "request_id": "req-123",
                "kind": "clarification",
                "summary": "Need copy direction",
            },
        })
        store.append_event(run.run_id, root.node_id, "root_request_upstream", {
            "request": {
                "request_id": "req-456",
                "kind": "capability",
                "summary": "Need API key",
            },
        })

        item = store.get_inbox_request(run.run_id, "req-456")
        assert item is not None
        assert item["request"]["kind"] == "capability"
        assert store.get_inbox_request(run.run_id, "req-missing") is None

    def test_get_latest_downstream_task_tracks_unresolved_work(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "child task")
        store.append_event(run.run_id, node.node_id, "downstream_task", {
            "kind": "revision",
            "summary": "Need tests",
            "task_spec": "add tests",
        })

        task = store.get_latest_downstream_task(node.node_id)
        assert task is not None
        assert task["task"]["kind"] == "revision"
        assert task["task"]["task_spec"] == "add tests"

    def test_get_latest_downstream_task_clears_after_result(self, store):
        run = store.create_run("/tmp/repo", "task")
        node = store.create_node(run.run_id, "child task")
        store.append_event(run.run_id, node.node_id, "downstream_task", {
            "kind": "merge_conflict",
            "summary": "Resolve conflict",
        })
        store.append_event(run.run_id, node.node_id, "downstream_task_result", {
            "kind": "merge_conflict",
            "status": "completed",
        })

        assert store.get_latest_downstream_task(node.node_id) is None

    def test_get_run_downstream_tasks_returns_all_unresolved_items(self, store):
        run = store.create_run("/tmp/repo", "task")
        first = store.create_node(run.run_id, "first task")
        second = store.create_node(run.run_id, "second task")
        store.append_event(run.run_id, first.node_id, "downstream_task", {
            "kind": "revision",
            "summary": "Fix tests",
            "task_spec": "add tests",
        })
        store.append_event(run.run_id, second.node_id, "reactivation_requested", {
            "new_task": "add auth",
            "previous_summary": "built api",
            "original_task": "build api",
        })

        tasks = store.get_run_downstream_tasks(run.run_id)
        assert len(tasks) == 2
        assert {item["task"]["kind"] for item in tasks} == {"revision", "reactivation"}


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
