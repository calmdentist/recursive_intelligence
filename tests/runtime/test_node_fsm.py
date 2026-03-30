"""Tests for the node FSM."""

import pytest

from recursive_intelligence.runtime.node_fsm import (
    ChildSpec,
    ExecutionResult,
    NodeFSM,
    PlanDecision,
    ReviewVerdict,
    child_spawn_dedupe_key,
    task_hash_short,
)
from recursive_intelligence.runtime.state_store import NodeState, StateStore


@pytest.fixture
def store(tmp_path):
    s = StateStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def run_and_node(store):
    run = store.create_run("/tmp/repo", "task")
    node = store.create_node(run.run_id, "implement feature X")
    return run, node


class TestPlanDecisions:
    def test_solve_directly(self, store, run_and_node):
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()

        decision = PlanDecision(action="solve_directly", rationale="Small task")
        fsm.apply_plan_decision(decision)
        assert fsm.node.state == NodeState.EXECUTING

    def test_spawn_children(self, store, run_and_node):
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()

        decision = PlanDecision(
            action="spawn_children",
            rationale="Complex task",
            children=[
                ChildSpec("slot-0", "subtask A", ["test passes"]),
                ChildSpec("slot-1", "subtask B", ["no errors"]),
            ],
        )
        fsm.apply_plan_decision(decision)
        assert fsm.node.state == NodeState.WAITING_ON_CHILDREN

        children = store.get_children(node.node_id)
        assert len(children) == 2

    def test_spawn_children_idempotent(self, store, run_and_node):
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()

        spec = ChildSpec("slot-0", "subtask A", ["test passes"])
        decision = PlanDecision(action="spawn_children", rationale="x", children=[spec, spec])
        fsm.apply_plan_decision(decision)

        # Second spec with same key should be deduped
        children = store.get_children(node.node_id)
        assert len(children) == 1


class TestExecution:
    def test_successful_execution(self, store, run_and_node):
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))

        result = ExecutionResult(
            status="implemented",
            summary="Done",
            changed_files=["src/foo.py"],
            commit_sha="abc123",
        )
        fsm.finish_execution(result)
        assert fsm.node.state == NodeState.COMPLETED


class TestReview:
    def test_accept_verdict_leads_to_merge(self, store, run_and_node):
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()

        child_spec = ChildSpec("slot-0", "subtask", ["works"])
        fsm.apply_plan_decision(PlanDecision(action="spawn_children", children=[child_spec]))

        # Simulate child completing
        children = store.get_children(node.node_id)
        assert len(children) == 1
        child = children[0]
        store.transition_node(child.node_id, NodeState.PLANNING)
        store.transition_node(child.node_id, NodeState.EXECUTING)
        store.transition_node(child.node_id, NodeState.COMPLETED)

        # Parent reviews
        fsm.wake_for_review()
        verdict = ReviewVerdict(child_id=child.node_id, verdict="accept", reason="looks good")
        fsm.apply_review_verdict(verdict)
        assert fsm.node.state == NodeState.MERGING

    def test_revise_verdict_loops_back(self, store, run_and_node):
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()

        child_spec = ChildSpec("slot-0", "subtask", ["works"])
        fsm.apply_plan_decision(PlanDecision(action="spawn_children", children=[child_spec]))

        children = store.get_children(node.node_id)
        child = children[0]
        store.transition_node(child.node_id, NodeState.PLANNING)
        store.transition_node(child.node_id, NodeState.EXECUTING)
        store.transition_node(child.node_id, NodeState.COMPLETED)

        fsm.wake_for_review()
        verdict = ReviewVerdict(child_id=child.node_id, verdict="revise", follow_up="add tests")
        fsm.apply_review_verdict(verdict)
        assert fsm.node.state == NodeState.WAITING_ON_CHILDREN


class TestVerification:
    def test_execution_to_verify(self, store, run_and_node):
        """finish_execution with verify=True goes to VERIFYING."""
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))

        result = ExecutionResult(status="implemented", summary="Done")
        fsm.finish_execution(result, verify=True)
        assert fsm.node.state == NodeState.VERIFYING

    def test_execution_without_verify(self, store, run_and_node):
        """finish_execution with verify=False goes to COMPLETED (default)."""
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))

        result = ExecutionResult(status="implemented", summary="Done")
        fsm.finish_execution(result, verify=False)
        assert fsm.node.state == NodeState.COMPLETED

    def test_verify_pass_completes(self, store, run_and_node):
        """Passing verification transitions to COMPLETED."""
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))
        fsm.finish_execution(ExecutionResult(status="implemented"), verify=True)
        assert fsm.node.state == NodeState.VERIFYING

        fsm.finish_verify_pass()
        assert fsm.node.state == NodeState.COMPLETED

    def test_verify_pass_pauses_persistent_root(self, store, run_and_node):
        """Passing verification on persistent root transitions to PAUSED."""
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))
        fsm.finish_execution(ExecutionResult(status="implemented"), verify=True)

        fsm.finish_verify_pass(persistent_root=True)
        assert fsm.node.state == NodeState.PAUSED

    def test_verify_fail_retries(self, store, run_and_node):
        """Failing verification with retries remaining goes to PLANNING."""
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))
        fsm.finish_execution(ExecutionResult(status="implemented"), verify=True)

        fsm.finish_verify_fail(retries_remaining=1)
        assert fsm.node.state == NodeState.PLANNING

    def test_verify_fail_exhausted(self, store, run_and_node):
        """Failing verification with no retries goes to FAILED."""
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))
        fsm.finish_execution(ExecutionResult(status="implemented"), verify=True)

        fsm.finish_verify_fail(retries_remaining=0)
        assert fsm.node.state == NodeState.FAILED

    def test_merge_to_verify(self, store, run_and_node):
        """finish_merge with verify=True goes to VERIFYING."""
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)
        fsm.start_planning()

        child_spec = ChildSpec("slot-0", "subtask", ["works"])
        fsm.apply_plan_decision(PlanDecision(action="spawn_children", children=[child_spec]))

        children = store.get_children(node.node_id)
        child = children[0]
        store.transition_node(child.node_id, NodeState.PLANNING)
        store.transition_node(child.node_id, NodeState.EXECUTING)
        store.transition_node(child.node_id, NodeState.COMPLETED)

        fsm.wake_for_review()
        verdict = ReviewVerdict(child_id=child.node_id, verdict="accept", reason="ok")
        fsm.apply_review_verdict(verdict)
        assert fsm.node.state == NodeState.MERGING

        fsm.finish_merge("abc123", verify=True)
        assert fsm.node.state == NodeState.VERIFYING

    def test_full_verify_retry_cycle(self, store, run_and_node):
        """Full cycle: execute → verify (fail) → re-plan → execute → verify (pass)."""
        _, node = run_and_node
        fsm = NodeFSM(store, node.node_id)

        # First pass
        fsm.start_planning()
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))
        fsm.finish_execution(ExecutionResult(status="implemented"), verify=True)
        assert fsm.node.state == NodeState.VERIFYING

        # Verification fails, retries remaining
        fsm.finish_verify_fail(retries_remaining=1)
        assert fsm.node.state == NodeState.PLANNING

        # Re-plan and re-execute
        fsm.apply_plan_decision(PlanDecision(action="solve_directly"))
        assert fsm.node.state == NodeState.EXECUTING
        fsm.finish_execution(ExecutionResult(status="implemented"), verify=True)
        assert fsm.node.state == NodeState.VERIFYING

        # Verification passes
        fsm.finish_verify_pass()
        assert fsm.node.state == NodeState.COMPLETED


class TestHelpers:
    def test_task_hash_deterministic(self):
        h1 = task_hash_short("implement feature X")
        h2 = task_hash_short("implement feature X")
        assert h1 == h2
        assert len(h1) == 8

    def test_task_hash_differs(self):
        h1 = task_hash_short("task A")
        h2 = task_hash_short("task B")
        assert h1 != h2

    def test_dedupe_key_format(self):
        key = child_spawn_dedupe_key("node-abc", "slot-0", "do stuff")
        assert key.startswith("node-abc:slot-0:")
        assert len(key.split(":")) == 3
