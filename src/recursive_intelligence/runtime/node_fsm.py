"""Node finite state machine – drives a single node through its lifecycle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from recursive_intelligence.runtime.state_store import NodeRecord, NodeState, StateStore


@dataclass
class PlanDecision:
    """Typed decision returned by a node's planning phase."""

    action: str  # solve_directly, spawn_children, review_children, integrate_and_finish
    rationale: str = ""
    children: list[ChildSpec] | None = None
    file_scope: list[str] | None = None


@dataclass
class ChildSpec:
    """Specification for a child node to spawn."""

    idempotency_key: str
    objective: str
    success_criteria: list[str]


@dataclass
class ExecutionResult:
    """Result of a node's execution phase."""

    status: str  # implemented, blocked
    summary: str = ""
    changed_files: list[str] | None = None
    commit_sha: str | None = None
    blocker: BlockerInfo | None = None


@dataclass
class ReviewVerdict:
    """Parent's verdict on a child's work."""

    child_id: str
    verdict: str  # accept, revise, reject
    reason: str = ""
    follow_up: str = ""


@dataclass
class BlockerInfo:
    kind: str
    recoverable: bool = True
    details: str = ""


def child_spawn_dedupe_key(parent_id: str, slot: str, task_spec: str) -> str:
    task_hash = hashlib.sha256(task_spec.encode()).hexdigest()[:8]
    return f"{parent_id}:{slot}:{task_hash}"


def task_hash_short(task_spec: str) -> str:
    return hashlib.sha256(task_spec.encode()).hexdigest()[:8]


class NodeFSM:
    """Drives state transitions for a single node."""

    def __init__(self, store: StateStore, node_id: str) -> None:
        self.store = store
        self.node_id = node_id

    @property
    def node(self) -> NodeRecord:
        n = self.store.get_node(self.node_id)
        if n is None:
            raise ValueError(f"Node {self.node_id} not found")
        return n

    def start_planning(self) -> NodeRecord:
        return self.store.transition_node(self.node_id, NodeState.PLANNING)

    def apply_plan_decision(self, decision: PlanDecision) -> NodeRecord:
        """Apply a planning decision and transition accordingly."""
        data: dict[str, Any] = {"action": decision.action, "rationale": decision.rationale}

        if decision.action == "solve_directly":
            return self.store.transition_node(self.node_id, NodeState.EXECUTING, data)

        elif decision.action == "spawn_children":
            node = self.store.transition_node(self.node_id, NodeState.WAITING_ON_CHILDREN, data)
            if decision.children:
                for child_spec in decision.children:
                    self._spawn_child(child_spec)
            return node

        elif decision.action == "review_children":
            return self.store.transition_node(self.node_id, NodeState.REVIEWING_CHILDREN, data)

        elif decision.action == "integrate_and_finish":
            return self.store.transition_node(self.node_id, NodeState.MERGING, data)

        else:
            raise ValueError(f"Unknown plan action: {decision.action}")

    def finish_execution(self, result: ExecutionResult) -> NodeRecord:
        """Finish execution and transition to completed or appropriate state."""
        data: dict[str, Any] = {"status": result.status, "summary": result.summary}
        if result.commit_sha:
            data["commit_sha"] = result.commit_sha
        if result.changed_files:
            data["changed_files"] = result.changed_files

        if result.status == "blocked":
            if result.blocker:
                data["blocker"] = {
                    "kind": result.blocker.kind,
                    "recoverable": result.blocker.recoverable,
                    "details": result.blocker.details,
                }
            return self.store.transition_node(self.node_id, NodeState.FAILED, data)

        return self.store.transition_node(self.node_id, NodeState.COMPLETED, data)

    def apply_review_verdict(self, verdict: ReviewVerdict) -> NodeRecord:
        """Apply a review verdict. May loop back to waiting if revising."""
        data: dict[str, Any] = {
            "child_id": verdict.child_id,
            "verdict": verdict.verdict,
            "reason": verdict.reason,
        }

        self.store.append_event(self.node.run_id, self.node_id, "review_verdict", data)

        # Check if all children have been reviewed
        children = self.store.get_children(self.node_id)
        events = self.store.get_node_events(self.node_id)
        review_verdicts = [e for e in events if e.event_type == "review_verdict"]
        reviewed_ids = {e.data.get("child_id") for e in review_verdicts}

        # If any child needs revision, go back to waiting
        if verdict.verdict == "revise":
            return self.store.transition_node(self.node_id, NodeState.WAITING_ON_CHILDREN, data)

        # If all children reviewed and at least one accepted, proceed to merging
        all_reviewed = all(c.node_id in reviewed_ids for c in children)
        has_accepted = any(
            e.data.get("verdict") == "accept" for e in review_verdicts
        )

        if all_reviewed and has_accepted:
            return self.store.transition_node(self.node_id, NodeState.MERGING, data)

        # If all reviewed but none accepted, fail
        if all_reviewed:
            return self.store.transition_node(self.node_id, NodeState.FAILED, {
                "failure_type": "all_children_rejected",
                "failure_reason": "No child produced acceptable work",
            })

        # Still waiting on more children to review
        return self.node

    def finish_merge(self, commit_sha: str) -> NodeRecord:
        data = {"final_commit_sha": commit_sha}
        return self.store.transition_node(self.node_id, NodeState.COMPLETED, data)

    def fail(self, reason: str, failure_type: str = "error") -> NodeRecord:
        return self.store.transition_node(self.node_id, NodeState.FAILED, {
            "failure_type": failure_type,
            "failure_reason": reason,
        })

    def cancel(self, reason: str = "") -> NodeRecord:
        return self.store.transition_node(self.node_id, NodeState.CANCELLED, {
            "cancel_reason": reason,
        })

    def wake_for_review(self) -> NodeRecord:
        """Wake a waiting parent to review children."""
        return self.store.transition_node(self.node_id, NodeState.REVIEWING_CHILDREN)

    def _spawn_child(self, spec: ChildSpec) -> NodeRecord | None:
        node = self.node
        task_hash = task_hash_short(spec.objective)

        if self.store.child_spawn_key_exists(self.node_id, spec.idempotency_key, task_hash):
            return None

        child = self.store.create_node(
            run_id=node.run_id,
            task_spec=spec.objective,
            parent_id=self.node_id,
        )

        self.store.append_event(node.run_id, self.node_id, "child_spawned", {
            "child_id": child.node_id,
            "child_slot": spec.idempotency_key,
            "task_hash": task_hash,
            "objective": spec.objective,
            "success_criteria": spec.success_criteria,
        })

        return child
