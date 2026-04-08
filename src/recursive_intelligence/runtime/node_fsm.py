"""Node finite state machine – drives a single node through its lifecycle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from recursive_intelligence.runtime.state_store import NodeRecord, NodeState, StateStore


@dataclass
class PlanDecision:
    """Typed decision returned by a node's planning phase."""

    action: str  # solve_directly, spawn_children, route_to_children, pause, done
    rationale: str = ""
    children: list[ChildSpec] | None = None
    file_scope: list[str] | None = None
    # For route_to_children: which existing children to reactivate
    routes: list[RouteSpec] | None = None


@dataclass
class ChildSpec:
    """Specification for a child node to spawn."""

    idempotency_key: str
    objective: str
    success_criteria: list[str] = field(default_factory=list)
    domain_name: str | None = None
    file_patterns: list[str] | None = None
    module_scope: str | None = None
    depends_on: list[str] = field(default_factory=list)
    interface_contract: str = ""
    handoff_artifacts: list[str] = field(default_factory=list)
    verification_command: str | None = None
    verification_notes: str = ""


@dataclass
class RouteSpec:
    """Route follow-up work to an existing child."""

    child_node_id: str
    domain_name: str
    task_spec: str


@dataclass
class ExecutionResult:
    """Result of a node's execution phase."""

    status: str  # implemented, blocked
    summary: str = ""
    changed_files: list[str] | None = None
    commit_sha: str | None = None
    request: UpstreamRequest | None = None
    blocker: BlockerInfo | None = None


@dataclass
class ReviewVerdict:
    """Parent's verdict on a child's work."""

    child_id: str
    verdict: str  # accept, revise, reject
    reason: str = ""
    follow_up: str = ""


@dataclass
class UpstreamRequest:
    kind: str
    summary: str
    details: str = ""
    action_requested: str = ""
    requires_input: bool = False
    urgency: str = "normal"
    request_id: str = ""


@dataclass
class EscalationInfo:
    kind: str
    summary: str
    details: str = ""
    action_requested: str = ""


@dataclass
class BlockerInfo:
    kind: str
    recoverable: bool = True
    details: str = ""
    needs_user_input: bool = False
    owner: str = "node"
    urgency: str = "normal"
    escalation: EscalationInfo | None = None


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

        elif decision.action in ("spawn_children", "route_to_children"):
            node = self.store.transition_node(self.node_id, NodeState.WAITING_ON_CHILDREN, data)
            if decision.children:
                for child_spec in decision.children:
                    self._spawn_child(child_spec)
            return node

        elif decision.action == "review_children":
            return self.store.transition_node(self.node_id, NodeState.REVIEWING_CHILDREN, data)

        elif decision.action == "integrate_and_finish":
            return self.store.transition_node(self.node_id, NodeState.MERGING, data)

        elif decision.action == "pause":
            return self.store.transition_node(self.node_id, NodeState.PAUSED, data)

        elif decision.action == "done":
            return self.store.transition_node(self.node_id, NodeState.COMPLETED, data)

        else:
            raise ValueError(f"Unknown plan action: {decision.action}")

    def finish_execution(self, result: ExecutionResult, verify: bool = False) -> NodeRecord:
        """Finish execution and transition to completed, verifying, or failed."""
        data: dict[str, Any] = {"status": result.status, "summary": result.summary}
        if result.commit_sha:
            data["commit_sha"] = result.commit_sha
        if result.changed_files:
            data["changed_files"] = result.changed_files

        if result.status == "blocked":
            if result.blocker:
                data["blocker"] = blocker_to_dict(result.blocker)
                if result.blocker.needs_user_input:
                    data["request"] = blocker_to_request_dict(result.blocker)
            next_state = NodeState.PAUSED if result.blocker and result.blocker.needs_user_input else NodeState.FAILED
            return self.store.transition_node(self.node_id, next_state, data)

        if result.status == "request_upstream":
            if result.request:
                data["request"] = request_to_dict(result.request)
            return self.store.transition_node(self.node_id, NodeState.PAUSED, data)

        next_state = NodeState.VERIFYING if verify else NodeState.COMPLETED
        return self.store.transition_node(self.node_id, next_state, data)

    def apply_review_verdict(self, verdict: ReviewVerdict) -> NodeRecord:
        """Apply a review verdict. May loop back to waiting if revising."""
        data: dict[str, Any] = {
            "child_id": verdict.child_id,
            "verdict": verdict.verdict,
            "reason": verdict.reason,
        }

        self.store.append_event(self.node.run_id, self.node_id, "review_verdict", data)

        if verdict.verdict == "revise":
            return self.store.transition_node(self.node_id, NodeState.WAITING_ON_CHILDREN, data)

        # Only consider verdicts from the CURRENT review round.
        children = self.store.get_children(self.node_id)
        events = self.store.get_node_events(self.node_id)

        current_round_start = 0
        for i, e in enumerate(events):
            if (e.event_type == "state_transition"
                    and e.data.get("to") == NodeState.REVIEWING_CHILDREN.value):
                current_round_start = i

        current_round_verdicts = [
            e for e in events[current_round_start:]
            if e.event_type == "review_verdict"
        ]
        reviewed_ids = {e.data.get("child_id") for e in current_round_verdicts}

        all_reviewed = all(c.node_id in reviewed_ids for c in children)
        has_accepted = any(
            e.data.get("verdict") == "accept" for e in current_round_verdicts
        )

        if all_reviewed and has_accepted:
            return self.store.transition_node(self.node_id, NodeState.MERGING, data)

        if all_reviewed:
            return self.store.transition_node(self.node_id, NodeState.FAILED, {
                "failure_type": "all_children_rejected",
                "failure_reason": "No child produced acceptable work",
            })

        return self.node

    def finish_merge(self, commit_sha: str, verify: bool = False) -> NodeRecord:
        data = {"final_commit_sha": commit_sha}
        next_state = NodeState.VERIFYING if verify else NodeState.COMPLETED
        return self.store.transition_node(self.node_id, next_state, data)

    def pause_after_merge(self, commit_sha: str, verify: bool = False) -> NodeRecord:
        """Pause after merge instead of completing — for persistent multi-pass runs.

        If verify=True, go to VERIFYING first (the orchestrator will pause
        after verification passes).
        """
        if verify:
            data = {"merge_commit_sha": commit_sha}
            return self.store.transition_node(self.node_id, NodeState.VERIFYING, data)
        data = {"merge_commit_sha": commit_sha}
        return self.store.transition_node(self.node_id, NodeState.PAUSED, data)

    def finish_verify_pass(self, persistent_root: bool = False) -> NodeRecord:
        """Verification passed — complete or pause."""
        data = {"verification": "passed"}
        next_state = NodeState.PAUSED if persistent_root else NodeState.COMPLETED
        return self.store.transition_node(self.node_id, next_state, data)

    def finish_verify_fail(self, retries_remaining: int) -> NodeRecord:
        """Verification failed — retry via re-planning or fail permanently."""
        data = {"verification": "failed", "retries_remaining": retries_remaining}
        if retries_remaining > 0:
            return self.store.transition_node(self.node_id, NodeState.PLANNING, data)
        return self.store.transition_node(self.node_id, NodeState.FAILED, data)

    def resume_from_pause(self, user_input: str) -> NodeRecord:
        """Resume a paused node for a new pass."""
        self.store.append_event(self.node.run_id, self.node_id, "user_input", {
            "input": user_input,
        })
        return self.store.transition_node(self.node_id, NodeState.PLANNING, {
            "reason": "user_follow_up",
        })

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
            "domain_name": spec.domain_name or "",
            "file_patterns": spec.file_patterns or [],
            "module_scope": spec.module_scope or "",
            "depends_on": spec.depends_on,
            "interface_contract": spec.interface_contract,
            "handoff_artifacts": spec.handoff_artifacts,
            "verification_command": spec.verification_command or "",
            "verification_notes": spec.verification_notes,
        })

        child_metadata = {
            "child_slot": spec.idempotency_key,
            "success_criteria": spec.success_criteria,
            "domain_name": spec.domain_name or "",
            "file_patterns": spec.file_patterns or [],
            "module_scope": spec.module_scope or "",
            "depends_on": spec.depends_on,
            "interface_contract": spec.interface_contract,
            "handoff_artifacts": spec.handoff_artifacts,
            "verification_command": spec.verification_command or "",
            "verification_notes": spec.verification_notes,
        }
        self.store.update_node(child.node_id, metadata=child_metadata)

        # Register domain if provided
        if spec.domain_name:
            self.store.register_domain(
                run_id=node.run_id,
                parent_node_id=self.node_id,
                child_node_id=child.node_id,
                domain_name=spec.domain_name,
                file_patterns=spec.file_patterns,
                module_scope=spec.module_scope or "",
            )

        return child


def escalation_to_dict(escalation: EscalationInfo) -> dict[str, Any]:
    return {
        "kind": escalation.kind,
        "summary": escalation.summary,
        "details": escalation.details,
        "action_requested": escalation.action_requested,
    }


def request_to_dict(request: UpstreamRequest) -> dict[str, Any]:
    return {
        "kind": request.kind,
        "summary": request.summary,
        "details": request.details,
        "action_requested": request.action_requested,
        "requires_input": request.requires_input,
        "urgency": request.urgency,
        "request_id": request.request_id,
    }


def blocker_to_dict(blocker: BlockerInfo) -> dict[str, Any]:
    data: dict[str, Any] = {
        "kind": blocker.kind,
        "recoverable": blocker.recoverable,
        "details": blocker.details,
        "needs_user_input": blocker.needs_user_input,
        "owner": blocker.owner,
        "urgency": blocker.urgency,
    }
    if blocker.escalation:
        data["escalation"] = escalation_to_dict(blocker.escalation)
    return data


def blocker_to_request_dict(blocker: BlockerInfo) -> dict[str, Any]:
    summary = blocker.escalation.summary if blocker.escalation else blocker.details or blocker.kind
    action_requested = blocker.escalation.action_requested if blocker.escalation else ""
    details = blocker.escalation.details if blocker.escalation else blocker.details
    return {
        "kind": blocker.kind,
        "summary": summary,
        "details": details,
        "action_requested": action_requested,
        "requires_input": blocker.needs_user_input,
        "urgency": blocker.urgency,
        "request_id": "",
    }
