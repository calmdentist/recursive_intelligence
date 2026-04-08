"""Event-driven orchestrator – the runtime's main loop.

The orchestrator never reasons about tasks. It only:
- creates worktrees
- launches or resumes Claude sessions via the adapter
- persists events and state
- delivers child-result summaries to parents
- performs git integration requested by parent decisions
- routes follow-up work to existing children (multi-pass)
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from recursive_intelligence.adapters.base import AgentAdapter, StreamCallback
from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.git.diffing import ArtifactBundle, build_artifact_bundle, find_scope_violations
from recursive_intelligence.git.merge import (
    cherry_pick_child,
    abort_cherry_pick,
    get_conflict_diff,
    stage_and_continue_cherry_pick,
)
from recursive_intelligence.git.worktrees import (
    branch_name,
    create_worktree,
    ensure_clean_repo,
    get_head_sha,
    remove_worktree,
)
from recursive_intelligence.adapters.claude.prompts import (
    downstream_response_prompt,
    planning_prompt,
    execution_prompt,
    review_prompt as build_review_prompt_template,
    revision_prompt,
    routing_prompt,
    reactivation_prompt,
    conflict_resolution_prompt,
    verification_retry_prompt,
)
from recursive_intelligence.runtime.node_fsm import (
    blocker_to_dict,
    blocker_to_request_dict,
    BlockerInfo,
    ChildSpec,
    EscalationInfo,
    ExecutionResult,
    NodeFSM,
    PlanDecision,
    ReviewVerdict,
    RouteSpec,
    UpstreamRequest,
    request_to_dict,
)
from recursive_intelligence.runtime.state_store import NodeState, StateStore
from recursive_intelligence.test_commands import run_test_command

log = logging.getLogger(__name__)


class Orchestrator:
    """Drives the recursive runtime."""

    def __init__(self, config: RuntimeConfig, adapter: AgentAdapter, on_message: StreamCallback = None) -> None:
        self.config = config
        self.adapter = adapter
        self.store: StateStore | None = None
        self._on_message = on_message
        self._root_node_id: str | None = None

    def _is_root(self, node_id: str) -> bool:
        """Check if a node is the root node of its run."""
        store = self._ensure_store()
        node = store.get_node(node_id)
        if node is None:
            return False
        run = store.get_run(node.run_id)
        return run is not None and run.root_node_id == node_id

    def _ensure_store(self) -> StateStore:
        if self.store is None:
            self.config.ensure_dirs()
            self.store = StateStore(self.config.db_path)
        return self.store

    def _stream_callback_for(self, node_id: str) -> StreamCallback:
        """Return the stream callback only if this is the root node."""
        if node_id == self._root_node_id and self._on_message:
            return self._on_message
        return None

    def _new_request_id(self) -> str:
        return f"req-{uuid4().hex[:12]}"

    def _normalize_request_response(self, resolution: str, response_text: str) -> str:
        text = response_text.strip()
        if text:
            return text
        if resolution == "approve":
            return "Approved."
        if resolution == "decline":
            return "Declined."
        raise ValueError("A response is required to answer this request")

    def _latest_root_inbox_request(self, run_id: str) -> dict[str, Any] | None:
        store = self._ensure_store()
        inbox = store.get_run_inbox(run_id)
        if not inbox:
            return None
        return inbox[0]

    def _resolve_request_chain(
        self,
        run,
        root,
        request_item: dict[str, Any],
        response_text: str,
        resolution: str = "answer",
    ) -> bool:
        """Route a resolved root-facing request back to the paused node."""
        store = self._ensure_store()
        request_payload = request_item.get("request", {})
        source_child_id = request_item.get("source_child_id")
        request_id = request_payload.get("request_id", "")
        request_summary = request_payload.get("summary", "")
        response_text = self._normalize_request_response(resolution, response_text)
        store.append_event(run.run_id, root.node_id, "user_input", {
            "input": response_text,
            "request_summary": request_summary,
            "request_id": request_id,
            "resolution": resolution,
        })

        if source_child_id:
            child = store.get_node(source_child_id)
            if child is None:
                return False
            store.append_event(run.run_id, root.node_id, "request_resolved", {
                "request_id": request_id,
                "resolved_by": "user_input",
                "response_text": response_text,
                "source_child_id": source_child_id,
                "resolution": resolution,
            })
            store.append_event(child.run_id, child.node_id, "response_downstream", {
                "response_text": response_text,
                "request": request_payload,
                "source_node_id": root.node_id,
                "request_id": request_id,
                "resolution": resolution,
            })
            store.append_event(child.run_id, child.node_id, "request_resolved", {
                "request_id": request_id,
                "resolved_by": "root_response",
                "response_text": response_text,
                "source_node_id": root.node_id,
                "resolution": resolution,
            })
            if child.state == NodeState.PAUSED:
                store.transition_node(child.node_id, NodeState.EXECUTING, {
                    "reason": "response_downstream",
                    "source_node_id": root.node_id,
                    "request_id": request_id,
                })
            store.transition_node(root.node_id, NodeState.WAITING_ON_CHILDREN, {
                "reason": "request_resolved",
                "source_child_id": source_child_id,
                "request_id": request_id,
            })
            store.increment_run_telemetry(run.run_id, resolved_requests_count=1)
            return True

        store.append_event(run.run_id, root.node_id, "request_resolved", {
            "request_id": request_id,
            "resolved_by": "user_input",
            "response_text": response_text,
            "resolution": resolution,
        })
        store.append_event(run.run_id, root.node_id, "response_downstream", {
            "response_text": response_text,
            "request": request_payload,
            "source_node_id": root.node_id,
            "request_id": request_id,
            "resolution": resolution,
        })
        store.transition_node(root.node_id, NodeState.EXECUTING, {
            "reason": "response_downstream",
            "request_id": request_id,
        })
        store.increment_run_telemetry(run.run_id, resolved_requests_count=1)
        return True

    def _resume_paused_request_chain(self, run, root, user_input: str) -> bool:
        """If the root is paused on an upstream request, route the response directly."""
        request_item = self._latest_root_inbox_request(run.run_id)
        if request_item is None:
            return False
        return self._resolve_request_chain(run, root, request_item, user_input, resolution="answer")

    # --- Run lifecycle ---

    async def start_run(
        self, task: str, persistent: bool = False, test_command: str | None = None,
    ) -> str:
        """Start a new run. Returns the run_id.

        If persistent=True, the run pauses after the first pass instead
        of completing, allowing follow-up via continue_run().
        If test_command is provided, nodes will verify their work by running
        this command after execution/merge and retry on failure.
        """
        ensure_clean_repo(self.config.repo_root)
        store = self._ensure_store()

        run = store.create_run(str(self.config.repo_root), task, persistent=persistent, test_command=test_command)
        log.info("Created run %s (persistent=%s)", run.run_id, persistent)

        b_name = branch_name(run.run_id, "root", task)
        wt_path = create_worktree(
            self.config.repo_root, self.config.worktrees_dir,
            f"{run.run_id}-root", b_name,
        )

        root = store.create_node(
            run_id=run.run_id, task_spec=task,
            worktree_path=str(wt_path), branch_name=b_name,
        )
        store.set_root_node(run.run_id, root.node_id)
        self._root_node_id = root.node_id
        log.info("Created root node %s at %s", root.node_id, wt_path)

        await self._drive_root(run.run_id, root.node_id, persistent)
        return run.run_id

    async def continue_run(self, run_id: str, user_input: str) -> str:
        """Continue a paused persistent run with new user instructions.

        Returns the run_id.
        """
        store = self._ensure_store()
        run = store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        if run.status != "paused":
            raise ValueError(f"Run {run_id} is {run.status}, not paused")

        root = store.get_node(run.root_node_id)
        if root is None or root.state != NodeState.PAUSED:
            raise ValueError(f"Root node is not paused (state={root.state.value if root else 'missing'})")

        self._root_node_id = root.node_id
        store.resume_paused_run(run_id)
        store.increment_run_telemetry(run_id, human_inputs_count=1)
        if not self._resume_paused_request_chain(run, root, user_input):
            fsm = NodeFSM(store, root.node_id)
            fsm.resume_from_pause(user_input)

        await self._drive_root(run_id, root.node_id, persistent=True)
        return run_id

    async def resolve_request(
        self,
        run_id: str,
        request_id: str,
        response_text: str,
        resolution: str = "answer",
    ) -> str:
        """Resolve a specific root-inbox request and continue the run."""
        store = self._ensure_store()
        run = store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        if run.status != "paused":
            raise ValueError(f"Run {run_id} is {run.status}, not paused")

        root = store.get_node(run.root_node_id)
        if root is None or root.state != NodeState.PAUSED:
            raise ValueError(f"Root node is not paused (state={root.state.value if root else 'missing'})")

        request_item = store.get_inbox_request(run_id, request_id)
        if request_item is None:
            raise ValueError(f"Request {request_id} not found in the root inbox for run {run_id}")

        self._root_node_id = root.node_id
        store.resume_paused_run(run_id)
        store.increment_run_telemetry(run_id, human_inputs_count=1)
        if not self._resolve_request_chain(run, root, request_item, response_text, resolution=resolution):
            raise ValueError(f"Could not resolve request {request_id}")

        await self._drive_root(run_id, root.node_id, persistent=True)
        return run_id

    async def _drive_root(self, run_id: str, root_node_id: str, persistent: bool) -> None:
        """Drive the root node and finalize the run status."""
        store = self._ensure_store()

        try:
            await self._drive_node(root_node_id)
        except Exception as e:
            log.error("Run %s failed: %s", run_id, e)
            try:
                fsm = NodeFSM(store, root_node_id)
                if not fsm.node.state.is_idle:
                    fsm.fail(str(e), failure_type="orchestrator_error")
            except Exception:
                pass

        root = store.get_node(root_node_id)
        if root.state == NodeState.PAUSED:
            store.pause_run(run_id)
        elif root.state == NodeState.COMPLETED:
            store.finish_run(run_id, "completed")
        elif root.state.is_terminal:
            store.finish_run(run_id, "failed")
        # else: still running (shouldn't happen, but don't finalize)

    async def resume_run(self, run_id: str) -> None:
        """Resume a crashed/interrupted run (not the same as continue_run for multi-pass)."""
        store = self._ensure_store()
        run = store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        if run.status not in ("running",):
            raise ValueError(f"Run {run_id} is {run.status}, not resumable")

        resumable = [
            NodeState.QUEUED, NodeState.PLANNING, NodeState.EXECUTING,
            NodeState.WAITING_ON_CHILDREN, NodeState.REVIEWING_CHILDREN,
            NodeState.VERIFYING,
            NodeState.MERGING,
        ]
        node_ids = []
        for state in resumable:
            nodes = store.get_nodes_in_state(run_id, state)
            node_ids.extend(n.node_id for n in nodes)

        if node_ids:
            await asyncio.gather(*[self._drive_node(nid) for nid in node_ids])

        root = store.get_node(run.root_node_id)
        if root is None:
            return
        if root.state == NodeState.PAUSED:
            store.pause_run(run_id)
        elif root.state == NodeState.COMPLETED:
            store.finish_run(run_id, "completed")
        elif root.state.is_terminal:
            store.finish_run(run_id, "failed")

    # --- Node driving ---

    async def _drive_node(self, node_id: str) -> None:
        """Drive a single node through its lifecycle via a loop."""
        store = self._ensure_store()
        fsm = NodeFSM(store, node_id)

        while True:
            node = fsm.node
            log.info("Driving node %s (state=%s)", node_id, node.state.value)

            if node.state.is_idle:
                break

            prev_state = node.state

            if node.state == NodeState.QUEUED:
                fsm.start_planning()
            elif node.state == NodeState.PLANNING:
                await self._run_planning(fsm)
            elif node.state == NodeState.EXECUTING:
                await self._run_execution(fsm)
            elif node.state == NodeState.WAITING_ON_CHILDREN:
                await self._wait_and_drive_children(fsm)
            elif node.state == NodeState.REVIEWING_CHILDREN:
                await self._run_review(fsm)
            elif node.state == NodeState.MERGING:
                await self._run_merge(fsm)
            elif node.state == NodeState.VERIFYING:
                await self._run_verify(fsm)
            else:
                break

            if fsm.node.state == prev_state:
                log.warning("Node %s stuck in %s, breaking", node_id, prev_state.value)
                break

    # --- Planning ---

    async def _run_planning(self, fsm: NodeFSM) -> None:
        node = fsm.node
        store = self._ensure_store()
        worktree = Path(node.worktree_path) if node.worktree_path else self.config.repo_root
        run = store.get_run(node.run_id)

        # Check if this is a verification retry (re-planning after test failure)
        is_verify_retry = self._is_verification_retry(node)

        # Check if this is a routing pass (multi-pass follow-up on root or parent)
        is_routing = run and run.pass_count > 1 and len(store.get_children(node.node_id)) > 0

        if is_verify_retry:
            prompt = self._build_verify_retry_prompt(node, run)
        elif is_routing:
            prompt = self._build_routing_prompt(node, run)
        else:
            prompt = planning_prompt(node.task_spec)

        try:
            result = await self.adapter.run(
                prompt=prompt, worktree=worktree, mode="plan",
                resume_session_id=node.session_id if (is_routing or is_verify_retry) else None,
                on_message=self._stream_callback_for(node.node_id),
                is_root=self._is_root(node.node_id),
            )
        except Exception as e:
            log.error("Planning failed for %s: %s", node.node_id, e)
            fsm.fail(str(e), failure_type="adapter_error")
            return

        previous_session_id = node.session_id
        store.update_node(node.node_id, session_id=result.session_id)
        if result.session_id != previous_session_id or not store.session_exists(result.session_id):
            store.create_session(result.session_id, node.node_id, self.adapter.name)
        store.finish_session(result.session_id)

        store.append_event(node.run_id, node.node_id, "plan_result", {
            "session_id": result.session_id,
            "raw": result.raw,
            "cost": _cost_to_dict(result.cost),
        })

        decision = self._parse_plan_decision(result.raw)
        log.info("Node %s plan: %s (%s)", node.node_id, decision.action, decision.rationale[:80])

        # Handle routing: reactivate existing children
        if decision.action == "route_to_children" and decision.routes:
            for route in decision.routes:
                self._prepare_child_reactivation(node, route)

        fsm.apply_plan_decision(decision)

    def _build_routing_prompt(self, node, run) -> str:
        """Build the routing prompt with domain registry for multi-pass."""
        store = self._ensure_store()
        domains = store.get_domains(node.node_id)

        # Get the latest user input
        events = store.get_node_events(node.node_id)
        user_input = node.task_spec
        for evt in reversed(events):
            if evt.event_type == "user_input":
                user_input = evt.data.get("input", node.task_spec)
                break

        # Enrich domains with child state and last summary
        domain_dicts = []
        for d in domains:
            child = store.get_node(d.child_node_id)
            child_summary = ""
            handoff_summary = ""
            interface_contract = ""
            depends_on: list[str] = []
            if child:
                child_events = store.get_node_events(child.node_id)
                for evt in reversed(child_events):
                    if evt.event_type == "execution_result":
                        child_summary = evt.data.get("raw", {}).get("summary", "")
                        break
                child_coordination = self._coordination_contract_for(child)
                depends_on = child_coordination["depends_on"]
                interface_contract = child_coordination["interface_contract"]
                handoff_summary = self._latest_handoff_for(child)["summary"]
            domain_dicts.append({
                "domain_name": d.domain_name,
                "child_node_id": d.child_node_id,
                "file_patterns": d.file_patterns,
                "module_scope": d.module_scope,
                "child_state": child.state.value if child else "unknown",
                "last_summary": child_summary,
                "depends_on": depends_on,
                "interface_contract": interface_contract,
                "handoff_summary": handoff_summary,
            })

        return routing_prompt(user_input, domain_dicts, run.pass_count)

    def _append_downstream_task(self, run_id: str, node_id: str, kind: str, **data: Any) -> dict[str, Any]:
        store = self._ensure_store()
        payload = {"kind": kind, **data}
        store.append_event(run_id, node_id, "downstream_task", payload)
        return payload

    def _prepare_child_reactivation(self, parent_node, route: RouteSpec) -> None:
        """Prepare a child for downstream follow-up work."""
        store = self._ensure_store()
        child = store.get_node(route.child_node_id)
        if child is None:
            log.warning("Route target %s not found", route.child_node_id)
            return

        # Get the child's previous summary
        child_events = store.get_node_events(child.node_id)
        prev_summary = ""
        for evt in reversed(child_events):
            if evt.event_type == "execution_result":
                prev_summary = evt.data.get("raw", {}).get("summary", "")
                break

        self._append_downstream_task(
            child.run_id,
            child.node_id,
            "reactivation",
            summary="Follow-up work routed from parent",
            task_spec=route.task_spec,
            previous_summary=prev_summary,
            original_task=child.task_spec,
            requested_by=parent_node.node_id,
            domain_name=route.domain_name,
        )

    def _latest_pending_work_event(self, events: list[Any]) -> Any | None:
        pending_event = None
        for event in events:
            if event.event_type in ("downstream_task", "reactivation_requested", "revision_requested", "response_downstream"):
                pending_event = event
            elif event.event_type in ("execution_result", "downstream_task_result"):
                pending_event = None
        return pending_event

    def _success_criteria_for(self, node) -> list[str]:
        criteria = (node.metadata or {}).get("success_criteria", [])
        return criteria if isinstance(criteria, list) else []

    def _verification_notes_for(self, node) -> str:
        notes = (node.metadata or {}).get("verification_notes", "")
        return notes.strip() if isinstance(notes, str) else ""

    def _ownership_contract_for(self, node) -> dict[str, Any]:
        store = self._ensure_store()
        metadata = node.metadata or {}
        domain = store.get_domain_by_child(node.node_id)

        file_patterns = metadata.get("file_patterns") or []
        if not isinstance(file_patterns, list):
            file_patterns = []
        if domain and not file_patterns:
            file_patterns = domain.file_patterns

        module_scope = metadata.get("module_scope", "")
        if not isinstance(module_scope, str):
            module_scope = ""
        if domain and not module_scope:
            module_scope = domain.module_scope

        domain_name = metadata.get("domain_name", "")
        if not isinstance(domain_name, str):
            domain_name = ""
        if domain and not domain_name:
            domain_name = domain.domain_name

        return {
            "domain_name": domain_name.strip(),
            "module_scope": module_scope.strip(),
            "file_patterns": file_patterns,
        }

    def _coordination_contract_for(self, node) -> dict[str, Any]:
        metadata = node.metadata or {}
        depends_on = metadata.get("depends_on") or []
        if not isinstance(depends_on, list):
            depends_on = []
        handoff_artifacts = metadata.get("handoff_artifacts") or []
        if not isinstance(handoff_artifacts, list):
            handoff_artifacts = []
        interface_contract = metadata.get("interface_contract", "")
        if not isinstance(interface_contract, str):
            interface_contract = ""
        child_slot = metadata.get("child_slot", "")
        if not isinstance(child_slot, str):
            child_slot = ""
        return {
            "child_slot": child_slot.strip(),
            "depends_on": [dep for dep in depends_on if isinstance(dep, str) and dep.strip()],
            "interface_contract": interface_contract.strip(),
            "handoff_artifacts": [item for item in handoff_artifacts if isinstance(item, str) and item.strip()],
        }

    def _latest_execution_payload(self, node) -> dict[str, Any]:
        store = self._ensure_store()
        for event in reversed(store.get_node_events(node.node_id)):
            if event.event_type == "execution_result":
                raw = event.data.get("raw", {})
                return raw if isinstance(raw, dict) else {}
        return {}

    def _latest_handoff_for(self, node) -> dict[str, Any]:
        payload = self._latest_execution_payload(node)
        handoff = payload.get("handoff", {})
        if not isinstance(handoff, dict):
            handoff = {}
        interfaces = handoff.get("interfaces") or []
        if not isinstance(interfaces, list):
            interfaces = []
        artifacts = handoff.get("artifacts") or []
        if not isinstance(artifacts, list):
            artifacts = []
        breaking_changes = handoff.get("breaking_changes") or []
        if not isinstance(breaking_changes, list):
            breaking_changes = []
        return {
            "summary": handoff.get("summary", ""),
            "interfaces": [item for item in interfaces if isinstance(item, str) and item.strip()],
            "artifacts": [item for item in artifacts if isinstance(item, str) and item.strip()],
            "breaking_changes": [item for item in breaking_changes if isinstance(item, str) and item.strip()],
            "execution_summary": payload.get("summary", ""),
        }

    def _dependency_context_for(self, node) -> list[dict[str, Any]]:
        store = self._ensure_store()
        if not node.parent_id:
            return []
        contract = self._coordination_contract_for(node)
        if not contract["depends_on"]:
            return []
        siblings = store.get_children(node.parent_id)
        items: list[dict[str, Any]] = []
        for dependency in contract["depends_on"]:
            target = None
            for sibling in siblings:
                if sibling.node_id == node.node_id:
                    continue
                sibling_contract = self._coordination_contract_for(sibling)
                sibling_ownership = self._ownership_contract_for(sibling)
                if dependency in {
                    sibling.node_id,
                    sibling_contract["child_slot"],
                    sibling_ownership["domain_name"],
                }:
                    target = sibling
                    break
            if target is None:
                items.append({
                    "dependency": dependency,
                    "summary": "Dependency not found in current sibling set.",
                    "handoff_summary": "",
                    "interfaces": [],
                    "breaking_changes": [],
                })
                continue
            handoff = self._latest_handoff_for(target)
            items.append({
                "dependency": dependency,
                "summary": handoff["execution_summary"] or f"Sibling state: {target.state.value}",
                "handoff_summary": handoff["summary"],
                "interfaces": handoff["interfaces"],
                "breaking_changes": handoff["breaking_changes"],
                "artifacts": handoff["artifacts"],
                "state": target.state.value,
            })
        return items

    def _verification_command_for(self, node, run) -> str | None:
        command = (node.metadata or {}).get("verification_command", "")
        if isinstance(command, str) and command.strip():
            return command.strip()
        if run and run.root_node_id == node.node_id and run.test_command:
            return run.test_command.strip()
        return None

    def _build_execution_prompt(self, node, run) -> str:
        ownership = self._ownership_contract_for(node)
        coordination = self._coordination_contract_for(node)
        return execution_prompt(
            node.task_spec,
            is_root=self._is_root(node.node_id),
            success_criteria=self._success_criteria_for(node),
            verification_command=self._verification_command_for(node, run),
            verification_notes=self._verification_notes_for(node),
            domain_name=ownership["domain_name"],
            module_scope=ownership["module_scope"],
            file_patterns=ownership["file_patterns"],
            depends_on=coordination["depends_on"],
            interface_contract=coordination["interface_contract"],
            handoff_artifacts=coordination["handoff_artifacts"],
            dependency_context=self._dependency_context_for(node),
        )

    def _latest_verification_result_after_execution(self, node) -> dict[str, Any] | None:
        store = self._ensure_store()
        for event in reversed(store.get_node_events(node.node_id)):
            if event.event_type == "verification_result":
                return event.data
            if event.event_type == "execution_result":
                return None
        return None

    def _local_verification_status(self, node, run) -> dict[str, Any]:
        command = self._verification_command_for(node, run)
        notes = self._verification_notes_for(node)
        if not command:
            return {
                "required": False,
                "status": "not_required",
                "command": "",
                "notes": notes,
                "output": "",
            }

        latest = self._latest_verification_result_after_execution(node)
        if latest and latest.get("test_command") == command:
            return {
                "required": True,
                "status": "passed" if latest.get("passed") else "failed",
                "command": command,
                "notes": notes,
                "output": latest.get("output", ""),
                "attempt": latest.get("attempt", 0),
            }

        return {
            "required": True,
            "status": "missing",
            "command": command,
            "notes": notes,
            "output": "",
            "attempt": 0,
        }

    def _build_verification_follow_up(self, verification: dict[str, Any]) -> str:
        lines = ["Get the focused local verification passing before asking for review again."]
        if verification.get("command"):
            lines.append(f"Run: {verification['command']}")
        if verification.get("notes"):
            lines.append(f"Notes: {verification['notes']}")
        if verification.get("status") == "failed" and verification.get("output"):
            lines.append("Latest failing output:")
            lines.append(verification["output"][:4000])
        return "\n".join(lines)

    def _build_ownership_follow_up(
        self,
        ownership: dict[str, Any],
        changed_files: list[str],
        scope_violations: list[str],
    ) -> str:
        lines = [
            "Bring this child back inside its ownership boundary before asking for review again.",
            f"Files outside scope: {', '.join(scope_violations)}",
        ]
        if ownership.get("file_patterns"):
            lines.append(f"Allowed files: {', '.join(ownership['file_patterns'])}")
        if ownership.get("module_scope"):
            lines.append(f"Module scope: {ownership['module_scope']}")
        if changed_files:
            lines.append(f"Current changed files: {', '.join(changed_files)}")
        return "\n".join(lines)

    def _queue_child_revision(self, parent_node, child, verdict: ReviewVerdict) -> None:
        self._append_downstream_task(
            child.run_id,
            child.node_id,
            "revision",
            summary=verdict.reason or "Review requested changes",
            task_spec=verdict.follow_up,
            reason=verdict.reason,
            requested_by=parent_node.node_id,
            source_child_id=child.node_id,
        )

    def _build_downstream_task_prompt(self, node, run, task_data: dict[str, Any]) -> str:
        ownership = self._ownership_contract_for(node)
        coordination = self._coordination_contract_for(node)
        contract_kwargs = {
            "success_criteria": self._success_criteria_for(node),
            "verification_command": self._verification_command_for(node, run),
            "verification_notes": self._verification_notes_for(node),
            "domain_name": ownership["domain_name"],
            "module_scope": ownership["module_scope"],
            "file_patterns": ownership["file_patterns"],
            "depends_on": coordination["depends_on"],
            "interface_contract": coordination["interface_contract"],
            "handoff_artifacts": coordination["handoff_artifacts"],
            "dependency_context": self._dependency_context_for(node),
        }
        task_kind = task_data.get("kind", "")
        if task_kind == "reactivation":
            return reactivation_prompt(
                original_task=task_data.get("original_task", node.task_spec),
                previous_summary=task_data.get("previous_summary", ""),
                new_task=task_data.get("task_spec", ""),
                is_root=self._is_root(node.node_id),
                **contract_kwargs,
            )
        if task_kind == "revision":
            return revision_prompt(
                task_data.get("task_spec", ""),
                is_root=self._is_root(node.node_id),
                **contract_kwargs,
            )
        if task_kind == "merge_conflict":
            return conflict_resolution_prompt(
                task_data.get("child_id", ""),
                task_data.get("conflict_files", []),
                task_data.get("conflict_diff", ""),
            )
        return self._build_execution_prompt(node, run)

    # --- Execution ---

    async def _run_execution(self, fsm: NodeFSM) -> None:
        node = fsm.node
        store = self._ensure_store()
        worktree = Path(node.worktree_path) if node.worktree_path else self.config.repo_root
        run = store.get_run(node.run_id)
        ownership = self._ownership_contract_for(node)
        coordination = self._coordination_contract_for(node)
        dependency_context = self._dependency_context_for(node)

        events = store.get_node_events(node.node_id)
        pending_event = self._latest_pending_work_event(events)

        if pending_event and pending_event.event_type == "downstream_task":
            prompt = self._build_downstream_task_prompt(node, run, pending_event.data)
            resume_id = node.session_id
        elif pending_event and pending_event.event_type == "reactivation_requested":
            prompt = reactivation_prompt(
                original_task=pending_event.data.get("original_task", node.task_spec),
                previous_summary=pending_event.data.get("previous_summary", ""),
                new_task=pending_event.data.get("new_task", ""),
                is_root=self._is_root(node.node_id),
                success_criteria=self._success_criteria_for(node),
                verification_command=self._verification_command_for(node, run),
                verification_notes=self._verification_notes_for(node),
                domain_name=ownership["domain_name"],
                module_scope=ownership["module_scope"],
                file_patterns=ownership["file_patterns"],
                depends_on=coordination["depends_on"],
                interface_contract=coordination["interface_contract"],
                handoff_artifacts=coordination["handoff_artifacts"],
                dependency_context=dependency_context,
            )
            resume_id = node.session_id
        elif pending_event and pending_event.event_type == "revision_requested":
            prompt = revision_prompt(
                pending_event.data.get("follow_up", ""),
                is_root=self._is_root(node.node_id),
                success_criteria=self._success_criteria_for(node),
                verification_command=self._verification_command_for(node, run),
                verification_notes=self._verification_notes_for(node),
                domain_name=ownership["domain_name"],
                module_scope=ownership["module_scope"],
                file_patterns=ownership["file_patterns"],
                depends_on=coordination["depends_on"],
                interface_contract=coordination["interface_contract"],
                handoff_artifacts=coordination["handoff_artifacts"],
                dependency_context=dependency_context,
            )
            resume_id = node.session_id
        elif pending_event and pending_event.event_type == "response_downstream":
            request = pending_event.data.get("request", {})
            prompt = downstream_response_prompt(
                request_summary=request.get("summary", ""),
                response_text=pending_event.data.get("response_text", ""),
                original_details=request.get("details", ""),
                resolution=pending_event.data.get("resolution", "answer"),
                is_root=self._is_root(node.node_id),
                success_criteria=self._success_criteria_for(node),
                verification_command=self._verification_command_for(node, run),
                verification_notes=self._verification_notes_for(node),
                domain_name=ownership["domain_name"],
                module_scope=ownership["module_scope"],
                file_patterns=ownership["file_patterns"],
                depends_on=coordination["depends_on"],
                interface_contract=coordination["interface_contract"],
                handoff_artifacts=coordination["handoff_artifacts"],
                dependency_context=dependency_context,
            )
            resume_id = node.session_id
        else:
            prompt = self._build_execution_prompt(node, run)
            resume_id = None

        try:
            result = await self.adapter.run(
                prompt=prompt, worktree=worktree, mode="execute",
                resume_session_id=resume_id,
                on_message=self._stream_callback_for(node.node_id),
                is_root=self._is_root(node.node_id),
            )
        except Exception as e:
            log.error("Execution failed for %s: %s", node.node_id, e)
            fsm.fail(str(e), failure_type="adapter_error")
            return

        store.update_node(node.node_id, session_id=result.session_id)
        store.append_event(node.run_id, node.node_id, "execution_result", {
            "session_id": result.session_id,
            "raw": result.raw,
            "cost": _cost_to_dict(result.cost),
        })

        # Auto-update domain registry with actual changed files
        execution = self._parse_execution_result(result.raw)
        if execution.request:
            request = execution.request
            if not request.request_id:
                request.request_id = self._new_request_id()
            request_dict = request_to_dict(request)
            store.append_event(node.run_id, node.node_id, "request_upstream", {
                "request": request_dict,
                "resume_state": "executing",
            })
            if self._is_root(node.node_id):
                self._record_root_request(node.run_id, node.node_id, request)
        elif execution.status == "blocked" and execution.blocker and execution.blocker.needs_user_input:
            request_payload = blocker_to_request_dict(execution.blocker)
            if not request_payload.get("request_id"):
                request_payload["request_id"] = self._new_request_id()
            store.append_event(node.run_id, node.node_id, "request_upstream", {
                "request": request_payload,
                "resume_state": "executing",
                "blocker": blocker_to_dict(execution.blocker),
            })
            if self._is_root(node.node_id):
                self._record_root_request(
                    node.run_id,
                    node.node_id,
                    self._request_from_dict(request_payload),
                )
        if execution.changed_files:
            self._update_domain_from_changed_files(node, execution.changed_files)

        should_verify = self._should_verify(node, run)
        log.info("Node %s execution: %s (verify=%s)", node.node_id, execution.status, should_verify)
        fsm.finish_execution(execution, verify=should_verify)

    def _update_domain_from_changed_files(self, node, changed_files: list[str]) -> None:
        """Backfill empty domain scopes from actual files touched."""
        store = self._ensure_store()
        domain = store.get_domain_by_child(node.node_id)
        if domain and changed_files and not domain.file_patterns:
            store.update_domain(domain.domain_id, file_patterns=sorted(set(changed_files)))

    # --- Children ---

    async def _wait_and_drive_children(self, fsm: NodeFSM) -> None:
        store = self._ensure_store()
        children = store.get_children(fsm.node_id)

        if not children:
            log.warning("Node %s waiting on children but has none, failing", fsm.node_id)
            fsm.fail("No children found", failure_type="orchestrator_error")
            return

        # Phase 1: Prepare all children serially.
        # Worktree creation and state transitions must be serial because
        # `git worktree add` modifies the shared .git directory.
        children_to_drive: list[str] = []
        for child in children:
            if child.state == NodeState.COMPLETED and self._has_pending_work(child):
                log.info("Child %s has pending work, re-executing", child.node_id)
                store.transition_node(child.node_id, NodeState.EXECUTING, {
                    "reason": "reactivation_or_revision",
                })
            elif child.state == NodeState.PAUSED and self._has_pending_work(child):
                log.info("Child %s paused with pending work, re-executing", child.node_id)
                store.transition_node(child.node_id, NodeState.EXECUTING, {
                    "reason": "reactivation_or_revision",
                })
            elif child.state.is_idle:
                continue

            if not child.worktree_path:
                parent_node = fsm.node
                parent_wt = Path(parent_node.worktree_path) if parent_node.worktree_path else self.config.repo_root
                b_name = branch_name(child.run_id, child.node_id, child.task_spec)
                parent_head = get_head_sha(parent_wt)
                wt_path = create_worktree(
                    self.config.repo_root, self.config.worktrees_dir,
                    child.node_id, b_name, base_ref=parent_head,
                )
                store.update_node(child.node_id, worktree_path=str(wt_path), branch_name=b_name)

            children_to_drive.append(child.node_id)

        # Phase 2: Drive all prepared children in parallel.
        # Each child has its own worktree and session — fully isolated.
        # Nested parallelism is automatic: if a child spawns grandchildren,
        # this same method will gather *those* in parallel too.
        if children_to_drive:
            sem = (
                asyncio.Semaphore(self.config.max_parallel_children)
                if self.config.max_parallel_children > 0
                else None
            )

            async def _drive_child_safe(child_node_id: str) -> None:
                async def _inner():
                    try:
                        await self._drive_node(child_node_id)
                    except Exception as e:
                        log.error("Child %s failed: %s", child_node_id, e)
                        child_fsm = NodeFSM(store, child_node_id)
                        if not child_fsm.node.state.is_idle:
                            child_fsm.fail(str(e), failure_type="orchestrator_error")

                if sem:
                    async with sem:
                        await _inner()
                else:
                    await _inner()

            log.info(
                "Driving %d children in parallel (max_parallel=%s)",
                len(children_to_drive),
                self.config.max_parallel_children or "unlimited",
            )
            await asyncio.gather(*[
                _drive_child_safe(cid) for cid in children_to_drive
            ])

        requests = self._collect_pending_child_requests(store.get_children(fsm.node_id))
        if requests:
            self._pause_for_child_request(fsm, requests)
            return

        fsm.wake_for_review()

    def _has_pending_work(self, child) -> bool:
        """Check if a child has pending downstream work."""
        store = self._ensure_store()
        events = store.get_node_events(child.node_id)
        return self._latest_pending_work_event(events) is not None

    def _collect_pending_child_requests(self, children: list[Any]) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        for child in children:
            request = self._latest_upstream_request(child)
            if request:
                requests.append({
                    "child_id": child.node_id,
                    "task_spec": child.task_spec,
                    "request": request,
                })
        return requests

    def _latest_upstream_request(self, node) -> dict[str, Any] | None:
        if node.state != NodeState.PAUSED:
            return None
        store = self._ensure_store()
        latest = store.get_latest_request(node.node_id)
        return latest["request"] if latest else None

    def _pause_for_child_request(self, fsm: NodeFSM, requests: list[dict[str, Any]]) -> None:
        store = self._ensure_store()
        node = fsm.node
        primary = requests[0]
        request = primary["request"]
        if self._is_root(node.node_id):
            self._record_root_request(
                node.run_id,
                node.node_id,
                self._request_from_dict(request),
                source_child_id=primary["child_id"],
            )
        else:
            store.append_event(node.run_id, node.node_id, "request_forwarded_upstream", {
                "source_child_id": primary["child_id"],
                "source_task_spec": primary["task_spec"],
                "request": request,
                "request_count": len(requests),
            })
        store.transition_node(node.node_id, NodeState.PAUSED, {
            "reason": "child_request",
            "source_child_id": primary["child_id"],
            "request": request,
            "request_id": request.get("request_id", ""),
        })

    def _record_root_request(
        self,
        run_id: str,
        node_id: str,
        request: UpstreamRequest,
        source_child_id: str | None = None,
    ) -> None:
        store = self._ensure_store()
        store.append_event(run_id, node_id, "root_request_upstream", {
            "source_child_id": source_child_id,
            "request": request_to_dict(request),
        })
        store.increment_run_telemetry(
            run_id,
            user_interruptions_count=1,
            root_escalations_count=1,
            root_requests_count=1,
        )

    # --- Review ---

    async def _run_review(self, fsm: NodeFSM) -> None:
        store = self._ensure_store()
        node = fsm.node
        run = store.get_run(node.run_id)
        children = store.get_children(node.node_id)

        spawn_events = [
            e for e in store.get_node_events(node.node_id)
            if e.event_type == "child_spawned"
        ]
        criteria_by_child: dict[str, list[str]] = {}
        for evt in spawn_events:
            criteria_by_child[evt.data.get("child_id", "")] = evt.data.get("success_criteria", [])

        # Determine which children were already accepted in prior review rounds
        # so we don't re-review them. Only review children that completed since
        # the last time the parent entered WAITING_ON_CHILDREN.
        parent_events = store.get_node_events(node.node_id)
        last_waiting_ts = ""
        for e in reversed(parent_events):
            if (e.event_type == "state_transition"
                    and e.data.get("to") == NodeState.WAITING_ON_CHILDREN.value):
                last_waiting_ts = e.timestamp
                break

        already_reviewed_this_round = set()
        for e in parent_events:
            if e.timestamp >= last_waiting_ts and e.event_type == "review_verdict":
                already_reviewed_this_round.add(e.data.get("child_id"))

        # Children already accepted in earlier passes don't need re-review
        # unless they were reactivated (have work after their last integration)
        previously_accepted = set()
        for e in parent_events:
            if (e.event_type == "review_verdict"
                    and e.data.get("verdict") == "accept"
                    and e.timestamp < last_waiting_ts):
                previously_accepted.add(e.data.get("child_id"))

        for child in children:
            # Skip children already reviewed in this round
            if child.node_id in already_reviewed_this_round:
                continue

            # Skip previously-accepted children unless they have new work
            if child.node_id in previously_accepted and not self._has_pending_work(child):
                # Auto-accept: they were accepted before and haven't changed
                verdict = ReviewVerdict(
                    child_id=child.node_id, verdict="accept",
                    reason="Previously accepted, no new work",
                )
                fsm.apply_review_verdict(verdict)
                if fsm.node.state == NodeState.WAITING_ON_CHILDREN:
                    return
                continue

            # PAUSED children have completed their work (merged and paused) — treat as reviewable
            if child.state not in (NodeState.COMPLETED, NodeState.PAUSED) or not child.worktree_path:
                if child.state in (NodeState.FAILED, NodeState.CANCELLED):
                    verdict = ReviewVerdict(
                        child_id=child.node_id, verdict="reject",
                        reason=f"Child {child.state.value}",
                    )
                    fsm.apply_review_verdict(verdict)
                    if fsm.node.state == NodeState.WAITING_ON_CHILDREN:
                        return
                continue

            parent_wt = Path(node.worktree_path) if node.worktree_path else self.config.repo_root
            child_wt = Path(child.worktree_path)
            base_sha = _merge_base(parent_wt, get_head_sha(parent_wt), get_head_sha(child_wt))
            diff_base = base_sha or "HEAD~1"

            child_events = store.get_node_events(child.node_id)
            child_summary = ""
            for evt in reversed(child_events):
                if evt.event_type == "execution_result":
                    child_summary = evt.data.get("raw", {}).get("summary", "")
                    break

            verification = self._local_verification_status(child, run)
            if verification["required"] and verification["status"] != "passed":
                reason = "Child local verification is not passing"
                if verification["status"] == "missing":
                    reason = "Child local verification has not been run"
                store.append_event(node.run_id, node.node_id, "review_skipped_for_verification", {
                    "child_id": child.node_id,
                    "verification": verification,
                })
                verdict = ReviewVerdict(
                    child_id=child.node_id,
                    verdict="revise",
                    reason=reason,
                    follow_up=self._build_verification_follow_up(verification),
                )
                self._queue_child_revision(node, child, verdict)
                fsm.apply_review_verdict(verdict)
                if fsm.node.state == NodeState.WAITING_ON_CHILDREN:
                    return
                continue

            bundle = build_artifact_bundle(
                node_id=child.node_id, worktree=child_wt,
                base_ref=diff_base, summary=child_summary,
            )
            ownership = self._ownership_contract_for(child)
            coordination = self._coordination_contract_for(child)
            scope_violations = find_scope_violations(
                bundle.changed_files,
                ownership["file_patterns"],
            )
            if scope_violations:
                store.append_event(node.run_id, node.node_id, "review_skipped_for_ownership", {
                    "child_id": child.node_id,
                    "ownership": ownership,
                    "changed_files": bundle.changed_files,
                    "violations": scope_violations,
                })
                verdict = ReviewVerdict(
                    child_id=child.node_id,
                    verdict="revise",
                    reason="Child changed files outside its ownership boundary",
                    follow_up=self._build_ownership_follow_up(
                        ownership,
                        bundle.changed_files,
                        scope_violations,
                    ),
                )
                self._queue_child_revision(node, child, verdict)
                fsm.apply_review_verdict(verdict)
                if fsm.node.state == NodeState.WAITING_ON_CHILDREN:
                    return
                continue

            criteria = criteria_by_child.get(child.node_id, [])

            prompt = build_review_prompt_template(
                child_id=child.node_id, diff=bundle.diff[:8000],
                summary=bundle.summary or child_summary,
                success_criteria=criteria,
                verification=verification,
                ownership={
                    **ownership,
                    "changed_files": bundle.changed_files,
                },
                coordination={
                    **coordination,
                    "dependency_context": self._dependency_context_for(child),
                },
            )

            try:
                result = await self.adapter.run(
                    prompt=prompt, worktree=parent_wt, mode="review",
                    resume_session_id=node.session_id,
                    on_message=self._stream_callback_for(node.node_id),
                    is_root=self._is_root(node.node_id),
                )
            except Exception as e:
                log.error("Review failed for %s: %s", child.node_id, e)
                fsm.fail(str(e), failure_type="review_error")
                return

            store.update_node(node.node_id, session_id=result.session_id)
            store.append_event(node.run_id, node.node_id, "review_result", {
                "child_id": child.node_id,
                "session_id": result.session_id,
                "raw": result.raw,
                "cost": _cost_to_dict(result.cost),
            })
            verdict = self._parse_review_verdict(result.raw, child.node_id)
            log.info("Node %s review of %s: %s", node.node_id, child.node_id, verdict.verdict)

            if verdict.verdict == "revise":
                self._queue_child_revision(node, child, verdict)

            fsm.apply_review_verdict(verdict)

            if fsm.node.state == NodeState.WAITING_ON_CHILDREN:
                return

    # --- Merge ---

    async def _run_merge(self, fsm: NodeFSM) -> None:
        store = self._ensure_store()
        node = fsm.node
        run = store.get_run(node.run_id)
        parent_wt = Path(node.worktree_path) if node.worktree_path else self.config.repo_root

        # Only consider accepted children from the current review round
        events = store.get_node_events(node.node_id)

        # Find current round start
        current_round_start = 0
        for i, e in enumerate(events):
            if (e.event_type == "state_transition"
                    and e.data.get("to") == NodeState.REVIEWING_CHILDREN.value):
                current_round_start = i

        accepted_ids = {
            e.data["child_id"]
            for e in events[current_round_start:]
            if e.event_type == "review_verdict" and e.data.get("verdict") == "accept"
        }

        if not accepted_ids:
            log.warning("No accepted children for %s", node.node_id)
            final_sha = get_head_sha(parent_wt)
            if run and run.persistent:
                fsm.pause_after_merge(final_sha)
            else:
                fsm.finish_merge(final_sha)
            return

        # Build map of last-integrated SHA per child (from previous passes)
        last_integrated: dict[str, str] = {}
        for e in events:
            if (e.event_type == "child_integrated"
                    and e.data.get("status") == "integrated"
                    and e.data.get("commit_sha")):
                last_integrated[e.data["child_id"]] = e.data["commit_sha"]

        children = store.get_children(node.node_id)
        for child in children:
            if child.node_id not in accepted_ids or not child.worktree_path:
                continue

            required_command = self._verification_command_for(child, run)
            if required_command and not self._has_passing_local_verification(child):
                store.append_event(node.run_id, node.node_id, "verification_gate_failed", {
                    "child_id": child.node_id,
                    "required_command": required_command,
                })
                fsm.fail(
                    f"Child {child.node_id} lacks a passing local verification result",
                    failure_type="verification_gate_failed",
                )
                return

            child_wt = Path(child.worktree_path)
            child_sha = get_head_sha(child_wt)

            # Use the last-integrated SHA as cherry-pick base if this child
            # was already integrated in a previous pass. This avoids re-picking
            # commits that were already merged.
            if child.node_id in last_integrated:
                # Use the child's HEAD from the last integration as the base.
                # If missing (old events without child_head_sha), fall back to merge-base.
                prev_child_sha = self._find_last_child_head_at_integration(events, child.node_id)
                if prev_child_sha:
                    base_sha = prev_child_sha
                else:
                    log.warning(
                        "Child %s was previously integrated but child_head_sha missing, "
                        "falling back to merge-base",
                        child.node_id,
                    )
                    parent_sha = get_head_sha(parent_wt)
                    base_sha = _merge_base(parent_wt, parent_sha, child_sha)
            else:
                parent_sha = get_head_sha(parent_wt)
                base_sha = _merge_base(parent_wt, parent_sha, child_sha)

            if child_sha == base_sha:
                log.info("Child %s has no new commits, skipping", child.node_id)
                store.append_event(node.run_id, node.node_id, "child_integrated", {
                    "child_id": child.node_id, "status": "no_change",
                    "commit_sha": child_sha,
                })
                continue

            bundle = build_artifact_bundle(
                node_id=child.node_id,
                worktree=child_wt,
                base_ref=base_sha or "HEAD~1",
                commit_sha=child_sha,
            )
            domain = store.get_domain_by_child(child.node_id)
            scope_violations = find_scope_violations(
                bundle.changed_files,
                domain.file_patterns if domain else None,
            )
            if scope_violations:
                store.append_event(node.run_id, node.node_id, "ownership_violation", {
                    "child_id": child.node_id,
                    "allowed_patterns": domain.file_patterns if domain else [],
                    "violations": scope_violations,
                })
                store.increment_run_telemetry(node.run_id, ownership_violations_count=1)
                fsm.fail(
                    f"Child {child.node_id} modified files outside its declared scope",
                    failure_type="ownership_violation",
                )
                return

            result = cherry_pick_child(parent_wt, child_sha, child.node_id, base_sha=base_sha)

            store.append_event(node.run_id, node.node_id, "child_integrated", {
                "child_id": child.node_id, "status": result.status,
                "commit_sha": result.commit_sha,
                "child_head_sha": child_sha,  # track for multi-pass
                "conflict_files": result.conflict_files,
            })

            if result.status == "conflict":
                log.warning("Conflict integrating %s: %s", child.node_id, result.conflict_files)
                resolved = await self._resolve_conflict(
                    fsm, parent_wt, child.node_id,
                    result.conflict_files or [],
                )
                if not resolved:
                    abort_cherry_pick(parent_wt)
                    fsm.fail(
                        f"Irreconcilable conflict with {child.node_id}",
                        failure_type="merge_conflict",
                    )
                    return
                # Record the successful resolution
                new_sha = get_head_sha(parent_wt)
                store.append_event(node.run_id, node.node_id, "conflict_resolved", {
                    "child_id": child.node_id,
                    "conflict_files": result.conflict_files,
                    "resolved_sha": new_sha,
                })

        final_sha = get_head_sha(parent_wt)
        # Only the root node pauses in persistent mode — children always complete
        is_root = run and run.root_node_id == node.node_id
        should_verify = self._should_verify(node, run)
        if is_root and run.persistent:
            fsm.pause_after_merge(final_sha, verify=should_verify)
            log.info("Node %s merged and %s at %s", node.node_id,
                     "verifying" if should_verify else "paused", final_sha[:8])
        else:
            fsm.finish_merge(final_sha, verify=should_verify)
            log.info("Node %s merged at %s (verify=%s)", node.node_id, final_sha[:8], should_verify)

    # --- Conflict resolution ---

    async def _resolve_conflict(
        self,
        fsm: NodeFSM,
        parent_wt: Path,
        child_id: str,
        conflict_files: list[str],
    ) -> bool:
        """Ask the parent node to resolve a merge conflict. Returns True if resolved."""
        store = self._ensure_store()
        node = fsm.node

        # Get the conflict diff showing markers
        conflict_diff = get_conflict_diff(parent_wt)
        if not conflict_diff:
            log.error("No conflict diff available for %s", child_id)
            store.append_event(node.run_id, node.node_id, "downstream_task_result", {
                "kind": "merge_conflict",
                "child_id": child_id,
                "status": "failed",
                "reason": "No conflict diff available",
                "conflict_files": conflict_files,
            })
            return False

        task_data = self._append_downstream_task(
            node.run_id,
            node.node_id,
            "merge_conflict",
            summary=f"Resolve merge conflict while integrating {child_id}",
            child_id=child_id,
            conflict_files=conflict_files,
            conflict_diff=conflict_diff,
            requested_by=node.node_id,
        )
        run = store.get_run(node.run_id)
        prompt = self._build_downstream_task_prompt(node, run, task_data)

        log.info("Asking parent %s to resolve conflict with %s (%s)",
                 node.node_id, child_id, conflict_files)

        try:
            result = await self.adapter.run(
                prompt=prompt,
                worktree=parent_wt,
                mode="execute",  # needs edit permissions to resolve
                resume_session_id=node.session_id,
                on_message=self._stream_callback_for(node.node_id),
                is_root=self._is_root(node.node_id),
            )
        except Exception as e:
            log.error("Conflict resolution failed for %s: %s", child_id, e)
            store.append_event(node.run_id, node.node_id, "downstream_task_result", {
                "kind": "merge_conflict",
                "child_id": child_id,
                "status": "failed",
                "reason": str(e),
                "conflict_files": conflict_files,
            })
            return False

        previous_session_id = node.session_id
        store.update_node(node.node_id, session_id=result.session_id)
        if result.session_id != previous_session_id or not store.session_exists(result.session_id):
            store.create_session(result.session_id, node.node_id, self.adapter.name)
        store.finish_session(result.session_id)

        raw = result.raw
        status = raw.get("status", "")

        if status == "irreconcilable":
            log.warning("Parent declared conflict irreconcilable: %s",
                        raw.get("reason", ""))
            store.append_event(node.run_id, node.node_id, "downstream_task_result", {
                "kind": "merge_conflict",
                "child_id": child_id,
                "status": "irreconcilable",
                "reason": raw.get("reason", ""),
                "conflict_files": conflict_files,
            })
            return False

        # Whether the LLM said "resolved" or returned unexpected JSON,
        # we need to ensure the cherry-pick is actually finalized.
        # The LLM may have edited files but not committed, or committed
        # but CHERRY_PICK_HEAD still exists. Use stage_and_continue to
        # guarantee the cherry-pick is complete.
        finalized = self._finalize_cherry_pick(parent_wt, child_id, raw.get("summary", ""))
        store.append_event(node.run_id, node.node_id, "downstream_task_result", {
            "kind": "merge_conflict",
            "child_id": child_id,
            "status": "completed" if finalized else "failed",
            "summary": raw.get("summary", ""),
            "conflict_files": conflict_files,
            "resolved_sha": get_head_sha(parent_wt) if finalized else "",
        })
        return finalized

    def _finalize_cherry_pick(self, worktree: Path, child_id: str, summary: str) -> bool:
        """Ensure a cherry-pick is fully completed after LLM conflict resolution.

        Handles cases where the LLM:
        - Resolved conflicts and committed correctly (CHERRY_PICK_HEAD gone → no-op)
        - Resolved conflicts but didn't commit (stage + continue)
        - Staged files but CHERRY_PICK_HEAD still exists (continue)
        """
        import subprocess

        # Check if CHERRY_PICK_HEAD still exists — means cherry-pick isn't done
        cp_head = subprocess.run(
            ["git", "rev-parse", "--verify", "CHERRY_PICK_HEAD"],
            cwd=str(worktree), capture_output=True, text=True,
        )

        if cp_head.returncode != 0:
            # No CHERRY_PICK_HEAD — the LLM committed correctly
            log.info("Cherry-pick already finalized for %s: %s", child_id, summary)
            return True

        # CHERRY_PICK_HEAD exists — finalize the cherry-pick
        log.info("CHERRY_PICK_HEAD still exists for %s, finalizing", child_id)
        try:
            stage_and_continue_cherry_pick(worktree)
            log.info("Cherry-pick finalized for %s", child_id)
            return True
        except Exception as e:
            log.error("Failed to finalize cherry-pick for %s: %s", child_id, e)
            return False

    # --- Verification ---

    async def _run_verify(self, fsm: NodeFSM) -> None:
        """Run the test command against the node's worktree and handle results."""
        store = self._ensure_store()
        node = fsm.node
        run = store.get_run(node.run_id)
        should_verify = self._should_verify(node, run)
        test_command = self._verification_command_for(node, run)

        if not run or not test_command or not should_verify:
            log.info("Node %s: no test command, skipping verification", node.node_id)
            is_root = run and run.root_node_id == node.node_id
            persistent_root = is_root and run.persistent if run else False
            fsm.finish_verify_pass(persistent_root=persistent_root)
            return

        worktree = Path(node.worktree_path) if node.worktree_path else self.config.repo_root

        log.info("Node %s: running verification: %s", node.node_id, test_command)

        try:
            completed = await asyncio.to_thread(
                _run_test_in_worktree, test_command, worktree, timeout=self.config.max_verify_retries * 600 + 600,
            )
            passed = completed.returncode == 0
            output = _truncate_test_output(completed.stdout or "", completed.stderr or "")
        except subprocess.TimeoutExpired:
            passed = False
            output = "Test command timed out"
        except Exception as e:
            log.error("Verification command failed for %s: %s", node.node_id, e)
            passed = False
            output = str(e)

        # Count how many verification attempts this node has had
        events = store.get_node_events(node.node_id)
        verify_count = sum(1 for e in events if e.event_type == "verification_result")

        store.append_event(node.run_id, node.node_id, "verification_result", {
            "passed": passed,
            "test_command": test_command,
            "output": output[:8000],
            "attempt": verify_count + 1,
        })

        if passed:
            log.info("Node %s: verification PASSED (attempt %d)", node.node_id, verify_count + 1)
            is_root = run.root_node_id == node.node_id
            persistent_root = is_root and run.persistent
            fsm.finish_verify_pass(persistent_root=persistent_root)
        else:
            retries_remaining = self.config.max_verify_retries - (verify_count + 1)
            log.info(
                "Node %s: verification FAILED (attempt %d, %d retries left)",
                node.node_id, verify_count + 1, max(0, retries_remaining),
            )
            fsm.finish_verify_fail(retries_remaining=max(0, retries_remaining))

    def _is_verification_retry(self, node) -> bool:
        """Check if this planning phase was triggered by a verification failure."""
        store = self._ensure_store()
        events = store.get_node_events(node.node_id)
        # Walk backwards: if the most recent state_transition into PLANNING
        # came from VERIFYING, this is a verification retry.
        for e in reversed(events):
            if e.event_type == "state_transition" and e.data.get("to") == "planning":
                return e.data.get("from") == "verifying" or e.data.get("verification") == "failed"
        return False

    def _build_verify_retry_prompt(self, node, run) -> str:
        """Build the prompt for re-planning after verification failure."""
        store = self._ensure_store()
        events = store.get_node_events(node.node_id)

        # Find the most recent verification_result event
        test_output = ""
        test_command = self._verification_command_for(node, run) or ""
        for e in reversed(events):
            if e.event_type == "verification_result":
                test_output = e.data.get("output", "")
                test_command = e.data.get("test_command", test_command)
                break

        has_children = len(store.get_children(node.node_id)) > 0
        return verification_retry_prompt(
            task_spec=node.task_spec,
            test_command=test_command,
            test_output=test_output,
            has_children=has_children,
        )

    def _should_verify(self, node, run) -> bool:
        return bool(self._verification_command_for(node, run))

    def _has_passing_local_verification(self, node) -> bool:
        latest = self._latest_verification_result_after_execution(node)
        return bool(latest and latest.get("passed") is True)

    # --- Cleanup ---

    def _find_last_child_head_at_integration(self, events: list, child_id: str) -> str | None:
        """Find the child's HEAD SHA from the last successful integration event."""
        for e in reversed(events):
            if (e.event_type == "child_integrated"
                    and e.data.get("child_id") == child_id
                    and e.data.get("status") == "integrated"):
                return e.data.get("child_head_sha")
        return None

    def cleanup_worktrees(self, run_id: str) -> None:
        store = self._ensure_store()
        nodes = store.get_run_nodes(run_id)
        for node in nodes:
            if node.worktree_path:
                wt = Path(node.worktree_path)
                if wt.exists():
                    try:
                        remove_worktree(self.config.repo_root, wt)
                        log.info("Cleaned up worktree %s", wt)
                    except Exception as e:
                        log.warning("Failed to clean worktree %s: %s", wt, e)

    # --- Parse helpers ---

    def _parse_plan_decision(self, raw: dict[str, Any]) -> PlanDecision:
        children = None
        if raw.get("children"):
            children = [
                ChildSpec(
                    idempotency_key=c.get("idempotency_key", f"slot-{i}"),
                    objective=c["objective"],
                    success_criteria=c.get("success_criteria", []),
                    domain_name=c.get("domain_name"),
                    file_patterns=c.get("file_patterns"),
                    module_scope=c.get("module_scope"),
                    depends_on=c.get("depends_on", []),
                    interface_contract=c.get("interface_contract", ""),
                    handoff_artifacts=c.get("handoff_artifacts", []),
                    verification_command=c.get("verification_command"),
                    verification_notes=c.get("verification_notes", ""),
                )
                for i, c in enumerate(raw["children"])
            ]

        routes = None
        if raw.get("routes"):
            routes = [
                RouteSpec(
                    child_node_id=r["child_node_id"],
                    domain_name=r.get("domain_name", ""),
                    task_spec=r["task_spec"],
                )
                for r in raw["routes"]
            ]

        return PlanDecision(
            action=raw.get("action", "solve_directly"),
            rationale=raw.get("rationale", ""),
            children=children,
            routes=routes,
            file_scope=raw.get("file_scope"),
        )

    def _parse_execution_result(self, raw: dict[str, Any]) -> ExecutionResult:
        request = None
        if raw.get("status") == "request_upstream":
            request_raw = raw.get("request", {})
            request = UpstreamRequest(
                kind=request_raw.get("kind", "clarification"),
                summary=request_raw.get("summary", raw.get("summary", raw.get("details", ""))),
                details=request_raw.get("details", raw.get("details", "")),
                action_requested=request_raw.get("action_requested", raw.get("action_requested", "")),
                requires_input=request_raw.get("requires_input", True),
                urgency=request_raw.get("urgency", "normal"),
            )

        blocker = None
        if raw.get("status", "implemented") == "blocked":
            blocker_raw = raw.get("blocker") or {}
            legacy_escalation = raw.get("escalation") or blocker_raw.get("escalation")
            escalation = None
            if legacy_escalation:
                escalation = EscalationInfo(
                    kind=legacy_escalation.get("kind", blocker_raw.get("kind", raw.get("kind", "blocked"))),
                    summary=legacy_escalation.get("summary", blocker_raw.get("summary", raw.get("details", ""))),
                    details=legacy_escalation.get("details", blocker_raw.get("details", raw.get("details", ""))),
                    action_requested=legacy_escalation.get("action_requested", raw.get("action_requested", "")),
                )
            elif blocker_raw.get("needs_user_input") or raw.get("needs_user_input"):
                escalation = EscalationInfo(
                    kind=blocker_raw.get("kind", raw.get("kind", "blocked")),
                    summary=blocker_raw.get("summary", raw.get("summary", raw.get("details", ""))),
                    details=blocker_raw.get("details", raw.get("details", "")),
                    action_requested=raw.get("action_requested", ""),
                )

            blocker = BlockerInfo(
                kind=blocker_raw.get("kind", raw.get("kind", "blocked")),
                recoverable=blocker_raw.get("recoverable", raw.get("recoverable", True)),
                details=blocker_raw.get("details", raw.get("details", "")),
                needs_user_input=blocker_raw.get("needs_user_input", raw.get("needs_user_input", False)),
                owner=blocker_raw.get("owner", raw.get("owner", "root" if escalation else "node")),
                urgency=blocker_raw.get("urgency", raw.get("urgency", "normal")),
                escalation=escalation,
            )
        return ExecutionResult(
            status=raw.get("status", "implemented"),
            summary=raw.get("summary", ""),
            changed_files=raw.get("changed_files"),
            commit_sha=raw.get("result_commit_sha") or raw.get("commit_sha"),
            request=request,
            blocker=blocker,
        )

    def _parse_review_verdict(self, raw: dict[str, Any], child_id: str) -> ReviewVerdict:
        return ReviewVerdict(
            child_id=raw.get("child_id", child_id),
            verdict=raw.get("verdict", "reject"),
            reason=raw.get("reason", ""),
            follow_up=raw.get("follow_up", ""),
        )

    def _blocker_from_dict(self, raw: dict[str, Any]) -> BlockerInfo:
        escalation = raw.get("escalation")
        return BlockerInfo(
            kind=raw.get("kind", "blocked"),
            recoverable=raw.get("recoverable", True),
            details=raw.get("details", ""),
            needs_user_input=raw.get("needs_user_input", False),
            owner=raw.get("owner", "node"),
            urgency=raw.get("urgency", "normal"),
            escalation=EscalationInfo(
                kind=escalation.get("kind", raw.get("kind", "blocked")),
                summary=escalation.get("summary", raw.get("details", "")),
                details=escalation.get("details", raw.get("details", "")),
                action_requested=escalation.get("action_requested", ""),
            ) if escalation else None,
        )

    def _request_from_dict(self, raw: dict[str, Any]) -> UpstreamRequest:
        return UpstreamRequest(
            kind=raw.get("kind", "clarification"),
            summary=raw.get("summary", raw.get("details", "")),
            details=raw.get("details", ""),
            action_requested=raw.get("action_requested", ""),
            requires_input=raw.get("requires_input", raw.get("needs_user_input", False)),
            urgency=raw.get("urgency", "normal"),
            request_id=raw.get("request_id", ""),
        )


def _merge_base(worktree: Path, sha_a: str, sha_b: str) -> str | None:
    result = subprocess.run(
        ["git", "merge-base", sha_a, sha_b],
        cwd=str(worktree), capture_output=True, text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _cost_to_dict(cost) -> dict[str, Any]:
    if cost is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_usd": 0.0}
    return {
        "input_tokens": cost.input_tokens,
        "output_tokens": cost.output_tokens,
        "total_usd": cost.total_usd,
    }


def _run_test_in_worktree(
    test_command: str, worktree: Path, timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    """Run a test command in a worktree directory."""
    return run_test_command(
        test_command,
        worktree,
        timeout_seconds=timeout,
    )


def _truncate_test_output(stdout: str, stderr: str, max_chars: int = 4000) -> str:
    """Combine and truncate test output for the retry prompt."""
    combined = ""
    if stderr.strip():
        combined += stderr.strip()
    if stdout.strip():
        if combined:
            combined += "\n\n"
        combined += stdout.strip()
    if len(combined) > max_chars:
        # Keep the tail (most useful part — test summary and failures)
        combined = "... (truncated)\n" + combined[-max_chars:]
    return combined


def get_node_tree(store: StateStore, run_id: str) -> list[dict[str, Any]]:
    """Build a tree representation of all nodes in a run."""
    nodes = store.get_run_nodes(run_id)
    node_map = {n.node_id: n for n in nodes}

    def build(node_id: str, depth: int = 0) -> dict[str, Any]:
        node = node_map[node_id]
        children = [n for n in nodes if n.parent_id == node_id]
        domain = store.get_domain_by_child(node_id)
        return {
            "node_id": node.node_id,
            "state": node.state.value,
            "task_spec": node.task_spec[:80],
            "domain": domain.domain_name if domain else None,
            "depth": depth,
            "children": [build(c.node_id, depth + 1) for c in children],
        }

    run = store.get_run(run_id)
    if run and run.root_node_id:
        return [build(run.root_node_id)]
    return []
