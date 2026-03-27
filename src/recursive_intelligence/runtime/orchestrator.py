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

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from recursive_intelligence.adapters.base import AgentAdapter
from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.git.diffing import ArtifactBundle, build_artifact_bundle
from recursive_intelligence.git.merge import cherry_pick_child, abort_cherry_pick
from recursive_intelligence.git.worktrees import (
    branch_name,
    create_worktree,
    ensure_clean_repo,
    get_head_sha,
    remove_worktree,
)
from recursive_intelligence.adapters.claude.prompts import (
    planning_prompt,
    execution_prompt,
    review_prompt as build_review_prompt_template,
    revision_prompt,
    routing_prompt,
    reactivation_prompt,
)
from recursive_intelligence.runtime.node_fsm import (
    ChildSpec,
    ExecutionResult,
    NodeFSM,
    PlanDecision,
    ReviewVerdict,
    RouteSpec,
)
from recursive_intelligence.runtime.state_store import NodeState, StateStore

log = logging.getLogger(__name__)


class Orchestrator:
    """Drives the recursive runtime."""

    def __init__(self, config: RuntimeConfig, adapter: AgentAdapter) -> None:
        self.config = config
        self.adapter = adapter
        self.store: StateStore | None = None

    def _ensure_store(self) -> StateStore:
        if self.store is None:
            self.config.ensure_dirs()
            self.store = StateStore(self.config.db_path)
        return self.store

    # --- Run lifecycle ---

    async def start_run(self, task: str, persistent: bool = False) -> str:
        """Start a new run. Returns the run_id.

        If persistent=True, the run pauses after the first pass instead
        of completing, allowing follow-up via continue_run().
        """
        ensure_clean_repo(self.config.repo_root)
        store = self._ensure_store()

        run = store.create_run(str(self.config.repo_root), task, persistent=persistent)
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

        store.resume_paused_run(run_id)

        root = store.get_node(run.root_node_id)
        if root is None or root.state != NodeState.PAUSED:
            raise ValueError(f"Root node is not paused (state={root.state.value if root else 'missing'})")

        # Resume the root with the new input
        fsm = NodeFSM(store, root.node_id)
        fsm.resume_from_pause(user_input)

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
            NodeState.MERGING,
        ]
        for state in resumable:
            nodes = store.get_nodes_in_state(run_id, state)
            for node in nodes:
                await self._drive_node(node.node_id)

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

        # Check if this is a routing pass (multi-pass follow-up on root or parent)
        is_routing = run and run.pass_count > 1 and len(store.get_children(node.node_id)) > 0

        if is_routing:
            prompt = self._build_routing_prompt(node, run)
        else:
            prompt = planning_prompt(node.task_spec)

        try:
            result = await self.adapter.run(
                prompt=prompt, worktree=worktree, mode="plan",
                resume_session_id=node.session_id if is_routing else None,
            )
        except Exception as e:
            log.error("Planning failed for %s: %s", node.node_id, e)
            fsm.fail(str(e), failure_type="adapter_error")
            return

        store.update_node(node.node_id, session_id=result.session_id)
        store.create_session(result.session_id, node.node_id, self.adapter.name)
        store.finish_session(result.session_id)

        store.append_event(node.run_id, node.node_id, "plan_result", {
            "session_id": result.session_id, "raw": result.raw,
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
            if child:
                child_events = store.get_node_events(child.node_id)
                for evt in reversed(child_events):
                    if evt.event_type == "execution_result":
                        child_summary = evt.data.get("raw", {}).get("summary", "")
                        break
            domain_dicts.append({
                "domain_name": d.domain_name,
                "child_node_id": d.child_node_id,
                "file_patterns": d.file_patterns,
                "module_scope": d.module_scope,
                "child_state": child.state.value if child else "unknown",
                "last_summary": child_summary,
            })

        return routing_prompt(user_input, domain_dicts, run.pass_count)

    def _prepare_child_reactivation(self, parent_node, route: RouteSpec) -> None:
        """Prepare a child for reactivation with a new task."""
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

        # Store the reactivation event (will be picked up by _run_execution)
        store.append_event(child.run_id, child.node_id, "reactivation_requested", {
            "new_task": route.task_spec,
            "previous_summary": prev_summary,
            "original_task": child.task_spec,
        })

    # --- Execution ---

    async def _run_execution(self, fsm: NodeFSM) -> None:
        node = fsm.node
        store = self._ensure_store()
        worktree = Path(node.worktree_path) if node.worktree_path else self.config.repo_root

        events = store.get_node_events(node.node_id)

        # Find the most recent pending work event AFTER the last execution_result.
        # This ensures revisions override stale reactivation events and vice versa.
        last_exec_idx = -1
        for i, e in enumerate(events):
            if e.event_type == "execution_result":
                last_exec_idx = i

        pending_event = None
        for e in events[last_exec_idx + 1:]:
            if e.event_type in ("reactivation_requested", "revision_requested"):
                pending_event = e  # last one wins

        if pending_event and pending_event.event_type == "reactivation_requested":
            prompt = reactivation_prompt(
                original_task=pending_event.data.get("original_task", node.task_spec),
                previous_summary=pending_event.data.get("previous_summary", ""),
                new_task=pending_event.data.get("new_task", ""),
            )
            resume_id = node.session_id
        elif pending_event and pending_event.event_type == "revision_requested":
            prompt = revision_prompt(pending_event.data.get("follow_up", ""))
            resume_id = node.session_id
        else:
            prompt = execution_prompt(node.task_spec)
            resume_id = None

        try:
            result = await self.adapter.run(
                prompt=prompt, worktree=worktree, mode="execute",
                resume_session_id=resume_id,
            )
        except Exception as e:
            log.error("Execution failed for %s: %s", node.node_id, e)
            fsm.fail(str(e), failure_type="adapter_error")
            return

        store.update_node(node.node_id, session_id=result.session_id)
        store.append_event(node.run_id, node.node_id, "execution_result", {
            "session_id": result.session_id, "raw": result.raw,
        })

        # Auto-update domain registry with actual changed files
        execution = self._parse_execution_result(result.raw)
        if execution.changed_files:
            self._update_domain_from_changed_files(node, execution.changed_files)

        log.info("Node %s execution: %s", node.node_id, execution.status)
        fsm.finish_execution(execution)

    def _update_domain_from_changed_files(self, node, changed_files: list[str]) -> None:
        """Auto-update domain file_patterns from actual files touched."""
        store = self._ensure_store()
        domain = store.get_domain_by_child(node.node_id)
        if domain and changed_files:
            existing = set(domain.file_patterns)
            existing.update(changed_files)
            store.update_domain(domain.domain_id, file_patterns=sorted(existing))

    # --- Children ---

    async def _wait_and_drive_children(self, fsm: NodeFSM) -> None:
        store = self._ensure_store()
        children = store.get_children(fsm.node_id)

        if not children:
            log.warning("Node %s waiting on children but has none, failing", fsm.node_id)
            fsm.fail("No children found", failure_type="orchestrator_error")
            return

        for child in children:
            # Check for pending revision
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

            # Create worktree for child if needed
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

            try:
                await self._drive_node(child.node_id)
            except Exception as e:
                log.error("Child %s failed: %s", child.node_id, e)
                child_fsm = NodeFSM(store, child.node_id)
                if not child_fsm.node.state.is_idle:
                    child_fsm.fail(str(e), failure_type="orchestrator_error")

        fsm.wake_for_review()

    def _has_pending_work(self, child) -> bool:
        """Check if a child has pending revision or reactivation."""
        store = self._ensure_store()
        events = store.get_node_events(child.node_id)
        last_exec_idx = -1
        last_work_idx = -1
        for i, e in enumerate(events):
            if e.event_type == "execution_result":
                last_exec_idx = i
            elif e.event_type in ("revision_requested", "reactivation_requested"):
                last_work_idx = i
        return last_work_idx > last_exec_idx

    # --- Review ---

    async def _run_review(self, fsm: NodeFSM) -> None:
        store = self._ensure_store()
        node = fsm.node
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

            if child.state != NodeState.COMPLETED or not child.worktree_path:
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

            bundle = build_artifact_bundle(
                node_id=child.node_id, worktree=child_wt,
                base_ref=diff_base, summary=child_summary,
            )

            criteria = criteria_by_child.get(child.node_id, [])

            prompt = build_review_prompt_template(
                child_id=child.node_id, diff=bundle.diff[:8000],
                summary=bundle.summary or child_summary,
                success_criteria=criteria,
            )

            try:
                result = await self.adapter.run(
                    prompt=prompt, worktree=parent_wt, mode="review",
                    resume_session_id=node.session_id,
                )
            except Exception as e:
                log.error("Review failed for %s: %s", child.node_id, e)
                verdict = ReviewVerdict(
                    child_id=child.node_id, verdict="accept",
                    reason=f"Review error, auto-accepting: {e}",
                )
                fsm.apply_review_verdict(verdict)
                continue

            store.update_node(node.node_id, session_id=result.session_id)
            verdict = self._parse_review_verdict(result.raw, child.node_id)
            log.info("Node %s review of %s: %s", node.node_id, child.node_id, verdict.verdict)

            if verdict.verdict == "revise":
                store.append_event(child.run_id, child.node_id, "revision_requested", {
                    "follow_up": verdict.follow_up,
                })

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

            result = cherry_pick_child(parent_wt, child_sha, child.node_id, base_sha=base_sha)

            store.append_event(node.run_id, node.node_id, "child_integrated", {
                "child_id": child.node_id, "status": result.status,
                "commit_sha": result.commit_sha,
                "child_head_sha": child_sha,  # track for multi-pass
                "conflict_files": result.conflict_files,
            })

            if result.status == "conflict":
                log.warning("Conflict integrating %s: %s", child.node_id, result.conflict_files)
                abort_cherry_pick(parent_wt)
                fsm.fail(
                    f"Cherry-pick conflict with {child.node_id}",
                    failure_type="merge_conflict",
                )
                return

        final_sha = get_head_sha(parent_wt)
        if run and run.persistent:
            fsm.pause_after_merge(final_sha)
            log.info("Node %s merged and paused at %s", node.node_id, final_sha[:8])
        else:
            fsm.finish_merge(final_sha)
            log.info("Node %s merged at %s", node.node_id, final_sha[:8])

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
        return ExecutionResult(
            status=raw.get("status", "implemented"),
            summary=raw.get("summary", ""),
            changed_files=raw.get("changed_files"),
            commit_sha=raw.get("result_commit_sha") or raw.get("commit_sha"),
        )

    def _parse_review_verdict(self, raw: dict[str, Any], child_id: str) -> ReviewVerdict:
        return ReviewVerdict(
            child_id=raw.get("child_id", child_id),
            verdict=raw.get("verdict", "reject"),
            reason=raw.get("reason", ""),
            follow_up=raw.get("follow_up", ""),
        )


def _merge_base(worktree: Path, sha_a: str, sha_b: str) -> str | None:
    result = subprocess.run(
        ["git", "merge-base", sha_a, sha_b],
        cwd=str(worktree), capture_output=True, text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


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
