"""Event-driven orchestrator – the runtime's main loop.

The orchestrator never reasons about tasks. It only:
- creates worktrees
- launches or resumes Claude sessions via the adapter
- persists events and state
- delivers child-result summaries to parents
- performs git integration requested by parent decisions
"""

from __future__ import annotations

import json
import logging
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
)
from recursive_intelligence.runtime.node_fsm import (
    ChildSpec,
    ExecutionResult,
    NodeFSM,
    PlanDecision,
    ReviewVerdict,
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

    async def start_run(self, task: str) -> str:
        """Start a new recursive run. Returns the run_id."""
        ensure_clean_repo(self.config.repo_root)
        store = self._ensure_store()

        run = store.create_run(str(self.config.repo_root), task)
        log.info("Created run %s", run.run_id)

        # Create root node with its own worktree
        b_name = branch_name(run.run_id, "root", task)
        wt_path = create_worktree(
            self.config.repo_root,
            self.config.worktrees_dir,
            f"{run.run_id}-root",
            b_name,
        )

        root = store.create_node(
            run_id=run.run_id,
            task_spec=task,
            worktree_path=str(wt_path),
            branch_name=b_name,
        )
        store.set_root_node(run.run_id, root.node_id)
        log.info("Created root node %s at %s", root.node_id, wt_path)

        try:
            await self._drive_node(root.node_id)
        except Exception as e:
            log.error("Run %s failed: %s", run.run_id, e)
            # Try to fail the root node gracefully
            try:
                fsm = NodeFSM(store, root.node_id)
                if not fsm.node.state.is_terminal:
                    fsm.fail(str(e), failure_type="orchestrator_error")
            except Exception:
                pass

        # Determine final run status
        root = store.get_node(root.node_id)
        final_status = "completed" if root.state == NodeState.COMPLETED else "failed"
        store.finish_run(run.run_id, final_status)

        return run.run_id

    async def resume_run(self, run_id: str) -> None:
        """Resume an existing run by finding and driving incomplete nodes."""
        store = self._ensure_store()
        run = store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        if run.status != "running":
            raise ValueError(f"Run {run_id} is {run.status}, not resumable")

        # Drive any non-terminal nodes, starting from leaves up
        resumable = [
            NodeState.QUEUED,
            NodeState.PLANNING,
            NodeState.EXECUTING,
            NodeState.WAITING_ON_CHILDREN,
            NodeState.REVIEWING_CHILDREN,
            NodeState.MERGING,
        ]
        for state in resumable:
            nodes = store.get_nodes_in_state(run_id, state)
            for node in nodes:
                await self._drive_node(node.node_id)

    async def _drive_node(self, node_id: str) -> None:
        """Drive a single node through its lifecycle.

        Uses a loop to handle revise cycles: after a review sends a child
        back for revision, the parent transitions through
        WAITING_ON_CHILDREN → REVIEWING_CHILDREN again, which this loop
        picks up automatically.
        """
        store = self._ensure_store()
        fsm = NodeFSM(store, node_id)

        while True:
            node = fsm.node
            log.info("Driving node %s (state=%s)", node_id, node.state.value)

            if node.state.is_terminal:
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

            # Safety: if no progress was made, break to avoid infinite loop
            if fsm.node.state == prev_state:
                log.warning("Node %s stuck in %s, breaking", node_id, prev_state.value)
                break

    async def _run_planning(self, fsm: NodeFSM) -> None:
        """Run the planning phase: ask the adapter for a plan decision."""
        node = fsm.node
        store = self._ensure_store()
        worktree = Path(node.worktree_path) if node.worktree_path else self.config.repo_root

        prompt = planning_prompt(node.task_spec)
        try:
            result = await self.adapter.run(prompt=prompt, worktree=worktree, mode="plan")
        except Exception as e:
            log.error("Planning failed for %s: %s", node.node_id, e)
            fsm.fail(str(e), failure_type="adapter_error")
            return

        store.update_node(node.node_id, session_id=result.session_id)
        store.create_session(result.session_id, node.node_id, self.adapter.name)
        store.finish_session(result.session_id)

        store.append_event(node.run_id, node.node_id, "plan_result", {
            "session_id": result.session_id,
            "raw": result.raw,
        })

        decision = self._parse_plan_decision(result.raw)
        log.info("Node %s plan: %s (%s)", node.node_id, decision.action, decision.rationale[:80])
        fsm.apply_plan_decision(decision)

    async def _run_execution(self, fsm: NodeFSM) -> None:
        """Run the execution phase."""
        node = fsm.node
        store = self._ensure_store()
        worktree = Path(node.worktree_path) if node.worktree_path else self.config.repo_root

        # Check if this is a revision (child got revise feedback)
        events = store.get_node_events(node.node_id)
        revision_events = [e for e in events if e.event_type == "revision_requested"]

        if revision_events:
            latest_revision = revision_events[-1]
            prompt = revision_prompt(latest_revision.data.get("follow_up", ""))
            resume_id = node.session_id
        else:
            prompt = execution_prompt(node.task_spec)
            resume_id = None

        try:
            result = await self.adapter.run(
                prompt=prompt,
                worktree=worktree,
                mode="execute",
                resume_session_id=resume_id,
            )
        except Exception as e:
            log.error("Execution failed for %s: %s", node.node_id, e)
            fsm.fail(str(e), failure_type="adapter_error")
            return

        store.update_node(node.node_id, session_id=result.session_id)
        store.append_event(node.run_id, node.node_id, "execution_result", {
            "session_id": result.session_id,
            "raw": result.raw,
        })

        execution = self._parse_execution_result(result.raw)
        log.info("Node %s execution: %s", node.node_id, execution.status)
        fsm.finish_execution(execution)

    async def _wait_and_drive_children(self, fsm: NodeFSM) -> None:
        """Drive all children, then wake the parent for review."""
        store = self._ensure_store()
        children = store.get_children(fsm.node_id)

        if not children:
            log.warning("Node %s waiting on children but has none, failing", fsm.node_id)
            fsm.fail("No children found", failure_type="orchestrator_error")
            return

        for child in children:
            # Check if a completed child has a pending revision
            if child.state == NodeState.COMPLETED and self._has_pending_revision(child):
                log.info("Child %s has pending revision, re-executing", child.node_id)
                store.transition_node(child.node_id, NodeState.EXECUTING, {
                    "reason": "revision_requested",
                })
            elif child.state.is_terminal:
                continue

            # Create worktree for child if needed
            if not child.worktree_path:
                parent_node = fsm.node
                parent_wt = Path(parent_node.worktree_path) if parent_node.worktree_path else self.config.repo_root

                b_name = branch_name(child.run_id, child.node_id, child.task_spec)
                parent_head = get_head_sha(parent_wt)

                wt_path = create_worktree(
                    self.config.repo_root,
                    self.config.worktrees_dir,
                    child.node_id,
                    b_name,
                    base_ref=parent_head,
                )
                store.update_node(child.node_id, worktree_path=str(wt_path), branch_name=b_name)

            try:
                await self._drive_node(child.node_id)
            except Exception as e:
                log.error("Child %s failed: %s", child.node_id, e)
                child_fsm = NodeFSM(store, child.node_id)
                if not child_fsm.node.state.is_terminal:
                    child_fsm.fail(str(e), failure_type="orchestrator_error")

        # All children driven — wake parent for review
        fsm.wake_for_review()

    def _has_pending_revision(self, child) -> bool:
        """Check if a child has a revision_requested event after its last execution_result."""
        store = self._ensure_store()
        events = store.get_node_events(child.node_id)
        last_exec_idx = -1
        last_rev_idx = -1
        for i, e in enumerate(events):
            if e.event_type == "execution_result":
                last_exec_idx = i
            elif e.event_type == "revision_requested":
                last_rev_idx = i
        return last_rev_idx > last_exec_idx

    async def _run_review(self, fsm: NodeFSM) -> None:
        """Review each completed child's work."""
        store = self._ensure_store()
        node = fsm.node
        children = store.get_children(node.node_id)

        # Collect success criteria from spawn events
        spawn_events = [
            e for e in store.get_node_events(node.node_id)
            if e.event_type == "child_spawned"
        ]
        criteria_by_child: dict[str, list[str]] = {}
        for evt in spawn_events:
            criteria_by_child[evt.data.get("child_id", "")] = evt.data.get("success_criteria", [])

        for child in children:
            if child.state != NodeState.COMPLETED or not child.worktree_path:
                # Failed/cancelled children get auto-rejected
                if child.state in (NodeState.FAILED, NodeState.CANCELLED):
                    verdict = ReviewVerdict(
                        child_id=child.node_id,
                        verdict="reject",
                        reason=f"Child {child.state.value}",
                    )
                    fsm.apply_review_verdict(verdict)
                    if fsm.node.state == NodeState.WAITING_ON_CHILDREN:
                        return
                continue

            # Find the base ref for a full diff (all child commits, not just last one)
            parent_wt = Path(node.worktree_path) if node.worktree_path else self.config.repo_root
            child_wt = Path(child.worktree_path)
            base_sha = _merge_base(parent_wt, get_head_sha(parent_wt), get_head_sha(child_wt))
            diff_base = base_sha or "HEAD~1"

            # Pull the child's execution summary from its events
            child_events = store.get_node_events(child.node_id)
            child_summary = ""
            for evt in reversed(child_events):
                if evt.event_type == "execution_result":
                    child_summary = evt.data.get("raw", {}).get("summary", "")
                    break

            bundle = build_artifact_bundle(
                node_id=child.node_id,
                worktree=child_wt,
                base_ref=diff_base,
                summary=child_summary,
            )

            criteria = criteria_by_child.get(child.node_id, [])
            worktree = parent_wt

            prompt = build_review_prompt_template(
                child_id=child.node_id,
                diff=bundle.diff[:8000],
                summary=bundle.summary or child_summary,
                success_criteria=criteria,
            )

            try:
                result = await self.adapter.run(
                    prompt=prompt,
                    worktree=worktree,
                    mode="review",
                    resume_session_id=node.session_id,
                )
            except Exception as e:
                log.error("Review failed for %s: %s", child.node_id, e)
                # On review failure, accept the child's work to avoid blocking
                verdict = ReviewVerdict(
                    child_id=child.node_id,
                    verdict="accept",
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
                # Reset child to executing for another round
                # (The child FSM transition will be handled when we drive it again)

            fsm.apply_review_verdict(verdict)

            # If parent went back to waiting (revise), let _drive_node loop handle it
            if fsm.node.state == NodeState.WAITING_ON_CHILDREN:
                return

    async def _run_merge(self, fsm: NodeFSM) -> None:
        """Cherry-pick accepted children's commits into the parent worktree."""
        store = self._ensure_store()
        node = fsm.node
        parent_wt = Path(node.worktree_path) if node.worktree_path else self.config.repo_root

        events = store.get_node_events(node.node_id)
        accepted_ids = {
            e.data["child_id"]
            for e in events
            if e.event_type == "review_verdict" and e.data.get("verdict") == "accept"
        }

        if not accepted_ids:
            log.warning("No accepted children for %s, completing without merge", node.node_id)
            final_sha = get_head_sha(parent_wt)
            fsm.finish_merge(final_sha)
            return

        children = store.get_children(node.node_id)
        for child in children:
            if child.node_id not in accepted_ids or not child.worktree_path:
                continue

            child_wt = Path(child.worktree_path)
            child_sha = get_head_sha(child_wt)
            parent_sha = get_head_sha(parent_wt)

            # Find the merge base (where child branched from parent)
            base_sha = _merge_base(parent_wt, parent_sha, child_sha)

            # Skip if child hasn't diverged from base
            if child_sha == base_sha:
                log.info("Child %s has no new commits, skipping", child.node_id)
                store.append_event(node.run_id, node.node_id, "child_integrated", {
                    "child_id": child.node_id,
                    "status": "no_change",
                    "commit_sha": child_sha,
                })
                continue

            result = cherry_pick_child(parent_wt, child_sha, child.node_id, base_sha=base_sha)

            store.append_event(node.run_id, node.node_id, "child_integrated", {
                "child_id": child.node_id,
                "status": result.status,
                "commit_sha": result.commit_sha,
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
        fsm.finish_merge(final_sha)
        log.info("Node %s merged at %s", node.node_id, final_sha[:8])

    def cleanup_worktrees(self, run_id: str) -> None:
        """Remove all worktrees for a completed run."""
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
                )
                for i, c in enumerate(raw["children"])
            ]
        return PlanDecision(
            action=raw.get("action", "solve_directly"),
            rationale=raw.get("rationale", ""),
            children=children,
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
    """Find the merge base between two commits."""
    import subprocess

    result = subprocess.run(
        ["git", "merge-base", sha_a, sha_b],
        cwd=str(worktree),
        capture_output=True,
        text=True,
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
        return {
            "node_id": node.node_id,
            "state": node.state.value,
            "task_spec": node.task_spec[:80],
            "depth": depth,
            "children": [build(c.node_id, depth + 1) for c in children],
        }

    run = store.get_run(run_id)
    if run and run.root_node_id:
        return [build(run.root_node_id)]
    return []
