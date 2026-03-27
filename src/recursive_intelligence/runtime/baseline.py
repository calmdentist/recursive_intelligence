"""Baseline runner – single flat Claude session (no recursion).

This is the control group for benchmarks. It runs one Claude session
against a task in a managed worktree and captures everything:
session ID, result, cost, transcript, changed files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult
from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.git.worktrees import (
    branch_name,
    create_worktree,
    ensure_clean_repo,
    get_head_sha,
)
from recursive_intelligence.runtime.artifact_store import ArtifactStore
from recursive_intelligence.runtime.state_store import NodeState, StateStore

log = logging.getLogger(__name__)


@dataclass
class BaselineReport:
    """Complete report from a baseline run."""

    run_id: str
    node_id: str
    task: str
    session_id: str
    status: str  # completed, failed
    result_text: str
    cost: CostRecord
    num_turns: int
    duration_ms: int
    duration_api_ms: int
    stop_reason: str
    branch_name: str
    base_sha: str
    final_sha: str
    changed_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class BaselineRunner:
    """Run a single flat Claude session against a repo."""

    def __init__(self, config: RuntimeConfig, adapter: AgentAdapter) -> None:
        self.config = config
        self.adapter = adapter
        self._store: StateStore | None = None
        self._artifacts: ArtifactStore | None = None

    def _ensure_store(self) -> StateStore:
        if self._store is None:
            self.config.ensure_dirs()
            self._store = StateStore(self.config.db_path)
        return self._store

    def _ensure_artifacts(self) -> ArtifactStore:
        if self._artifacts is None:
            self.config.ensure_dirs()
            self._artifacts = ArtifactStore(self.config.artifacts_dir)
        return self._artifacts

    async def run(self, task: str) -> BaselineReport:
        """Execute a single flat session. Returns a complete report."""
        ensure_clean_repo(self.config.repo_root)
        store = self._ensure_store()
        artifacts = self._ensure_artifacts()

        # Create run + single node
        run = store.create_run(str(self.config.repo_root), task)
        log.info("Baseline run %s: %s", run.run_id, task[:80])

        b_name = branch_name(run.run_id, "baseline", task)
        base_sha = get_head_sha(self.config.repo_root)

        wt_path = create_worktree(
            self.config.repo_root,
            self.config.worktrees_dir,
            f"{run.run_id}-baseline",
            b_name,
        )

        node = store.create_node(
            run_id=run.run_id,
            task_spec=task,
            worktree_path=str(wt_path),
            branch_name=b_name,
        )
        store.set_root_node(run.run_id, node.node_id)

        # Transition: queued -> planning -> executing (baseline skips planning)
        store.transition_node(node.node_id, NodeState.PLANNING)
        store.transition_node(node.node_id, NodeState.EXECUTING)

        # Run the session
        log.info("Starting baseline session in %s", wt_path)
        result = await self.adapter.run(
            prompt=task,
            worktree=wt_path,
            mode="execute",
        )

        store.update_node(node.node_id, session_id=result.session_id)
        store.create_session(result.session_id, node.node_id, self.adapter.name)

        # Extract cost from ResultMessage fields stored in NodeResult
        cost = result.cost or CostRecord()

        # Get final state
        final_sha = get_head_sha(wt_path)
        changed = final_sha != base_sha
        changed_files = _get_changed_files(wt_path, base_sha)

        # Mark complete
        if result.stop_reason == "error" or (result.raw and result.raw.get("status") == "blocked"):
            store.transition_node(node.node_id, NodeState.FAILED, {
                "reason": result.result_text[:500],
            })
            status = "failed"
        else:
            store.transition_node(node.node_id, NodeState.COMPLETED, {
                "commit_sha": final_sha if changed else None,
                "changed_files": changed_files,
            })
            status = "completed"

        store.finish_session(result.session_id)
        store.finish_run(run.run_id, status)

        # Capture transcript
        transcript = _capture_transcript(result.session_id)
        if transcript:
            artifacts.save_transcript(run.run_id, node.node_id, result.session_id, transcript)
            log.info("Saved transcript: %d messages", len(transcript))

        # Build report
        report = BaselineReport(
            run_id=run.run_id,
            node_id=node.node_id,
            task=task,
            session_id=result.session_id,
            status=status,
            result_text=result.result_text,
            cost=cost,
            num_turns=result.raw.get("_num_turns", 0) if result.raw else 0,
            duration_ms=result.raw.get("_duration_ms", 0) if result.raw else 0,
            duration_api_ms=result.raw.get("_duration_api_ms", 0) if result.raw else 0,
            stop_reason=result.stop_reason,
            branch_name=b_name,
            base_sha=base_sha,
            final_sha=final_sha,
            changed_files=changed_files,
        )

        artifacts.save_report(run.run_id, report.to_dict())
        log.info("Baseline %s: status=%s, changed=%d files", run.run_id, status, len(changed_files))

        return report


def _get_changed_files(worktree: Path, base_sha: str) -> list[str]:
    """Get list of files changed between base_sha and HEAD in worktree."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_sha, "HEAD"],
            cwd=str(worktree),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().splitlines() if f]
    except Exception:
        pass
    return []


def _capture_transcript(session_id: str) -> list[dict[str, Any]] | None:
    """Capture the full transcript from a session via the SDK."""
    try:
        from claude_agent_sdk import get_session_messages

        messages = get_session_messages(session_id=session_id)
        return [
            {
                "type": msg.type,
                "uuid": msg.uuid,
                "message": msg.message,
            }
            for msg in messages
        ]
    except Exception as e:
        log.warning("Could not capture transcript for %s: %s", session_id, e)
        return None
