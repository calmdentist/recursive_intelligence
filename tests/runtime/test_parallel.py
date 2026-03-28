"""Tests for parallel child execution."""

import asyncio
import subprocess
import time
from pathlib import Path

import pytest

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult
from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.orchestrator import Orchestrator
from recursive_intelligence.runtime.state_store import NodeState, StateStore


class TimedAdapter(AgentAdapter):
    """Mock adapter that tracks concurrency via timestamps."""

    def __init__(self, responses: list[dict], delay: float = 0.1) -> None:
        self._responses = list(responses)
        self._call_log: list[dict] = []
        self._delay = delay
        self._concurrent_peak = 0
        self._concurrent_now = 0
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "timed"

    @property
    def peak_concurrency(self) -> int:
        return self._concurrent_peak

    async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None, is_root=False):
        async with self._lock:
            self._concurrent_now += 1
            if self._concurrent_now > self._concurrent_peak:
                self._concurrent_peak = self._concurrent_now

        try:
            # Simulate I/O delay (this is where real Claude calls would await)
            await asyncio.sleep(self._delay)

            if not self._responses:
                raise RuntimeError("Out of responses")
            resp = self._responses.pop(0)

            self._call_log.append({"mode": mode, "worktree": str(worktree)})

            if resp.get("_commit"):
                _make_commit(Path(worktree), resp.get("_commit_file", "out.txt"), resp.get("_commit_msg", "mock"))

            raw = {k: v for k, v in resp.items() if not k.startswith("_")}
            return NodeResult(
                session_id=f"session-{len(self._call_log)}",
                raw=raw, result_text="", cost=CostRecord(total_usd=0.01),
                stop_reason="end_turn",
            )
        finally:
            async with self._lock:
                self._concurrent_now -= 1


def _make_commit(worktree: Path, filename: str, message: str) -> None:
    (worktree / filename).write_text(f"content of {filename}\n")
    subprocess.run(["git", "add", "."], cwd=str(worktree), capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=str(worktree), capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo


@pytest.fixture
def config(git_repo):
    return RuntimeConfig(repo_root=git_repo, max_parallel_children=0)


class TestParallelChildren:
    @pytest.mark.asyncio
    async def test_children_run_concurrently(self, config):
        """Three children should run in parallel, not serially."""
        adapter = TimedAdapter([
            # Root: spawn 3 children
            {"action": "spawn_children", "rationale": "parallel test",
             "children": [
                 {"idempotency_key": "a", "objective": "task A", "success_criteria": ["ok"]},
                 {"idempotency_key": "b", "objective": "task B", "success_criteria": ["ok"]},
                 {"idempotency_key": "c", "objective": "task C", "success_criteria": ["ok"]},
             ]},
            # Child A: plan + execute
            {"action": "solve_directly", "rationale": "ok"},
            # Child B: plan + execute
            {"action": "solve_directly", "rationale": "ok"},
            # Child C: plan + execute
            {"action": "solve_directly", "rationale": "ok"},
            # Child A execute
            {"status": "implemented", "summary": "A done",
             "_commit": True, "_commit_file": "a.txt", "_commit_msg": "a"},
            # Child B execute
            {"status": "implemented", "summary": "B done",
             "_commit": True, "_commit_file": "b.txt", "_commit_msg": "b"},
            # Child C execute
            {"status": "implemented", "summary": "C done",
             "_commit": True, "_commit_file": "c.txt", "_commit_msg": "c"},
            # Root reviews all 3
            {"verdict": "accept", "reason": "ok"},
            {"verdict": "accept", "reason": "ok"},
            {"verdict": "accept", "reason": "ok"},
        ], delay=0.15)

        orch = Orchestrator(config, adapter)
        start = time.monotonic()
        run_id = await orch.start_run("parallel test")
        elapsed = time.monotonic() - start

        store = StateStore(config.db_path)
        run = store.get_run(run_id)
        store.close()

        assert run.status == "completed"

        # Key assertion: peak concurrency > 1 proves children ran in parallel
        assert adapter.peak_concurrency >= 2, (
            f"Expected concurrent execution, got peak_concurrency={adapter.peak_concurrency}"
        )

        # Timing check: 10 calls total at 0.15s each.
        # Serial would be ~1.5s. Parallel children overlap, so total should be less.
        # But root plan (1 call) + serial reviews (3 calls) add ~0.6s of non-parallel time.
        # Generous bound to avoid flaky tests — the peak_concurrency check above
        # is the real proof of parallelism.
        assert elapsed < 2.5, f"Took {elapsed:.1f}s — unexpectedly slow"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, git_repo):
        """max_parallel_children=2 should limit concurrency to 2."""
        config = RuntimeConfig(repo_root=git_repo, max_parallel_children=2)

        adapter = TimedAdapter([
            {"action": "spawn_children", "rationale": "sem test",
             "children": [
                 {"idempotency_key": "a", "objective": "A", "success_criteria": ["ok"]},
                 {"idempotency_key": "b", "objective": "B", "success_criteria": ["ok"]},
                 {"idempotency_key": "c", "objective": "C", "success_criteria": ["ok"]},
             ]},
            {"action": "solve_directly", "rationale": "ok"},
            {"action": "solve_directly", "rationale": "ok"},
            {"action": "solve_directly", "rationale": "ok"},
            {"status": "implemented", "summary": "A",
             "_commit": True, "_commit_file": "a.txt", "_commit_msg": "a"},
            {"status": "implemented", "summary": "B",
             "_commit": True, "_commit_file": "b.txt", "_commit_msg": "b"},
            {"status": "implemented", "summary": "C",
             "_commit": True, "_commit_file": "c.txt", "_commit_msg": "c"},
            {"verdict": "accept", "reason": "ok"},
            {"verdict": "accept", "reason": "ok"},
            {"verdict": "accept", "reason": "ok"},
        ], delay=0.1)

        orch = Orchestrator(config, adapter)
        await orch.start_run("sem test")

        # With semaphore=2, peak should be at most 2
        assert adapter.peak_concurrency <= 2, (
            f"Expected max 2 concurrent, got {adapter.peak_concurrency}"
        )

    @pytest.mark.asyncio
    async def test_one_child_failure_doesnt_kill_siblings(self, config):
        """If child A fails, children B and C should still complete."""
        adapter = TimedAdapter([
            {"action": "spawn_children", "rationale": "error test",
             "children": [
                 {"idempotency_key": "a", "objective": "will fail", "success_criteria": ["ok"]},
                 {"idempotency_key": "b", "objective": "will succeed", "success_criteria": ["ok"]},
             ]},
            # Child A: plan then fail
            {"action": "solve_directly", "rationale": "ok"},
            # Child B: plan then succeed
            {"action": "solve_directly", "rationale": "ok"},
            # Child A: blocked
            {"status": "blocked", "kind": "test_failure", "details": "intentional"},
            # Child B: success
            {"status": "implemented", "summary": "B done",
             "_commit": True, "_commit_file": "b.txt", "_commit_msg": "b"},
            # Root reviews: reject A (failed), accept B
            {"verdict": "accept", "reason": "ok"},
        ], delay=0.05)

        orch = Orchestrator(config, adapter)
        run_id = await orch.start_run("error isolation test")

        store = StateStore(config.db_path)
        nodes = store.get_run_nodes(run_id)
        store.close()

        child_states = {n.task_spec: n.state for n in nodes if n.parent_id}
        assert child_states["will fail"] == NodeState.FAILED
        assert child_states["will succeed"] == NodeState.COMPLETED
