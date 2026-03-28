"""Tests for the baseline runner."""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult
from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.baseline import BaselineRunner
from recursive_intelligence.runtime.state_store import NodeState, StateStore


class MockAdapter(AgentAdapter):
    """Mock adapter that simulates a successful execution."""

    def __init__(self, result_text: str = "", raw: dict = None, commit: bool = True):
        self._result_text = result_text
        self._raw = raw or {}
        self._commit = commit

    @property
    def name(self) -> str:
        return "mock"

    async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None):
        # Simulate making a change if requested
        if self._commit and mode == "execute":
            test_file = Path(worktree) / "output.txt"
            test_file.write_text("mock output\n")
            subprocess.run(["git", "add", "."], cwd=str(worktree), capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "mock change"],
                cwd=str(worktree),
                capture_output=True,
            )

        return NodeResult(
            session_id="mock-session-123",
            raw=self._raw,
            result_text=self._result_text,
            cost=CostRecord(input_tokens=1000, output_tokens=500, total_usd=0.035),
            stop_reason="end_turn",
        )


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo."""
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
    return RuntimeConfig(repo_root=git_repo)


class TestBaselineRunner:
    @pytest.mark.asyncio
    async def test_successful_run(self, config, git_repo):
        adapter = MockAdapter(
            result_text='{"status": "implemented", "summary": "done"}',
            raw={"status": "implemented", "summary": "done"},
            commit=True,
        )
        runner = BaselineRunner(config, adapter)

        with patch("recursive_intelligence.runtime.baseline._capture_transcript", return_value=None):
            report = await runner.run("add a feature")

        assert report.status == "completed"
        assert report.session_id == "mock-session-123"
        assert report.cost.total_usd == 0.035
        assert report.cost.input_tokens == 1000
        assert report.cost.output_tokens == 500
        assert report.run_id.startswith("run-")
        assert report.branch_name.startswith("ri/")
        assert len(report.changed_files) > 0  # mock adapter commits output.txt

    @pytest.mark.asyncio
    async def test_run_creates_db_records(self, config, git_repo):
        adapter = MockAdapter(commit=True)
        runner = BaselineRunner(config, adapter)

        with patch("recursive_intelligence.runtime.baseline._capture_transcript", return_value=None):
            report = await runner.run("fix bug")

        store = StateStore(config.db_path)
        run = store.get_run(report.run_id)
        assert run is not None
        assert run.status == "completed"

        node = store.get_node(report.node_id)
        assert node is not None
        assert node.state == NodeState.COMPLETED
        assert node.session_id == "mock-session-123"
        store.close()

    @pytest.mark.asyncio
    async def test_run_saves_report(self, config, git_repo):
        adapter = MockAdapter(commit=True)
        runner = BaselineRunner(config, adapter)

        with patch("recursive_intelligence.runtime.baseline._capture_transcript", return_value=None):
            report = await runner.run("refactor module")

        report_path = config.artifacts_dir / report.run_id / "report.json"
        assert report_path.exists()

        import json
        saved = json.loads(report_path.read_text())
        assert saved["run_id"] == report.run_id
        assert saved["status"] == "completed"
        assert saved["cost"]["total_usd"] == 0.035

    @pytest.mark.asyncio
    async def test_run_no_changes(self, config, git_repo):
        adapter = MockAdapter(commit=False)  # no commit made
        runner = BaselineRunner(config, adapter)

        with patch("recursive_intelligence.runtime.baseline._capture_transcript", return_value=None):
            report = await runner.run("analyze code")

        # Still completes even with no changes
        assert report.status == "completed"
        assert report.changed_files == []
        assert report.base_sha == report.final_sha

    @pytest.mark.asyncio
    async def test_worktree_isolation(self, config, git_repo):
        """Baseline runs in a worktree, main repo stays clean."""
        adapter = MockAdapter(commit=True)
        runner = BaselineRunner(config, adapter)

        with patch("recursive_intelligence.runtime.baseline._capture_transcript", return_value=None):
            report = await runner.run("add output file")

        # Main repo should not have output.txt
        assert not (git_repo / "output.txt").exists()
        # But the worktree should
        wt_path = config.worktrees_dir / f"{report.run_id}-baseline"
        assert (wt_path / "output.txt").exists()

    @pytest.mark.asyncio
    async def test_dirty_repo_fails(self, config, git_repo):
        (git_repo / "README.md").write_text("modified tracked file")
        adapter = MockAdapter()
        runner = BaselineRunner(config, adapter)

        from recursive_intelligence.git.worktrees import WorktreeError

        with pytest.raises(WorktreeError, match="uncommitted"):
            await runner.run("task")
