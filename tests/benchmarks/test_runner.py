"""End-to-end tests for the benchmark runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult
from recursive_intelligence.benchmarks.models import SWEBenchTask
from recursive_intelligence.benchmarks.reporting import export_report
from recursive_intelligence.benchmarks.runner import BenchmarkRunner
from recursive_intelligence.config import RuntimeConfig


def _commit_fix(worktree: Path) -> None:
    (worktree / "app.py").write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "add", "app.py"], cwd=str(worktree), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "fix add"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )


class BaselineFixAdapter(AgentAdapter):
    @property
    def name(self) -> str:
        return "baseline-fix"

    async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None, is_root=False):
        _commit_fix(Path(worktree))
        return NodeResult(
            session_id="baseline-session",
            raw={"status": "implemented", "summary": "fixed add"},
            result_text="",
            cost=CostRecord(total_usd=0.11),
            stop_reason="end_turn",
        )


class RecursiveFixAdapter(AgentAdapter):
    def __init__(self) -> None:
        self._responses = [
            {
                "action": "spawn_children",
                "rationale": "split out implementation",
                "children": [
                    {
                        "idempotency_key": "fix-add",
                        "objective": "fix add implementation",
                        "success_criteria": ["tests pass"],
                    }
                ],
            },
            {"action": "solve_directly", "rationale": "simple fix"},
            {"status": "implemented", "summary": "fixed add", "changed_files": ["app.py"]},
            {"verdict": "accept", "reason": "looks good"},
        ]
        self._count = 0

    @property
    def name(self) -> str:
        return "recursive-fix"

    async def run(self, prompt, worktree, mode, system_prompt=None, resume_session_id=None, on_message=None, is_root=False):
        self._count += 1
        payload = self._responses.pop(0)
        if mode == "execute":
            _commit_fix(Path(worktree))
        return NodeResult(
            session_id=f"recursive-session-{self._count}",
            raw=payload,
            result_text="",
            cost=CostRecord(total_usd=0.05),
            stop_reason="end_turn",
        )


@pytest.fixture
def source_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True, capture_output=True)
    (repo / "app.py").write_text("def add(a, b):\n    return a + b + 1\n")
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, base_commit


@pytest.mark.asyncio
async def test_benchmark_runner_persists_reports_and_exports(tmp_path: Path, source_repo: tuple[Path, str]):
    repo, base_commit = source_repo
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = RuntimeConfig(repo_root=workspace)

    task = SWEBenchTask(
        instance_id="local-task-1",
        repo=str(repo),
        base_commit=base_commit,
        problem_statement="Fix the add function so the unit test passes.",
        patch="",
        test_patch="",
        version="local",
        fail_to_pass=["tests/test_app.py::test_add"],
        pass_to_pass=[],
        gold_patch_files=["app.py"],
        test_files=["tests/test_app.py"],
        test_directives=["tests/test_app.py"],
        complexity_score=8,
        test_command="python3 -m pytest tests/test_app.py",
    )

    def adapter_factory(mode: str, _: SWEBenchTask) -> AgentAdapter:
        if mode == "baseline":
            return BaselineFixAdapter()
        return RecursiveFixAdapter()

    runner = BenchmarkRunner(config, adapter_factory=adapter_factory)
    report = await runner.run_swebench_suite([task], suite="tier-a", dataset="local", split="test")

    assert report.task_count == 1
    assert report.baseline.solved == 1
    assert report.recursive.solved == 1
    assert report.comparison.tie_solved == 1

    run_dir = config.benchmarks_dir / report.run_id
    task_dir = run_dir / "artifacts" / task.instance_id
    assert (run_dir / "report.json").exists()
    assert (task_dir / "baseline.patch").exists()
    assert (task_dir / "recursive.patch").exists()
    assert (task_dir / "baseline-ri").exists()
    assert (task_dir / "recursive-ri").exists()
    assert (task_dir / "baseline-score.log").exists()
    assert (task_dir / "recursive-score.log").exists()

    exported = export_report(run_dir / "report.json", tmp_path / "exports")
    assert len(exported) == 2
    assert all(path.exists() for path in exported)
