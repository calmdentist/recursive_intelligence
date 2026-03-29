"""End-to-end tests for the benchmark runner."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult
from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter
from recursive_intelligence.benchmarks.evaluation import (
    FeatureBenchEvaluator,
    LocalPatchEvaluator,
    OfficialHarnessEvaluator,
    SWEBenchProEvaluator,
    _run_test_command,
    _select_task_python,
)
from recursive_intelligence.benchmarks.models import (
    BenchmarkModeResult,
    PatchScore,
    SWEBenchTask,
    TaskBenchmarkResult,
)
from recursive_intelligence.benchmarks.reporting import build_suite_report, export_report
from recursive_intelligence.benchmarks.runner import BenchmarkRunner, compare_modes
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
        benchmark="swebench",
        dataset_name="local",
        dataset_split="test",
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

    runner = BenchmarkRunner(
        config,
        model="claude-opus-4-6",
        root_model="claude-sonnet-4-6",
        child_model="claude-haiku-4-5",
        adapter_factory=adapter_factory,
        patch_evaluator=LocalPatchEvaluator(),
        max_concurrency=2,
    )
    report = await runner.run_swebench_suite(
        [task],
        suite="tier-a",
        dataset="local",
        split="test",
        requested_limit=1,
    )

    assert report.task_count == 1
    assert report.baseline.solved == 1
    assert report.recursive.solved == 1
    assert report.comparison.tie_solved == 1
    assert report.config.fallback_model == "claude-opus-4-6"
    assert report.config.root_model == "claude-sonnet-4-6"
    assert report.config.child_model == "claude-haiku-4-5"
    assert report.config.evaluation_backend == "local_patch"
    assert report.config.requested_limit == 1
    assert report.config.max_concurrency == 2
    assert report.tasks[0].benchmark == "swebench"
    assert report.tasks[0].dataset_name == "local"
    assert report.tasks[0].dataset_split == "test"

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
    assert "config_root_model" in exported[1].read_text()
    assert "dataset_name" in exported[1].read_text()


def test_run_test_command_uses_python3_for_env_python_scripts(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    bin_dir = repo / "bin"
    bin_dir.mkdir()
    script = bin_dir / "test"
    script.write_text(
        "#!/usr/bin/env python\n"
        "import os, sys\n"
        "print(sys.executable)\n"
        "print(os.environ['BENCH_FLAG'])\n"
    )
    script.chmod(0o755)

    completed = _run_test_command("BENCH_FLAG=ok bin/test", repo, timeout_seconds=10)

    assert completed.returncode == 0
    lines = completed.stdout.strip().splitlines()
    assert Path(lines[0]).name.startswith("python3")
    assert lines[1] == "ok"


def test_select_task_python_marks_unsupported_when_required_interpreter_missing(monkeypatch: pytest.MonkeyPatch):
    task = SWEBenchTask(
        instance_id="sympy__sympy-13091",
        repo="sympy/sympy",
        base_commit="abc123",
        problem_statement="fix it",
        patch="",
        test_patch="",
        version="1.1",
        fail_to_pass=[],
        pass_to_pass=[],
    )

    monkeypatch.setattr(
        "recursive_intelligence.benchmarks.evaluation._available_python_interpreters",
        lambda: (("/usr/bin/python3.13", (3, 13, 1)),),
    )

    selection = _select_task_python(task)

    assert selection["status"] == "unsupported"
    assert selection["python_executable"] is None
    assert "Python <=3.9" in selection["requirement"]
    assert "python3.13" in selection["error"]


def test_official_harness_evaluator_reads_instance_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    task = SWEBenchTask(
        instance_id="sympy__sympy-13091",
        repo="sympy/sympy",
        base_commit="abc123",
        problem_statement="fix it",
        patch="",
        test_patch="",
        version="1.1",
        fail_to_pass=[],
        pass_to_pass=[],
    )
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "swebench" else None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else f"/usr/bin/{name}")

    def _fake_run(command, cwd=None, capture_output=False, text=False, **kwargs):
        eval_dir = Path(command[command.index("--report_dir") + 1])
        run_id = command[command.index("--run_id") + 1]
        instance_dir = eval_dir / "logs" / "run_evaluation" / run_id / "recursive-intelligence__baseline" / task.instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "report.json").write_text(f'{{"{task.instance_id}": {{"resolved": true}}}}')
        (instance_dir / "test_output.txt").write_text("tests passed")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    evaluator = OfficialHarnessEvaluator(
        dataset_name="SWE-bench/SWE-bench_Verified",
        split="test",
        python_executable="python3",
    )
    score = evaluator.score_patch(task, "--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+new\n", task_dir, "baseline")

    assert score.status == "passed"
    assert score.patch_applied is True
    assert score.tests_passed is True
    assert score.report_path is not None
    assert score.log_path is not None

def test_swebench_pro_evaluator_reads_eval_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    task = SWEBenchTask(
        benchmark="swebench_pro",
        dataset_name="ScaleAI/SWE-bench_Pro",
        dataset_split="test",
        instance_id="instance_nodebb-1",
        repo="NodeBB/NodeBB",
        base_commit="abc123",
        problem_statement="fix it",
        patch="",
        test_patch="",
        version="javascript",
        fail_to_pass=["test/a"],
        pass_to_pass=["test/b"],
        extra={
            "before_repo_set_cmd": "git checkout test/a.js",
            "selected_test_files_to_run": ["test/a.js"],
        },
    )
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    repo_path = tmp_path / "swebench-pro-os"
    (repo_path / "run_scripts").mkdir(parents=True)
    (repo_path / "swe_bench_pro_eval.py").write_text("print('stub')\n")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else f"/usr/bin/{name}")

    def _fake_run(command, cwd=None, capture_output=False, text=False, **kwargs):
        output_dir = Path(command[command.index("--output_dir") + 1])
        instance_dir = output_dir / task.instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval_results.json").write_text(f'{{"{task.instance_id}": true}}')
        (instance_dir / "baseline_output.json").write_text('{"tests": []}')
        (instance_dir / "baseline_stdout.log").write_text("ok")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    evaluator = SWEBenchProEvaluator(
        dataset_name="ScaleAI/SWE-bench_Pro",
        split="test",
        python_executable="python3",
        repo_path=repo_path,
    )
    score = evaluator.score_patch(task, "--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+new\n", task_dir, "baseline")

    assert score.status == "passed"
    assert score.patch_applied is True
    assert score.tests_passed is True
    assert score.report_path is not None
    assert score.log_path is not None
    assert "--dockerhub_username jefzda" in score.test_command


def test_featurebench_evaluator_reads_instance_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    task = SWEBenchTask(
        benchmark="featurebench",
        dataset_name="LiberCoders/FeatureBench",
        dataset_split="full",
        instance_id="trl-1",
        repo="huggingface/trl",
        base_commit="abc123",
        problem_statement="fix it",
        patch="",
        test_patch="",
        version="python",
        fail_to_pass=[],
        pass_to_pass=[],
    )
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "featurebench" else None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else f"/usr/bin/{name}")

    def _fake_run(command, cwd=None, capture_output=False, text=False, **kwargs):
        eval_dir = Path(cwd)
        instance_dir = eval_dir / "eval_outputs" / task.instance_id / "attempt-1"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "report.json").write_text('{"attempt_1": {"resolved_instances": 1}}')
        (instance_dir / "report.json").write_text(
            f'{{"{task.instance_id}": {{"resolved": true, "patch_successfully_applied": true}}}}'
        )
        (instance_dir / "test_output.txt").write_text("tests passed")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    evaluator = FeatureBenchEvaluator(
        dataset_name="LiberCoders/FeatureBench",
        split="full",
        python_executable="python3",
    )
    score = evaluator.score_patch(task, "--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+new\n", task_dir, "recursive")

    assert score.status == "passed"
    assert score.patch_applied is True
    assert score.tests_passed is True
    assert score.report_path is not None
    assert score.log_path is not None
    assert "featurebench.harness.run_evaluation" in score.test_command

def test_benchmark_runner_configures_distinct_root_and_child_models(tmp_path: Path):
    config = RuntimeConfig(repo_root=tmp_path)
    runner = BenchmarkRunner(
        config,
        model="claude-sonnet-4-6",
        root_model="claude-opus-4-6",
        child_model="claude-haiku-4-5",
    )

    baseline_adapter = runner._make_adapter("baseline", _dummy_task())
    recursive_adapter = runner._make_adapter("recursive", _dummy_task())

    assert isinstance(baseline_adapter, ClaudeAdapter)
    assert isinstance(recursive_adapter, ClaudeAdapter)
    assert baseline_adapter._model_for_node(is_root=True) == "claude-opus-4-6"
    assert baseline_adapter._model_for_node(is_root=False) == "claude-opus-4-6"
    assert recursive_adapter._model_for_node(is_root=True) == "claude-opus-4-6"
    assert recursive_adapter._model_for_node(is_root=False) == "claude-haiku-4-5"


def test_benchmark_runner_routes_evaluators_by_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = RuntimeConfig(repo_root=tmp_path)
    runner = BenchmarkRunner(config)

    monkeypatch.setattr("recursive_intelligence.benchmarks.evaluation.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("recursive_intelligence.benchmarks.evaluation.shutil.which", lambda name: "/usr/bin/docker")

    pro_repo = tmp_path / ".ri" / "tools" / "SWE-bench_Pro-os"
    (pro_repo / "run_scripts").mkdir(parents=True)
    (pro_repo / "swe_bench_pro_eval.py").write_text("print('stub')\n")

    swebench_task = _task_with_id("task-swebench")
    swebench_task.benchmark = "swebench"
    swebench_task.dataset_name = "SWE-bench/SWE-bench_Verified"
    swebench_task.dataset_split = "test"
    swebench_task.evaluation_backend = "swebench_harness"

    pro_task = _task_with_id("task-pro")
    pro_task.benchmark = "swebench_pro"
    pro_task.dataset_name = "ScaleAI/SWE-bench_Pro"
    pro_task.dataset_split = "test"
    pro_task.evaluation_backend = "swebench_pro"
    pro_task.extra["before_repo_set_cmd"] = ""
    pro_task.extra["selected_test_files_to_run"] = []

    feature_task = _task_with_id("task-feature")
    feature_task.benchmark = "featurebench"
    feature_task.dataset_name = "LiberCoders/FeatureBench"
    feature_task.dataset_split = "full"
    feature_task.evaluation_backend = "featurebench"

    assert isinstance(runner._evaluator_for_task(swebench_task), OfficialHarnessEvaluator)
    assert isinstance(runner._evaluator_for_task(pro_task), SWEBenchProEvaluator)
    assert isinstance(runner._evaluator_for_task(feature_task), FeatureBenchEvaluator)


@pytest.mark.asyncio
async def test_benchmark_runner_executes_tasks_in_parallel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = RuntimeConfig(repo_root=tmp_path)
    runner = BenchmarkRunner(
        config,
        patch_evaluator=LocalPatchEvaluator(),
        max_concurrency=2,
    )
    tasks = [
        _task_with_id("task-a"),
        _task_with_id("task-b"),
    ]
    started = 0
    both_started = asyncio.Event()

    async def _fake_run_task(benchmark_run_id, task):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        return _task_result(task.instance_id)

    monkeypatch.setattr(runner, "run_task", _fake_run_task)

    report = await runner.run_swebench_suite(tasks, suite="tier-a", dataset="local", split="test")

    assert report.task_count == 2
    assert [task.instance_id for task in report.tasks] == ["task-a", "task-b"]
    assert report.config.max_concurrency == 2


@pytest.mark.asyncio
async def test_benchmark_runner_persists_manifest_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = RuntimeConfig(repo_root=tmp_path)
    runner = BenchmarkRunner(
        config,
        patch_evaluator=LocalPatchEvaluator(),
        max_concurrency=1,
    )
    swebench_task = _task_with_id("task-a")
    swebench_task.benchmark = "swebench_pro"
    swebench_task.dataset_name = "ScaleAI/SWE-bench_Pro"
    swebench_task.dataset_split = "test"
    feature_task = _task_with_id("task-b")
    feature_task.benchmark = "featurebench"
    feature_task.dataset_name = "LiberCoders/FeatureBench"
    feature_task.dataset_split = "full"

    async def _fake_run_task(benchmark_run_id, task):
        return _task_result(task.instance_id, benchmark=task.benchmark, dataset_name=task.dataset_name, dataset_split=task.dataset_split)

    monkeypatch.setattr(runner, "run_task", _fake_run_task)

    report = await runner.run_manifest_suite(
        [swebench_task, feature_task],
        suite="recursive-hard-v1",
        manifest_id="recursive-hard-v1",
        manifest_path="/tmp/recursive-hard-v1.json",
    )

    assert report.benchmark == "manifest"
    assert report.config.manifest_id == "recursive-hard-v1"
    assert report.config.manifest_path == "/tmp/recursive-hard-v1.json"
    assert report.config.dataset_sources == [
        "featurebench:LiberCoders/FeatureBench:full",
        "swebench_pro:ScaleAI/SWE-bench_Pro:test",
    ]

def test_report_excludes_unsupported_tasks_from_solve_rate():
    unsupported_score = PatchScore(
        status="unsupported_environment",
        patch_applied=True,
        tests_passed=False,
        exit_code=None,
        test_command="bin/test",
        error="requires Python <=3.9",
        python_requirement="Python <=3.9",
    )
    unsupported_mode = BenchmarkModeResult(
        mode="baseline",
        run_id="run-1",
        runtime_status="completed",
        solved=False,
        changed_files=[],
        cost=CostRecord(),
        duration_ms=100,
        session_ids=["session-1"],
        session_count=1,
        node_count=1,
        tree_depth=0,
        tree_breadth=0,
        patch_path=None,
        patch_bytes=0,
        ri_artifacts_path=None,
        score=unsupported_score,
    )
    solved_mode = BenchmarkModeResult(
        mode="recursive",
        run_id="run-2",
        runtime_status="completed",
        solved=True,
        changed_files=["app.py"],
        cost=CostRecord(total_usd=1.0),
        duration_ms=200,
        session_ids=["session-2"],
        session_count=1,
        node_count=1,
        tree_depth=0,
        tree_breadth=0,
        patch_path=None,
        patch_bytes=10,
        ri_artifacts_path=None,
        score=PatchScore(
            status="passed",
            patch_applied=True,
            tests_passed=True,
            exit_code=0,
            test_command="pytest",
        ),
    )

    task = TaskBenchmarkResult(
        benchmark="swebench",
        dataset_name="local",
        dataset_split="test",
        instance_id="task-1",
        repo="sympy/sympy",
        version="1.1",
        complexity_score=10,
        baseline=unsupported_mode,
        recursive=unsupported_mode,
        comparison=compare_modes(unsupported_mode, unsupported_mode),
    )
    report = build_suite_report(
        run_id="bench-1",
        benchmark="swebench",
        suite="tier-a",
        dataset="local",
        split="test",
        tasks=[task],
    )

    assert report.baseline.total == 1
    assert report.baseline.eligible == 0
    assert report.baseline.unsupported == 1
    assert report.baseline.solve_rate == 0.0
    assert report.comparison.unsupported == 1

    supported_task = TaskBenchmarkResult(
        benchmark="swebench",
        dataset_name="local",
        dataset_split="test",
        instance_id="task-2",
        repo="local/repo",
        version="local",
        complexity_score=5,
        baseline=solved_mode,
        recursive=solved_mode,
        comparison=compare_modes(solved_mode, solved_mode),
    )
    mixed_report = build_suite_report(
        run_id="bench-2",
        benchmark="swebench",
        suite="tier-a",
        dataset="local",
        split="test",
        tasks=[task, supported_task],
    )

    assert mixed_report.baseline.total == 2
    assert mixed_report.baseline.eligible == 1
    assert mixed_report.baseline.unsupported == 1
    assert mixed_report.baseline.solve_rate == 1.0


def _dummy_task() -> SWEBenchTask:
    return SWEBenchTask(
        benchmark="swebench",
        dataset_name="local",
        dataset_split="test",
        instance_id="task-1",
        repo="owner/repo",
        base_commit="abc123",
        problem_statement="fix it",
        patch="",
        test_patch="",
        version="test",
        fail_to_pass=[],
        pass_to_pass=[],
    )


def _task_with_id(instance_id: str) -> SWEBenchTask:
    task = _dummy_task()
    task.instance_id = instance_id
    return task


def _task_result(
    instance_id: str,
    benchmark: str = "swebench",
    dataset_name: str = "local",
    dataset_split: str = "test",
) -> TaskBenchmarkResult:
    solved_score = PatchScore(
        status="passed",
        patch_applied=True,
        tests_passed=True,
        exit_code=0,
        test_command="pytest",
    )
    mode = BenchmarkModeResult(
        mode="baseline",
        run_id=f"run-{instance_id}",
        runtime_status="completed",
        solved=True,
        changed_files=["app.py"],
        cost=CostRecord(total_usd=0.1),
        duration_ms=100,
        session_ids=[f"session-{instance_id}"],
        session_count=1,
        node_count=1,
        tree_depth=0,
        tree_breadth=0,
        patch_path=None,
        patch_bytes=10,
        ri_artifacts_path=None,
        score=solved_score,
    )
    recursive_mode = BenchmarkModeResult(
        mode="recursive",
        run_id=f"run-{instance_id}-recursive",
        runtime_status="completed",
        solved=True,
        changed_files=["app.py"],
        cost=CostRecord(total_usd=0.1),
        duration_ms=100,
        session_ids=[f"session-{instance_id}-recursive"],
        session_count=1,
        node_count=1,
        tree_depth=0,
        tree_breadth=0,
        patch_path=None,
        patch_bytes=10,
        ri_artifacts_path=None,
        score=solved_score,
    )
    return TaskBenchmarkResult(
        benchmark=benchmark,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        instance_id=instance_id,
        repo="owner/repo",
        version="test",
        complexity_score=1,
        baseline=mode,
        recursive=recursive_mode,
        comparison="tie_solved",
    )
