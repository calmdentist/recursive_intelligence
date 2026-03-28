"""Benchmark execution harness for flat vs recursive runs."""

from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Callable

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord
from recursive_intelligence.benchmarks.models import (
    BenchmarkModeResult,
    PatchScore,
    SWEBenchTask,
    TaskBenchmarkResult,
)
from recursive_intelligence.benchmarks.reporting import build_suite_report
from recursive_intelligence.benchmarks.swebench import (
    DEFAULT_DATASET,
    DEFAULT_SPLIT,
    resolve_test_command,
    resolve_python_requirement,
)
from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.baseline import BaselineRunner
from recursive_intelligence.runtime.orchestrator import Orchestrator, get_node_tree
from recursive_intelligence.runtime.state_store import StateStore


AdapterFactory = Callable[[str, SWEBenchTask], AgentAdapter]


class BenchmarkRunner:
    """Run SWE-bench tasks through baseline and recursive modes."""

    def __init__(
        self,
        config: RuntimeConfig,
        model: str = "claude-opus-4-6",
        adapter_factory: AdapterFactory | None = None,
        keep_task_dirs: bool = False,
        cleanup_task_dirs: bool = True,
        test_timeout_seconds: int = 1800,
    ) -> None:
        self.config = config
        self.model = model
        self.adapter_factory = adapter_factory
        self.keep_task_dirs = keep_task_dirs
        self.cleanup_task_dirs = cleanup_task_dirs
        self.test_timeout_seconds = test_timeout_seconds

    async def run_swebench_suite(
        self,
        tasks: list[SWEBenchTask],
        suite: str,
        dataset: str = DEFAULT_DATASET,
        split: str = DEFAULT_SPLIT,
    ):
        self.config.ensure_dirs()
        run_id = _new_benchmark_id()
        run_dir = self.config.benchmarks_dir / run_id
        tasks_dir = run_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)

        results: list[TaskBenchmarkResult] = []
        for task in tasks:
            result = await self.run_task(run_id, task)
            results.append(result)
            task_path = tasks_dir / f"{task.instance_id}.json"
            task_path.write_text(json.dumps(result.to_dict(), indent=2))
            report = build_suite_report(run_id, "swebench", suite, dataset, split, results)
            (run_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2))

        return build_suite_report(run_id, "swebench", suite, dataset, split, results)

    async def run_task(self, benchmark_run_id: str, task: SWEBenchTask) -> TaskBenchmarkResult:
        task_dir = self.config.benchmarks_dir / benchmark_run_id / "artifacts" / task.instance_id
        task_dir.mkdir(parents=True, exist_ok=True)

        baseline = await self._run_mode(task, task_dir, mode="baseline")
        recursive = await self._run_mode(task, task_dir, mode="recursive")
        comparison = compare_modes(baseline, recursive)

        result = TaskBenchmarkResult(
            instance_id=task.instance_id,
            repo=task.repo,
            version=task.version,
            complexity_score=task.complexity_score,
            baseline=baseline,
            recursive=recursive,
            comparison=comparison,
        )
        (task_dir / "task.json").write_text(json.dumps(result.to_dict(), indent=2))
        return result

    async def _run_mode(self, task: SWEBenchTask, task_dir: Path, mode: str) -> BenchmarkModeResult:
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"rari-{task.instance_id}-{mode}-"))
        repo_dir = scratch_dir / "repo"
        patch_path = task_dir / f"{mode}.patch"
        ri_artifacts_dir = task_dir / f"{mode}-ri"

        try:
            _clone_repo(task.repo, task.base_commit, repo_dir)
            runtime_config = RuntimeConfig(repo_root=repo_dir)
            adapter = self._make_adapter(mode, task)
            start = time.perf_counter()

            if mode == "baseline":
                runner = BaselineRunner(runtime_config, adapter)
                report = await runner.run(task.build_prompt())
                duration_ms = int((time.perf_counter() - start) * 1000)
                baseline_worktree = runtime_config.worktrees_dir / f"{report.run_id}-baseline"
                patch_text = _git_diff(repo_dir, task.base_commit, worktree=str(baseline_worktree))
                patch_path.write_text(patch_text)
                _copy_ri_artifacts(repo_dir, ri_artifacts_dir)
                score = self._score_patch(task, patch_text, task_dir, mode)
                return BenchmarkModeResult(
                    mode=mode,
                    run_id=report.run_id,
                    runtime_status=report.status,
                    solved=score.tests_passed,
                    changed_files=report.changed_files,
                    cost=report.cost,
                    duration_ms=duration_ms,
                    session_ids=[report.session_id],
                    session_count=1,
                    node_count=1,
                    tree_depth=0,
                    tree_breadth=0,
                    patch_path=str(patch_path),
                    patch_bytes=len(patch_text.encode("utf-8")),
                    ri_artifacts_path=str(ri_artifacts_dir) if ri_artifacts_dir.exists() else None,
                    score=score,
                )

            orchestrator = Orchestrator(runtime_config, adapter)
            run_id = await orchestrator.start_run(task.build_prompt())
            duration_ms = int((time.perf_counter() - start) * 1000)
            summary = _summarize_recursive_run(runtime_config, run_id, task.base_commit)
            patch_text = _git_diff(repo_dir, task.base_commit, worktree=summary["root_worktree"])
            patch_path.write_text(patch_text)
            _copy_ri_artifacts(repo_dir, ri_artifacts_dir)
            score = self._score_patch(task, patch_text, task_dir, mode)
            return BenchmarkModeResult(
                mode=mode,
                run_id=run_id,
                runtime_status=summary["status"],
                solved=score.tests_passed,
                changed_files=summary["changed_files"],
                cost=summary["cost"],
                duration_ms=duration_ms,
                session_ids=summary["session_ids"],
                session_count=len(summary["session_ids"]),
                node_count=summary["node_count"],
                tree_depth=summary["tree_depth"],
                tree_breadth=summary["tree_breadth"],
                patch_path=str(patch_path),
                patch_bytes=len(patch_text.encode("utf-8")),
                ri_artifacts_path=str(ri_artifacts_dir) if ri_artifacts_dir.exists() else None,
                score=score,
            )
        except Exception as exc:
            empty_score = PatchScore(
                status="runtime_error",
                patch_applied=False,
                tests_passed=False,
                exit_code=None,
                test_command=task.test_command or "",
                error=str(exc),
            )
            return BenchmarkModeResult(
                mode=mode,
                run_id=None,
                runtime_status="failed",
                solved=False,
                changed_files=[],
                cost=CostRecord(),
                duration_ms=0,
                session_ids=[],
                session_count=0,
                node_count=0,
                tree_depth=0,
                tree_breadth=0,
                patch_path=str(patch_path) if patch_path.exists() else None,
                patch_bytes=0,
                ri_artifacts_path=str(ri_artifacts_dir) if ri_artifacts_dir.exists() else None,
                score=empty_score,
                error=str(exc),
            )
        finally:
            if self.cleanup_task_dirs and not self.keep_task_dirs:
                shutil.rmtree(scratch_dir, ignore_errors=True)

    def _score_patch(self, task: SWEBenchTask, patch_text: str, task_dir: Path, mode: str) -> PatchScore:
        if not patch_text.strip():
            return PatchScore(
                status="no_patch",
                patch_applied=False,
                tests_passed=False,
                exit_code=None,
                test_command=resolve_test_command(task),
                error="Solver produced an empty patch",
            )

        scratch_dir = Path(tempfile.mkdtemp(prefix=f"rari-score-{task.instance_id}-{mode}-"))
        repo_dir = scratch_dir / "repo"
        log_path = task_dir / f"{mode}-score.log"
        patch_file = scratch_dir / "candidate.patch"

        try:
            _clone_repo(task.repo, task.base_commit, repo_dir)
            patch_file.write_text(patch_text)
            apply_check = subprocess.run(
                ["git", "apply", "--check", str(patch_file)],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
            )
            if apply_check.returncode != 0:
                log_path.write_text(apply_check.stderr or apply_check.stdout)
                return PatchScore(
                    status="patch_failed",
                    patch_applied=False,
                    tests_passed=False,
                    exit_code=apply_check.returncode,
                    test_command=resolve_test_command(task),
                    log_path=str(log_path),
                    error=(apply_check.stderr or apply_check.stdout).strip(),
                )

            apply_patch = subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
            )
            if apply_patch.returncode != 0:
                log_path.write_text(apply_patch.stderr or apply_patch.stdout)
                return PatchScore(
                    status="patch_failed",
                    patch_applied=False,
                    tests_passed=False,
                    exit_code=apply_patch.returncode,
                    test_command=resolve_test_command(task),
                    log_path=str(log_path),
                    error=(apply_patch.stderr or apply_patch.stdout).strip(),
                )

            command = resolve_test_command(task)
            python_selection = _select_task_python(task)
            if python_selection["status"] == "unsupported":
                message = python_selection["error"]
                log_path.write_text(message)
                return PatchScore(
                    status="unsupported_environment",
                    patch_applied=True,
                    tests_passed=False,
                    exit_code=None,
                    test_command=command,
                    log_path=str(log_path),
                    error=message,
                    python_requirement=python_selection["requirement"],
                )

            completed = _run_test_command(
                command,
                repo_dir,
                self.test_timeout_seconds,
                python_executable=python_selection["python_executable"],
            )
            log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""))
            passed = completed.returncode == 0
            return PatchScore(
                status="passed" if passed else "failed",
                patch_applied=True,
                tests_passed=passed,
                exit_code=completed.returncode,
                test_command=command,
                log_path=str(log_path),
                python_executable=python_selection["python_executable"],
                python_requirement=python_selection["requirement"],
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + "\n" + (exc.stderr or "")
            log_path.write_text(output)
            return PatchScore(
                status="timeout",
                patch_applied=True,
                tests_passed=False,
                exit_code=None,
                test_command=resolve_test_command(task),
                log_path=str(log_path),
                error="Test command timed out",
                python_requirement=_describe_python_requirement(task),
            )
        finally:
            if self.cleanup_task_dirs and not self.keep_task_dirs:
                shutil.rmtree(scratch_dir, ignore_errors=True)

    def _make_adapter(self, mode: str, task: SWEBenchTask) -> AgentAdapter:
        if self.adapter_factory is not None:
            return self.adapter_factory(mode, task)
        from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

        return ClaudeAdapter(model=self.model)


def compare_modes(baseline: BenchmarkModeResult, recursive: BenchmarkModeResult) -> str:
    if _is_unsupported_score(baseline.score) or _is_unsupported_score(recursive.score):
        return "unsupported"
    if recursive.solved and not baseline.solved:
        return "recursive_win"
    if baseline.solved and not recursive.solved:
        return "baseline_win"
    if baseline.solved and recursive.solved:
        return "tie_solved"
    return "tie_failed"


def _clone_repo(repo: str, base_commit: str, destination: Path) -> None:
    clone_source = repo
    if not Path(repo).exists() and "://" not in repo:
        clone_source = f"https://github.com/{repo}.git"

    subprocess.run(
        ["git", "clone", "--quiet", clone_source, str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", base_commit],
        cwd=str(destination),
        check=True,
        capture_output=True,
        text=True,
    )


def _copy_ri_artifacts(repo_dir: Path, destination: Path) -> None:
    source = repo_dir / ".ri"
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _git_diff(repo_dir: Path, base_commit: str, worktree: str | None = None) -> str:
    cwd = worktree or str(repo_dir)
    result = subprocess.run(
        ["git", "diff", "--binary", base_commit, "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_changed_files(worktree: str, base_commit: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _run_test_command(
    command: str,
    repo_dir: Path,
    timeout_seconds: int,
    python_executable: str | None = None,
) -> subprocess.CompletedProcess[str]:
    tokens = shlex.split(command)
    env = os.environ.copy()

    idx = 0
    while idx < len(tokens) and _is_env_assignment(tokens[idx]):
        key, _, value = tokens[idx].partition("=")
        env[key] = value
        idx += 1

    argv = tokens[idx:]
    if not argv:
        raise ValueError(f"Malformed test command: {command}")

    argv = _normalize_python_runner(argv, repo_dir, python_executable)
    return subprocess.run(
        argv,
        cwd=str(repo_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _normalize_python_runner(
    argv: list[str],
    repo_dir: Path,
    python_executable: str | None = None,
) -> list[str]:
    executable = argv[0]
    executable_name = Path(executable).name
    script_path = (repo_dir / executable).resolve() if not Path(executable).is_absolute() else Path(executable)
    python_missing = shutil.which("python") is None
    fallback_python = python_executable or shutil.which("python3")

    if python_executable and executable_name.startswith("python"):
        return [python_executable, *argv[1:]]

    if python_executable and executable in {"pytest", "py.test"}:
        return [python_executable, "-m", "pytest", *argv[1:]]

    if (
        python_missing
        and fallback_python
        and script_path.exists()
        and script_path.is_file()
        and _uses_env_python(script_path)
    ):
        return [fallback_python, str(script_path), *argv[1:]]

    if (
        python_executable
        and script_path.exists()
        and script_path.is_file()
        and _uses_python_shebang(script_path)
    ):
        return [python_executable, str(script_path), *argv[1:]]

    return argv


def _uses_env_python(script_path: Path) -> bool:
    try:
        first_line = script_path.read_text(errors="ignore").splitlines()[0]
    except IndexError:
        return False
    return first_line.startswith("#!") and "env python" in first_line


def _uses_python_shebang(script_path: Path) -> bool:
    try:
        first_line = script_path.read_text(errors="ignore").splitlines()[0]
    except IndexError:
        return False
    return first_line.startswith("#!") and "python" in first_line


def _is_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    name, _, _ = token.partition("=")
    return bool(name) and (name[0].isalpha() or name[0] == "_") and all(
        ch.isalnum() or ch == "_" for ch in name
    )


def _is_unsupported_score(score: PatchScore) -> bool:
    return score.status == "unsupported_environment"


def _describe_python_requirement(task: SWEBenchTask) -> str | None:
    requirement = resolve_python_requirement(task)
    if requirement is None:
        return None
    return requirement.describe()


def _select_task_python(task: SWEBenchTask) -> dict[str, str | None]:
    requirement = resolve_python_requirement(task)
    if requirement is None:
        return {
            "status": "supported",
            "python_executable": None,
            "requirement": None,
            "error": None,
        }

    for executable, version in _available_python_interpreters():
        if requirement.matches(version):
            return {
                "status": "supported",
                "python_executable": executable,
                "requirement": requirement.describe(),
                "error": None,
            }

    available = ", ".join(
        f"{Path(executable).name} ({version[0]}.{version[1]}.{version[2]})"
        for executable, version in _available_python_interpreters()
    ) or "none"
    return {
        "status": "unsupported",
        "python_executable": None,
        "requirement": requirement.describe(),
        "error": (
            f"Unsupported benchmark environment: requires {requirement.describe()}, "
            f"but available interpreters are {available}."
        ),
    }


@lru_cache(maxsize=1)
def _available_python_interpreters() -> tuple[tuple[str, tuple[int, int, int]], ...]:
    names = [
        Path(sys.executable).name,
        "python",
        "python3",
        "python3.13",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3.9",
        "python3.8",
    ]
    interpreters: list[tuple[str, tuple[int, int, int]]] = []
    seen: set[str] = set()
    for name in names:
        path = shutil.which(name)
        if path is None or path in seen:
            continue
        version = _python_version(path)
        if version is None:
            continue
        interpreters.append((path, version))
        seen.add(path)
    return tuple(interpreters)


def _python_version(executable: str) -> tuple[int, int, int] | None:
    result = subprocess.run(
        [
            executable,
            "-c",
            "import sys; print('.'.join(str(part) for part in sys.version_info[:3]))",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    parts = raw.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def _summarize_recursive_run(config: RuntimeConfig, run_id: str, base_commit: str) -> dict:
    store = StateStore(config.db_path)
    try:
        run = store.get_run(run_id)
        if run is None or run.root_node_id is None:
            raise ValueError(f"Run {run_id} not found")
        root = store.get_node(run.root_node_id)
        if root is None or not root.worktree_path:
            raise ValueError(f"Root node missing for run {run_id}")

        tree = get_node_tree(store, run_id)
        events = store.get_run_events(run_id)
        session_ids: list[str] = []
        seen_sessions: set[str] = set()
        total_cost = CostRecord()
        for event in events:
            if event.event_type not in {"plan_result", "execution_result", "review_result"}:
                continue
            session_id = event.data.get("session_id")
            if session_id and session_id not in seen_sessions:
                seen_sessions.add(session_id)
                session_ids.append(session_id)
            cost = event.data.get("cost") or {}
            total_cost.input_tokens += int(cost.get("input_tokens", 0))
            total_cost.output_tokens += int(cost.get("output_tokens", 0))
            total_cost.total_usd += float(cost.get("total_usd", 0.0))

        nodes = store.get_run_nodes(run_id)
        return {
            "status": run.status,
            "node_count": len(nodes),
            "tree_depth": _tree_depth(tree[0]) if tree else 0,
            "tree_breadth": _tree_breadth(tree[0]) if tree else 0,
            "session_ids": session_ids,
            "cost": total_cost,
            "root_worktree": root.worktree_path,
            "changed_files": _git_changed_files(root.worktree_path, base_commit),
        }
    finally:
        store.close()


def _tree_depth(node: dict) -> int:
    children = node.get("children", [])
    if not children:
        return node.get("depth", 0)
    return max(_tree_depth(child) for child in children)


def _tree_breadth(node: dict) -> int:
    children = node.get("children", [])
    breadth = len(children)
    if not children:
        return breadth
    return max(breadth, max(_tree_breadth(child) for child in children))


def _new_benchmark_id() -> str:
    return f"bench-{uuid.uuid4().hex[:12]}"
