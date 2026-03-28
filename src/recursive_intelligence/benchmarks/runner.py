"""Benchmark execution harness for flat vs recursive runs."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord
from recursive_intelligence.benchmarks.evaluation import OfficialHarnessEvaluator, PatchEvaluator
from recursive_intelligence.benchmarks.models import (
    BenchmarkModeResult,
    PatchScore,
    SWEBenchTask,
    TaskBenchmarkResult,
)
from recursive_intelligence.benchmarks.reporting import build_suite_report
from recursive_intelligence.benchmarks.swebench import DEFAULT_DATASET, DEFAULT_SPLIT
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
        root_model: str | None = None,
        child_model: str | None = None,
        adapter_factory: AdapterFactory | None = None,
        patch_evaluator: PatchEvaluator | None = None,
        keep_task_dirs: bool = False,
        cleanup_task_dirs: bool = True,
        test_timeout_seconds: int = 1800,
        evaluation_namespace: str | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.root_model = root_model or model
        self.child_model = child_model or self.root_model
        self.adapter_factory = adapter_factory
        self.patch_evaluator = patch_evaluator
        self.keep_task_dirs = keep_task_dirs
        self.cleanup_task_dirs = cleanup_task_dirs
        self.test_timeout_seconds = test_timeout_seconds
        self.evaluation_namespace = evaluation_namespace

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
        evaluator = self.patch_evaluator or OfficialHarnessEvaluator(
            dataset_name=dataset,
            split=split,
            timeout_seconds=self.test_timeout_seconds,
            namespace=self.evaluation_namespace,
        )

        results: list[TaskBenchmarkResult] = []
        for task in tasks:
            result = await self.run_task(run_id, task, evaluator)
            results.append(result)
            task_path = tasks_dir / f"{task.instance_id}.json"
            task_path.write_text(json.dumps(result.to_dict(), indent=2))
            report = build_suite_report(run_id, "swebench", suite, dataset, split, results)
            (run_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2))

        return build_suite_report(run_id, "swebench", suite, dataset, split, results)

    async def run_task(
        self,
        benchmark_run_id: str,
        task: SWEBenchTask,
        evaluator: PatchEvaluator,
    ) -> TaskBenchmarkResult:
        task_dir = self.config.benchmarks_dir / benchmark_run_id / "artifacts" / task.instance_id
        task_dir.mkdir(parents=True, exist_ok=True)

        baseline = await self._run_mode(task, task_dir, mode="baseline", evaluator=evaluator)
        recursive = await self._run_mode(task, task_dir, mode="recursive", evaluator=evaluator)
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

    async def _run_mode(
        self,
        task: SWEBenchTask,
        task_dir: Path,
        mode: str,
        evaluator: PatchEvaluator,
    ) -> BenchmarkModeResult:
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
                score = self._score_patch(task, patch_text, task_dir, mode, evaluator)
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
            score = self._score_patch(task, patch_text, task_dir, mode, evaluator)
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

    def _score_patch(
        self,
        task: SWEBenchTask,
        patch_text: str,
        task_dir: Path,
        mode: str,
        evaluator: PatchEvaluator,
    ) -> PatchScore:
        return evaluator.score_patch(task, patch_text, task_dir, mode)

    def _make_adapter(self, mode: str, task: SWEBenchTask) -> AgentAdapter:
        if self.adapter_factory is not None:
            return self.adapter_factory(mode, task)
        from recursive_intelligence.adapters.claude.adapter import ClaudeAdapter

        if mode == "baseline":
            return ClaudeAdapter(root_model=self.root_model, child_model=self.root_model)
        return ClaudeAdapter(root_model=self.root_model, child_model=self.child_model)


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


def _is_unsupported_score(score: PatchScore) -> bool:
    return score.status == "unsupported_environment"


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
