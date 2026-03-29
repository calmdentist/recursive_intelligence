"""Tests for frozen benchmark manifests."""

from __future__ import annotations

import json
from pathlib import Path

from recursive_intelligence.benchmarks.manifest import (
    build_recursive_hard_manifest,
    load_benchmark_manifest,
    resolve_manifest_tasks,
)
from recursive_intelligence.benchmarks.models import BenchmarkTask


class _FakeLoader:
    def __init__(self, tasks: list[BenchmarkTask]) -> None:
        self._tasks = tasks

    def load_tasks(self, refresh: bool = False) -> list[BenchmarkTask]:
        return self._tasks


def test_load_and_resolve_manifest_tasks(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_id": "recursive-hard-v1",
                "title": "Recursive Hard v1",
                "tasks": [
                    {
                        "benchmark": "swebench_pro",
                        "dataset_name": "ScaleAI/SWE-bench_Pro",
                        "split": "test",
                        "instance_id": "pro-1",
                        "selection_scores": {"total": 37},
                    },
                    {
                        "benchmark": "featurebench",
                        "dataset_name": "LiberCoders/FeatureBench",
                        "split": "full",
                        "instance_id": "feat-1",
                        "selection_scores": {"total": 35},
                    },
                ],
            }
        )
    )

    tasks_by_source = {
        ("swebench_pro", "ScaleAI/SWE-bench_Pro", "test"): [
            BenchmarkTask(
                benchmark="swebench_pro",
                dataset_name="ScaleAI/SWE-bench_Pro",
                dataset_split="test",
                instance_id="pro-1",
                repo="repo/pro",
                base_commit="abc123",
                problem_statement="Fix the multi-module issue.",
                patch="",
                test_patch="",
                version="python",
                fail_to_pass=["test"],
                pass_to_pass=[],
            )
        ],
        ("featurebench", "LiberCoders/FeatureBench", "full"): [
            BenchmarkTask(
                benchmark="featurebench",
                dataset_name="LiberCoders/FeatureBench",
                dataset_split="full",
                instance_id="feat-1",
                repo="repo/feat",
                base_commit="def456",
                problem_statement="Implement the feature.",
                patch="",
                test_patch="",
                version="python",
                fail_to_pass=["test"],
                pass_to_pass=[],
            )
        ],
    }

    def _fake_build_loader(benchmark, cache_dir, dataset=None, split=None):
        return _FakeLoader(tasks_by_source[(benchmark, dataset, split)])

    monkeypatch.setattr("recursive_intelligence.benchmarks.manifest.build_loader", _fake_build_loader)

    manifest = load_benchmark_manifest(manifest_path)
    resolved = resolve_manifest_tasks(manifest, tmp_path)

    assert [task.instance_id for task in resolved] == ["pro-1", "feat-1"]
    assert all(task.manifest_id == "recursive-hard-v1" for task in resolved)
    assert resolved[0].selection_scores["total"] == 37
    assert resolved[1].selection_scores["total"] == 35


def test_build_recursive_hard_manifest_combines_sources(tmp_path, monkeypatch):
    pro_task = BenchmarkTask(
        benchmark="swebench_pro",
        dataset_name="ScaleAI/SWE-bench_Pro",
        dataset_split="test",
        instance_id="pro-1",
        repo="repo/pro",
        base_commit="abc123",
        problem_statement=(
            "Investigate the bug across multiple modules and implement a coordinated runtime-safe fix "
            "without regressing the surrounding integration tests. The issue reproduces only after exercising "
            "several related flows, and the final patch needs to reconcile behavior across the API layer, "
            "persistence layer, and admin-facing output without relying on any single-file shortcut."
        ),
        patch="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n"
        "diff --git a/c.py b/c.py\n--- a/c.py\n+++ b/c.py\n",
        test_patch="diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n",
        version="python",
        fail_to_pass=["t1", "t2", "t3", "t4", "t5"],
        pass_to_pass=[],
        gold_patch_files=["a.py", "b.py", "c.py", "d.py", "e.py"],
        test_files=["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"],
        extra={"issue_categories": ["api", "db", "ui", "runtime"]},
    )
    feat_task = BenchmarkTask(
        benchmark="featurebench",
        dataset_name="LiberCoders/FeatureBench",
        dataset_split="full",
        instance_id="feat-1",
        repo="repo/feat",
        base_commit="def456",
        problem_statement=(
            "Implement the missing feature by coordinating changes across the package, tests, and integration path "
            "while preserving current behavior. The request is intentionally underspecified, requires tracing "
            "runtime behavior across multiple modules, and the implementation should satisfy both focused and "
            "broader regression coverage."
        ),
        patch="diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n"
        "diff --git a/z.py b/z.py\n--- a/z.py\n+++ b/z.py\n",
        test_patch="diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n",
        version="python",
        fail_to_pass=["t1", "t2", "t3", "t4"],
        pass_to_pass=[],
        gold_patch_files=["x.py", "y.py", "z.py", "u.py", "v.py"],
        test_files=["tests/test_x.py", "tests/test_y.py", "tests/test_z.py"],
        extra={"issue_categories": ["api", "runtime", "ux", "integration"], "timeout_run": 1200},
    )

    def _fake_build_loader(benchmark, cache_dir, dataset=None, split=None):
        if benchmark == "swebench_pro":
            return _FakeLoader([pro_task])
        if benchmark == "featurebench":
            return _FakeLoader([feat_task])
        raise AssertionError(f"unexpected benchmark {benchmark}")

    monkeypatch.setattr("recursive_intelligence.benchmarks.manifest.build_loader", _fake_build_loader)

    manifest = build_recursive_hard_manifest(
        tmp_path,
        output_path=tmp_path / "recursive-hard-v1.json",
        swebench_pro_count=1,
        featurebench_count=1,
        repo_cap=1,
    )

    assert manifest.manifest_id == "recursive-hard-v1"
    assert len(manifest.tasks) == 2
    assert {task.benchmark for task in manifest.tasks} == {"swebench_pro", "featurebench"}
    assert all(task.selection_scores["total"] >= 28 for task in manifest.tasks)
    assert (tmp_path / "recursive-hard-v1.json").exists()
