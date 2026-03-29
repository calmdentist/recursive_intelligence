"""Benchmark aggregation and report export."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from recursive_intelligence.benchmarks.models import (
    BenchmarkRunConfig,
    BenchmarkSuiteReport,
    ComparisonAggregate,
    ModeAggregate,
    TaskBenchmarkResult,
)


def build_suite_report(
    run_id: str,
    benchmark: str,
    suite: str,
    dataset: str,
    split: str,
    tasks: list[TaskBenchmarkResult],
    config: BenchmarkRunConfig | None = None,
) -> BenchmarkSuiteReport:
    return BenchmarkSuiteReport(
        run_id=run_id,
        benchmark=benchmark,
        suite=suite,
        dataset=dataset,
        split=split,
        created_at=datetime.now(timezone.utc).isoformat(),
        task_count=len(tasks),
        baseline=_build_mode_aggregate(tasks, "baseline"),
        recursive=_build_mode_aggregate(tasks, "recursive"),
        comparison=_build_comparison_aggregate(tasks),
        config=config or BenchmarkRunConfig(),
        tasks=tasks,
    )


def export_report(report_path: Path, output_dir: Path | None = None) -> list[Path]:
    report = json.loads(report_path.read_text())
    destination = output_dir or report_path.parent
    destination.mkdir(parents=True, exist_ok=True)

    json_path = destination / f"{report['run_id']}.json"
    csv_path = destination / f"{report['run_id']}.csv"
    json_path.write_text(json.dumps(report, indent=2))
    _write_csv(report, csv_path)
    return [json_path, csv_path]


def _build_mode_aggregate(tasks: list[TaskBenchmarkResult], mode: str) -> ModeAggregate:
    mode_results = [getattr(task, mode) for task in tasks]
    total = len(mode_results)
    unsupported = sum(1 for result in mode_results if result.score.status == "unsupported_environment")
    eligible = total - unsupported
    solved = sum(1 for result in mode_results if result.solved)
    latencies = [result.duration_ms for result in mode_results]
    costs = [result.cost.total_usd for result in mode_results]
    depths = [result.tree_depth for result in mode_results]
    breadths = [result.tree_breadth for result in mode_results]
    nodes = [result.node_count for result in mode_results]
    return ModeAggregate(
        solve_rate=(solved / eligible) if eligible else 0.0,
        solved=solved,
        total=total,
        eligible=eligible,
        unsupported=unsupported,
        avg_cost_usd=_mean(costs),
        avg_latency_ms=_mean(latencies),
        median_latency_ms=_median(latencies),
        avg_tree_depth=_mean(depths),
        avg_tree_breadth=_mean(breadths),
        avg_node_count=_mean(nodes),
    )


def _build_comparison_aggregate(tasks: list[TaskBenchmarkResult]) -> ComparisonAggregate:
    baseline_win = sum(1 for task in tasks if task.comparison == "baseline_win")
    recursive_win = sum(1 for task in tasks if task.comparison == "recursive_win")
    tie_solved = sum(1 for task in tasks if task.comparison == "tie_solved")
    tie_failed = sum(1 for task in tasks if task.comparison == "tie_failed")
    unsupported = sum(1 for task in tasks if task.comparison == "unsupported")
    return ComparisonAggregate(
        baseline_win=baseline_win,
        recursive_win=recursive_win,
        tie_solved=tie_solved,
        tie_failed=tie_failed,
        unsupported=unsupported,
    )


def _write_csv(report: dict, path: Path) -> None:
    fieldnames = [
        "benchmark",
        "dataset_name",
        "dataset_split",
        "instance_id",
        "repo",
        "version",
        "complexity_score",
        "comparison",
        "config_fallback_model",
        "config_root_model",
        "config_child_model",
        "config_evaluation_backend",
        "config_evaluation_namespace",
        "config_requested_limit",
        "config_max_concurrency",
        "config_test_timeout_seconds",
        "config_manifest_id",
        "config_manifest_path",
        "config_dataset_sources",
        "baseline_solved",
        "baseline_runtime_status",
        "baseline_score_status",
        "baseline_score_error",
        "baseline_score_log_path",
        "baseline_score_report_path",
        "baseline_cost_usd",
        "baseline_duration_ms",
        "baseline_tree_depth",
        "baseline_tree_breadth",
        "baseline_node_count",
        "recursive_solved",
        "recursive_runtime_status",
        "recursive_score_status",
        "recursive_score_error",
        "recursive_score_log_path",
        "recursive_score_report_path",
        "recursive_cost_usd",
        "recursive_duration_ms",
        "recursive_tree_depth",
        "recursive_tree_breadth",
        "recursive_node_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        config = report.get("config", {})
        for task in report.get("tasks", []):
            baseline = task["baseline"]
            recursive = task["recursive"]
            writer.writerow(
                {
                    "benchmark": task.get("benchmark"),
                    "dataset_name": task.get("dataset_name"),
                    "dataset_split": task.get("dataset_split"),
                    "instance_id": task["instance_id"],
                    "repo": task["repo"],
                    "version": task["version"],
                    "complexity_score": task["complexity_score"],
                    "comparison": task["comparison"],
                    "config_fallback_model": config.get("fallback_model"),
                    "config_root_model": config.get("root_model"),
                    "config_child_model": config.get("child_model"),
                    "config_evaluation_backend": config.get("evaluation_backend"),
                    "config_evaluation_namespace": config.get("evaluation_namespace"),
                    "config_requested_limit": config.get("requested_limit"),
                    "config_max_concurrency": config.get("max_concurrency"),
                    "config_test_timeout_seconds": config.get("test_timeout_seconds"),
                    "config_manifest_id": config.get("manifest_id"),
                    "config_manifest_path": config.get("manifest_path"),
                    "config_dataset_sources": json.dumps(config.get("dataset_sources", [])),
                    "baseline_solved": baseline["solved"],
                    "baseline_runtime_status": baseline["runtime_status"],
                    "baseline_score_status": baseline["score"]["status"],
                    "baseline_score_error": baseline["score"]["error"],
                    "baseline_score_log_path": baseline["score"]["log_path"],
                    "baseline_score_report_path": baseline["score"]["report_path"],
                    "baseline_cost_usd": baseline["cost"]["total_usd"],
                    "baseline_duration_ms": baseline["duration_ms"],
                    "baseline_tree_depth": baseline["tree_depth"],
                    "baseline_tree_breadth": baseline["tree_breadth"],
                    "baseline_node_count": baseline["node_count"],
                    "recursive_solved": recursive["solved"],
                    "recursive_runtime_status": recursive["runtime_status"],
                    "recursive_score_status": recursive["score"]["status"],
                    "recursive_score_error": recursive["score"]["error"],
                    "recursive_score_log_path": recursive["score"]["log_path"],
                    "recursive_score_report_path": recursive["score"]["report_path"],
                    "recursive_cost_usd": recursive["cost"]["total_usd"],
                    "recursive_duration_ms": recursive["duration_ms"],
                    "recursive_tree_depth": recursive["tree_depth"],
                    "recursive_tree_breadth": recursive["tree_breadth"],
                    "recursive_node_count": recursive["node_count"],
                }
            )


def _mean(values: list[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[int | float]) -> float:
    return float(statistics.median(values)) if values else 0.0
