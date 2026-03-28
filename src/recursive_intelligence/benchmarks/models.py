"""Benchmark data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from recursive_intelligence.adapters.base import CostRecord


@dataclass
class SWEBenchTask:
    """Normalized SWE-bench task metadata."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    version: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    hints_text: str = ""
    created_at: str = ""
    environment_setup_commit: str | None = None
    difficulty: str | None = None
    gold_patch_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    test_directives: list[str] = field(default_factory=list)
    complexity_score: int = 0
    test_command: str | None = None

    def build_prompt(self) -> str:
        prompt = (
            f"Resolve SWE-bench instance {self.instance_id} in repo {self.repo}.\n\n"
            f"Problem statement:\n{self.problem_statement.strip()}\n"
        )
        if self.hints_text.strip():
            prompt += f"\nHints:\n{self.hints_text.strip()}\n"
        prompt += (
            "\nSuccess criteria:\n"
            "- implement the fix in code\n"
            "- make the relevant tests pass\n"
            "- avoid regressing existing tests\n"
        )
        return prompt

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchScore:
    """Deterministic patch application + test result."""

    status: str
    patch_applied: bool
    tests_passed: bool
    exit_code: int | None
    test_command: str
    log_path: str | None = None
    error: str | None = None
    python_executable: str | None = None
    python_requirement: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkModeResult:
    """Result of running one solver mode for one task."""

    mode: str
    run_id: str | None
    runtime_status: str
    solved: bool
    changed_files: list[str]
    cost: CostRecord
    duration_ms: int
    session_ids: list[str]
    session_count: int
    node_count: int
    tree_depth: int
    tree_breadth: int
    patch_path: str | None
    patch_bytes: int
    ri_artifacts_path: str | None
    score: PatchScore
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskBenchmarkResult:
    """Flat-vs-recursive comparison for one benchmark task."""

    instance_id: str
    repo: str
    version: str
    complexity_score: int
    baseline: BenchmarkModeResult
    recursive: BenchmarkModeResult
    comparison: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModeAggregate:
    """Aggregate statistics for one mode."""

    solve_rate: float
    solved: int
    total: int
    eligible: int
    unsupported: int
    avg_cost_usd: float
    avg_latency_ms: float
    median_latency_ms: float
    avg_tree_depth: float
    avg_tree_breadth: float
    avg_node_count: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonAggregate:
    """Head-to-head comparison counts."""

    baseline_win: int
    recursive_win: int
    tie_solved: int
    tie_failed: int
    unsupported: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSuiteReport:
    """Full persisted report for a benchmark suite run."""

    run_id: str
    benchmark: str
    suite: str
    dataset: str
    split: str
    created_at: str
    task_count: int
    baseline: ModeAggregate
    recursive: ModeAggregate
    comparison: ComparisonAggregate
    tasks: list[TaskBenchmarkResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
