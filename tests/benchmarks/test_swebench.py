"""Tests for SWE-bench loading and suite selection."""

from recursive_intelligence.benchmarks.models import SWEBenchTask
from recursive_intelligence.benchmarks.swebench import (
    resolve_python_requirement,
    resolve_test_command,
    select_tier_a,
)


def _task(
    instance_id: str,
    repo: str,
    patch_files: int,
    test_files: int,
    score: int,
    version: str = "1.0",
    test_directives: list[str] | None = None,
    test_command: str | None = None,
) -> SWEBenchTask:
    return SWEBenchTask(
        instance_id=instance_id,
        repo=repo,
        base_commit="abc123",
        problem_statement="fix it",
        patch="",
        test_patch="",
        version=version,
        fail_to_pass=["a"] * max(1, score),
        pass_to_pass=["b"],
        gold_patch_files=[f"file_{i}.py" for i in range(patch_files)],
        test_files=[f"test_{i}.py" for i in range(test_files)],
        test_directives=test_directives or [],
        complexity_score=score,
        test_command=test_command,
    )


def test_select_tier_a_prefers_multifile_and_respects_repo_cap():
    tasks = [
        _task("a", "django/django", patch_files=4, test_files=2, score=20),
        _task("b", "django/django", patch_files=3, test_files=2, score=18),
        _task("c", "django/django", patch_files=2, test_files=2, score=16),
        _task("d", "django/django", patch_files=2, test_files=1, score=14),
        _task("e", "pytest-dev/pytest", patch_files=5, test_files=1, score=19),
        _task("f", "pytest-dev/pytest", patch_files=4, test_files=1, score=17),
        _task("g", "sympy/sympy", patch_files=1, test_files=1, score=30),
    ]

    selected = select_tier_a(tasks, target_size=4, repo_cap=2)

    assert [task.instance_id for task in selected] == ["a", "e", "b", "f"]


def test_resolve_test_command_appends_repo_directives():
    task = _task(
        "django-1",
        "django/django",
        patch_files=2,
        test_files=1,
        score=10,
        version="5.0",
        test_directives=["aggregation.tests", "queries.tests"],
    )

    command = resolve_test_command(task)

    assert command == (
        "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 "
        "aggregation.tests queries.tests"
    )


def test_resolve_test_command_prefers_explicit_override():
    task = _task(
        "local-1",
        "/tmp/local-repo",
        patch_files=2,
        test_files=1,
        score=10,
        test_command="python3 -m pytest tests/test_app.py",
    )

    assert resolve_test_command(task) == "python3 -m pytest tests/test_app.py"


def test_resolve_python_requirement_for_legacy_sympy():
    task = _task(
        "sympy-1",
        "sympy/sympy",
        patch_files=2,
        test_files=1,
        score=10,
        version="1.1",
    )

    requirement = resolve_python_requirement(task)

    assert requirement is not None
    assert requirement.maximum == (3, 9)
    assert "collections.Mapping" in requirement.reason
