"""Tests for benchmark dataset loading and suite selection."""

from recursive_intelligence.benchmarks.models import SWEBenchTask
from recursive_intelligence.benchmarks.swebench import (
    FeatureBenchLoader,
    SWEBenchProLoader,
    resolve_python_requirement,
    resolve_test_command,
    score_recursive_hard_task,
    select_recursive_hard,
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


def test_swebench_pro_loader_normalizes_rows(tmp_path):
    loader = SWEBenchProLoader(tmp_path)
    row = {
        "repo": "NodeBB/NodeBB",
        "instance_id": "nodebb-1",
        "base_commit": "abc123",
        "patch": "diff --git a/src/a.js b/src/a.js\n--- a/src/a.js\n+++ b/src/a.js\n",
        "test_patch": "diff --git a/test/a.js b/test/a.js\n--- a/test/a.js\n+++ b/test/a.js\n",
        "problem_statement": "Fix the admin validation flow",
        "requirements": "Preserve existing email confirmation behavior.",
        "interface": "Expose the new pending/expired status in the admin panel.",
        "repo_language": "javascript",
        "fail_to_pass": "[\"test/a.js::test_pending\"]",
        "pass_to_pass": "[\"test/a.js::test_existing\"]",
        "issue_specificity": "[\"major_bug\"]",
        "issue_categories": "[\"database_knowledge\", \"ui_ux_knowledge\"]",
        "before_repo_set_cmd": "git checkout test/a.js",
        "selected_test_files_to_run": "[\"test/a.js\", \"test/b.js\"]",
        "dockerhub_tag": "nodebb:test",
    }

    task = loader._task_from_row(row)

    assert task.benchmark == "swebench_pro"
    assert task.dataset_name == loader.dataset
    assert task.dataset_split == loader.split
    assert task.test_directives == ["test/a.js", "test/b.js"]
    assert "Requirements:" in task.hints_text
    assert task.extra["issue_categories"] == ["database_knowledge", "ui_ux_knowledge"]


def test_featurebench_loader_normalizes_repo_settings(tmp_path):
    loader = FeatureBenchLoader(tmp_path, split="lite")
    row = {
        "instance_id": "trl-1",
        "patch": "diff --git a/trl/a.py b/trl/a.py\n--- a/trl/a.py\n+++ b/trl/a.py\n",
        "test_patch": "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n",
        "FAIL_TO_PASS": ["tests/test_a.py::test_it"],
        "PASS_TO_PASS": ["tests/test_a.py::test_existing"],
        "image_name": "python312",
        "repo": "huggingface/trl",
        "base_commit": "def456",
        "problem_statement": "Implement the missing multimodal helper interface.",
        "repo_settings": (
            "{\"repository\":\"huggingface/trl\",\"commit\":\"02a3477\",\"test_cmd\":\"pytest -rA\","
            "\"timeout_run\":1200,\"timeout_one\":120}"
        ),
    }

    task = loader._task_from_row(row)

    assert task.benchmark == "featurebench"
    assert task.dataset_name == loader.dataset
    assert task.dataset_split == "lite"
    assert task.test_command == "pytest -rA"
    assert task.version == "02a3477"
    assert task.extra["repo_settings"]["repository"] == "huggingface/trl"


def test_select_recursive_hard_assigns_scores_and_respects_repo_cap():
    tasks = [
        _task("a", "repo/a", patch_files=5, test_files=3, score=5),
        _task("b", "repo/a", patch_files=4, test_files=3, score=5),
        _task("c", "repo/a", patch_files=4, test_files=2, score=4),
        _task("d", "repo/b", patch_files=6, test_files=4, score=5),
        _task("e", "repo/c", patch_files=2, test_files=1, score=2),
    ]
    for task in tasks:
        task.problem_statement = (
            "Investigate the runtime behavior across multiple modules and implement a coordinated fix "
            "without regressing the existing integration coverage."
        )
        task.extra["issue_categories"] = ["api", "database", "ui"]

    selected = select_recursive_hard(tasks, target_size=2, repo_cap=1)

    assert [task.instance_id for task in selected] == ["d", "a"]
    assert score_recursive_hard_task(selected[0])["total"] >= score_recursive_hard_task(selected[1])["total"]
    assert all(task.selection_scores["total"] >= 28 for task in selected)
