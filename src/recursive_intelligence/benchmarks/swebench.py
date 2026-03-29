"""Benchmark dataset loading, normalization, and suite selection."""

from __future__ import annotations

import json
import re
import shlex
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recursive_intelligence.benchmarks.models import BenchmarkTask, SWEBenchTask

HF_DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"
DEFAULT_SWEBENCH_DATASET = DEFAULT_DATASET
DEFAULT_SWEBENCH_PRO_DATASET = "ScaleAI/SWE-bench_Pro"
DEFAULT_FEATUREBENCH_DATASET = "LiberCoders/FeatureBench"
DEFAULT_SPLIT = "test"
DEFAULT_SWEBENCH_PRO_SPLIT = "test"
DEFAULT_FEATUREBENCH_SPLIT = "full"
DEFAULT_SUITE = "tier-a"
DEFAULT_SUITE_SIZE = 30
DEFAULT_REPO_CAP = 4
RECURSIVE_HARD_MIN_SCORE = 28

NON_TEST_EXTS = {
    ".json",
    ".png",
    ".csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
}

REPO_TEST_COMMANDS: dict[str, str] = {
    "astropy/astropy": "pytest -rA",
    "django/django": "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    "matplotlib/matplotlib": "pytest -rA",
    "mwaskom/seaborn": "pytest --no-header -rA",
    "pallets/flask": "pytest -rA",
    "psf/requests": "pytest -rA",
    "pydata/xarray": "pytest -rA",
    "pylint-dev/pylint": "pytest -rA",
    "pytest-dev/pytest": "pytest -rA",
    "scikit-learn/scikit-learn": "pytest -rA",
    "sphinx-doc/sphinx": "tox --current-env -epy39 -v --",
    "sympy/sympy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
}

REPO_TEST_COMMAND_OVERRIDES: dict[tuple[str, str], str] = {
    ("django/django", "1.9"): "./tests/runtests.py --verbosity 2",
}


@dataclass(frozen=True)
class PythonRequirement:
    """Interpreter requirement needed to score a benchmark task."""

    minimum: tuple[int, int] | None = None
    maximum: tuple[int, int] | None = None
    reason: str = ""

    def matches(self, version: tuple[int, int, int]) -> bool:
        major_minor = version[:2]
        if self.minimum is not None and major_minor < self.minimum:
            return False
        if self.maximum is not None and major_minor > self.maximum:
            return False
        return True

    def describe(self) -> str:
        bounds: list[str] = []
        if self.minimum is not None:
            bounds.append(f">={self.minimum[0]}.{self.minimum[1]}")
        if self.maximum is not None:
            bounds.append(f"<={self.maximum[0]}.{self.maximum[1]}")
        summary = " and ".join(bounds) if bounds else "any"
        if self.reason:
            return f"Python {summary} ({self.reason})"
        return f"Python {summary}"


class HFDatasetLoader:
    """Download, parse, and cache benchmark datasets from Hugging Face."""

    benchmark_name = "benchmark"

    def __init__(self, cache_dir: Path, dataset: str, split: str) -> None:
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.split = split

    def load_tasks(self, refresh: bool = False) -> list[BenchmarkTask]:
        rows = self._load_rows(refresh=refresh)
        return [self._task_from_row(row) for row in rows]

    def _load_rows(self, refresh: bool) -> list[dict[str, Any]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{self.dataset.replace('/', '__')}_{self.split}.json"
        if cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text())

        rows: list[dict[str, Any]] = []
        offset = 0
        total = None
        while total is None or len(rows) < total:
            payload = self._fetch_page(offset=offset, length=100)
            page_rows = [entry["row"] for entry in payload.get("rows", [])]
            if not page_rows:
                break
            rows.extend(page_rows)
            offset += len(page_rows)
            total = payload.get("num_rows_total", len(rows))

        cache_path.write_text(json.dumps(rows, indent=2))
        return rows

    def _fetch_page(self, offset: int, length: int) -> dict[str, Any]:
        params = urllib.parse.urlencode(
            {
                "dataset": self.dataset,
                "config": "default",
                "split": self.split,
                "offset": offset,
                "length": length,
            }
        )
        with urllib.request.urlopen(f"{HF_DATASET_ROWS_URL}?{params}", timeout=30) as response:
            return json.load(response)

    def _task_from_row(self, row: dict[str, Any]) -> BenchmarkTask:
        raise NotImplementedError


class SWEBenchLoader(HFDatasetLoader):
    """Loader for SWE-bench Verified."""

    benchmark_name = "swebench"

    def __init__(
        self,
        cache_dir: Path,
        dataset: str = DEFAULT_SWEBENCH_DATASET,
        split: str = DEFAULT_SPLIT,
    ) -> None:
        super().__init__(cache_dir=cache_dir, dataset=dataset, split=split)

    def load_suite(self, suite: str = DEFAULT_SUITE, refresh: bool = False) -> list[SWEBenchTask]:
        tasks = self.load_tasks(refresh=refresh)
        if suite == "tier-a":
            return select_tier_a(tasks)
        raise ValueError(f"Unsupported SWE-bench suite: {suite}")

    def _task_from_row(self, row: dict[str, Any]) -> SWEBenchTask:
        patch_files = parse_diff_files(row.get("patch", ""))
        test_files = parse_diff_files(row.get("test_patch", ""))
        test_directives = extract_test_directives(row.get("repo", ""), row.get("test_patch", ""))
        fail_to_pass = _parse_json_list(row.get("FAIL_TO_PASS", ""))
        complexity = compute_complexity_score(len(patch_files), len(test_files), len(fail_to_pass))
        return SWEBenchTask(
            benchmark=self.benchmark_name,
            dataset_name=self.dataset,
            dataset_split=self.split,
            evaluation_backend="swebench_harness",
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            version=row.get("version", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=_parse_json_list(row.get("PASS_TO_PASS", "")),
            hints_text=row.get("hints_text", ""),
            created_at=row.get("created_at", ""),
            environment_setup_commit=row.get("environment_setup_commit"),
            difficulty=row.get("difficulty"),
            gold_patch_files=patch_files,
            test_files=test_files,
            test_directives=test_directives,
            complexity_score=complexity,
        )


class SWEBenchProLoader(HFDatasetLoader):
    """Loader for SWE-Bench Pro."""

    benchmark_name = "swebench_pro"

    def __init__(
        self,
        cache_dir: Path,
        dataset: str = DEFAULT_SWEBENCH_PRO_DATASET,
        split: str = DEFAULT_SWEBENCH_PRO_SPLIT,
    ) -> None:
        super().__init__(cache_dir=cache_dir, dataset=dataset, split=split)

    def _task_from_row(self, row: dict[str, Any]) -> BenchmarkTask:
        patch_files = parse_diff_files(row.get("patch", ""))
        test_files = parse_diff_files(row.get("test_patch", ""))
        selected_tests = _parse_json_list(row.get("selected_test_files_to_run", ""))
        fail_to_pass = _parse_json_list(row.get("fail_to_pass", ""))
        issue_specificity = _parse_json_list(row.get("issue_specificity", ""))
        issue_categories = _parse_json_list(row.get("issue_categories", ""))
        requirements = row.get("requirements", "").strip()
        interface = row.get("interface", "").strip()
        hints: list[str] = []
        if requirements:
            hints.append(f"Requirements:\n{requirements}")
        if interface:
            hints.append(f"Interface:\n{interface}")
        complexity = compute_complexity_score(
            len(patch_files),
            max(len(test_files), len(selected_tests)),
            len(fail_to_pass),
        )
        return BenchmarkTask(
            benchmark=self.benchmark_name,
            dataset_name=self.dataset,
            dataset_split=self.split,
            evaluation_backend="swebench_pro",
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            version=row.get("repo_language", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=_parse_json_list(row.get("pass_to_pass", "")),
            hints_text="\n\n".join(hints),
            gold_patch_files=patch_files,
            test_files=test_files or selected_tests,
            test_directives=selected_tests,
            complexity_score=complexity,
            extra={
                "requirements": requirements,
                "interface": interface,
                "repo_language": row.get("repo_language", ""),
                "issue_specificity": issue_specificity,
                "issue_categories": issue_categories,
                "before_repo_set_cmd": row.get("before_repo_set_cmd", ""),
                "selected_test_files_to_run": selected_tests,
                "dockerhub_tag": row.get("dockerhub_tag", ""),
            },
        )


class FeatureBenchLoader(HFDatasetLoader):
    """Loader for FeatureBench."""

    benchmark_name = "featurebench"

    def __init__(
        self,
        cache_dir: Path,
        dataset: str = DEFAULT_FEATUREBENCH_DATASET,
        split: str = DEFAULT_FEATUREBENCH_SPLIT,
    ) -> None:
        super().__init__(cache_dir=cache_dir, dataset=dataset, split=split)

    def _task_from_row(self, row: dict[str, Any]) -> BenchmarkTask:
        patch_files = parse_diff_files(row.get("patch", ""))
        test_files = parse_diff_files(row.get("test_patch", ""))
        repo_settings = _parse_json_object(row.get("repo_settings", ""))
        fail_to_pass = _normalize_string_list(row.get("FAIL_TO_PASS", []))
        complexity = compute_complexity_score(len(patch_files), len(test_files), len(fail_to_pass))
        return BenchmarkTask(
            benchmark=self.benchmark_name,
            dataset_name=self.dataset,
            dataset_split=self.split,
            evaluation_backend="featurebench",
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            version=str(repo_settings.get("commit", "")),
            fail_to_pass=fail_to_pass,
            pass_to_pass=_normalize_string_list(row.get("PASS_TO_PASS", [])),
            gold_patch_files=patch_files,
            test_files=test_files,
            complexity_score=complexity,
            test_command=_coerce_str(repo_settings.get("test_cmd")),
            extra={
                "image_name": row.get("image_name", ""),
                "repo_settings": repo_settings,
                "test_dynamic_cmd": _coerce_str(repo_settings.get("test_dynamic_cmd")),
                "timeout_run": repo_settings.get("timeout_run"),
                "timeout_one": repo_settings.get("timeout_one"),
            },
        )


def build_loader(
    benchmark: str,
    cache_dir: Path,
    dataset: str | None = None,
    split: str | None = None,
) -> HFDatasetLoader:
    normalized = benchmark.lower()
    if normalized in {"swebench", "swebench_verified"}:
        return SWEBenchLoader(cache_dir, dataset=dataset or DEFAULT_SWEBENCH_DATASET, split=split or DEFAULT_SPLIT)
    if normalized == "swebench_pro":
        return SWEBenchProLoader(
            cache_dir,
            dataset=dataset or DEFAULT_SWEBENCH_PRO_DATASET,
            split=split or DEFAULT_SWEBENCH_PRO_SPLIT,
        )
    if normalized == "featurebench":
        return FeatureBenchLoader(
            cache_dir,
            dataset=dataset or DEFAULT_FEATUREBENCH_DATASET,
            split=split or DEFAULT_FEATUREBENCH_SPLIT,
        )
    raise ValueError(f"Unsupported benchmark dataset: {benchmark}")


def select_tier_a(
    tasks: list[BenchmarkTask],
    target_size: int = DEFAULT_SUITE_SIZE,
    repo_cap: int = DEFAULT_REPO_CAP,
) -> list[BenchmarkTask]:
    """Deterministic ~30-task slice biased toward multi-file issues."""
    ranked = sorted(
        tasks,
        key=lambda task: (
            -task.complexity_score,
            -len(task.gold_patch_files),
            -len(task.test_files),
            task.instance_id,
        ),
    )

    selected: list[BenchmarkTask] = []
    per_repo: dict[str, int] = {}
    for task in ranked:
        if len(selected) >= target_size:
            break
        if len(task.gold_patch_files) < 2:
            continue
        if per_repo.get(task.repo, 0) >= repo_cap:
            continue
        selected.append(task)
        per_repo[task.repo] = per_repo.get(task.repo, 0) + 1

    if len(selected) < target_size:
        selected_ids = {task.instance_id for task in selected}
        for task in ranked:
            if len(selected) >= target_size:
                break
            if task.instance_id in selected_ids:
                continue
            selected.append(task)
            selected_ids.add(task.instance_id)

    return selected


def score_recursive_hard_task(task: BenchmarkTask) -> dict[str, int]:
    """Proxy rubric for hard, long-context benchmark slices."""
    issue_categories = _normalize_string_list(task.extra.get("issue_categories", []))
    prompt_words = len(task.problem_statement.split())
    explicit_path_mentions = bool(re.search(r"\b(?:[\w.-]+/)+[\w.-]+\b", task.problem_statement))
    timeout_run = _coerce_int(task.extra.get("timeout_run"))
    if timeout_run is None:
        repo_settings = task.extra.get("repo_settings", {})
        if isinstance(repo_settings, dict):
            timeout_run = _coerce_int(repo_settings.get("timeout_run"))

    context_spread = _bucket_score(max(len(task.gold_patch_files), len(task.test_files)), [2, 3, 5, 8])

    integration_pressure = _bucket_score(len(task.gold_patch_files) + len(task.test_files), [4, 6, 9, 13])
    if len(issue_categories) >= 3 or (len(task.gold_patch_files) >= 4 and len(task.test_files) >= 2):
        integration_pressure = min(integration_pressure + 1, 4)

    runtime_dependence = _bucket_score(max(len(task.fail_to_pass), len(task.test_files)), [2, 4, 8, 16])
    if task.extra.get("before_repo_set_cmd"):
        runtime_dependence = min(runtime_dependence + 1, 4)
    if timeout_run is not None and timeout_run >= 900:
        runtime_dependence = min(runtime_dependence + 1, 4)

    underspecification = _bucket_score(prompt_words, [80, 160, 280, 420])
    if explicit_path_mentions:
        underspecification = max(underspecification - 1, 0)

    decomposability = _bucket_score(
        len(task.gold_patch_files) + max(len(task.test_files) - 1, 0) + len(issue_categories),
        [4, 6, 9, 12],
    )
    if len(issue_categories) >= 4:
        decomposability = min(decomposability + 1, 4)

    total = (
        (3 * context_spread)
        + (2 * integration_pressure)
        + (2 * runtime_dependence)
        + (2 * underspecification)
        + (3 * decomposability)
    )
    scores = {
        "context_spread": context_spread,
        "integration_pressure": integration_pressure,
        "runtime_dependence": runtime_dependence,
        "underspecification": underspecification,
        "decomposability": decomposability,
        "total": total,
    }
    task.selection_scores = scores
    return scores


def select_recursive_hard(
    tasks: list[BenchmarkTask],
    target_size: int,
    repo_cap: int = 2,
    min_total: int = RECURSIVE_HARD_MIN_SCORE,
) -> list[BenchmarkTask]:
    """Deterministically select hard, multi-file tasks from a benchmark dataset."""
    ranked = sorted(
        tasks,
        key=lambda task: (
            -score_recursive_hard_task(task)["total"],
            -len(task.gold_patch_files),
            -len(task.test_files),
            task.instance_id,
        ),
    )

    selected: list[BenchmarkTask] = []
    selected_ids: set[str] = set()
    per_repo: dict[str, int] = {}

    def _eligible(task: BenchmarkTask) -> bool:
        scores = task.selection_scores or score_recursive_hard_task(task)
        return (
            scores.get("context_spread", 0) >= 2
            and scores.get("decomposability", 0) >= 2
            and scores.get("total", 0) >= min_total
        )

    for task in ranked:
        if len(selected) >= target_size:
            break
        if not _eligible(task):
            continue
        if per_repo.get(task.repo, 0) >= repo_cap:
            continue
        selected.append(task)
        selected_ids.add(task.instance_id)
        per_repo[task.repo] = per_repo.get(task.repo, 0) + 1

    if len(selected) < target_size:
        for task in ranked:
            if len(selected) >= target_size:
                break
            if task.instance_id in selected_ids:
                continue
            if per_repo.get(task.repo, 0) >= repo_cap:
                continue
            selected.append(task)
            selected_ids.add(task.instance_id)
            per_repo[task.repo] = per_repo.get(task.repo, 0) + 1

    return selected


def resolve_test_command(task: BenchmarkTask) -> str:
    """Resolve the repo-specific test command plus directives for one task."""
    if task.test_command:
        return task.test_command

    base_command = REPO_TEST_COMMAND_OVERRIDES.get((task.repo, task.version))
    if base_command is None:
        base_command = REPO_TEST_COMMANDS.get(task.repo)
    if base_command is None:
        raise ValueError(f"No benchmark test command configured for repo {task.repo}")

    if not task.test_directives:
        return base_command

    quoted_directives = " ".join(shlex.quote(directive) for directive in task.test_directives)
    return f"{base_command} {quoted_directives}".strip()


def resolve_python_requirement(task: BenchmarkTask) -> PythonRequirement | None:
    """Return interpreter constraints needed to run a task's repo tests."""
    if task.repo == "sympy/sympy" and task.version == "1.1":
        return PythonRequirement(
            maximum=(3, 9),
            reason="SymPy 1.1 imports collections.Mapping, removed in Python 3.10+",
        )
    return None


def parse_diff_files(diff_text: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", diff_text, flags=re.MULTILINE):
        path = match.group(2)
        if path not in seen:
            seen.add(path)
            files.append(path)
    return files


def extract_test_directives(repo: str, test_patch: str) -> list[str]:
    directives = [
        path
        for path in parse_diff_files(test_patch)
        if Path(path).suffix.lower() not in NON_TEST_EXTS
    ]

    if repo == "django/django":
        transformed = []
        for directive in directives:
            value = directive[:-3] if directive.endswith(".py") else directive
            if value.startswith("tests/"):
                value = value[len("tests/") :]
            transformed.append(value.replace("/", "."))
        return transformed

    return directives


def compute_complexity_score(patch_files: int, test_files: int, fail_to_pass: int) -> int:
    return (patch_files * 4) + (test_files * 2) + min(fail_to_pass, 5)


def _bucket_score(value: int, thresholds: list[int]) -> int:
    score = 0
    for threshold in thresholds:
        if value >= threshold:
            score += 1
    return min(score, 4)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return _parse_json_list(value)
    return []


def _parse_json_list(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [line.strip() for line in raw.splitlines() if line.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _parse_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw or not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
