"""SWE-bench dataset loading and task selection."""

from __future__ import annotations

import json
import re
import shlex
import urllib.parse
import urllib.request
from pathlib import Path

from recursive_intelligence.benchmarks.models import SWEBenchTask

HF_DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_SUITE = "tier-a"
DEFAULT_SUITE_SIZE = 30
DEFAULT_REPO_CAP = 4

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


class SWEBenchLoader:
    """Download, parse, cache, and slice SWE-bench tasks."""

    def __init__(
        self,
        cache_dir: Path,
        dataset: str = DEFAULT_DATASET,
        split: str = DEFAULT_SPLIT,
    ) -> None:
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.split = split

    def load_tasks(self, refresh: bool = False) -> list[SWEBenchTask]:
        rows = self._load_rows(refresh=refresh)
        return [self._task_from_row(row) for row in rows]

    def load_suite(self, suite: str = DEFAULT_SUITE, refresh: bool = False) -> list[SWEBenchTask]:
        tasks = self.load_tasks(refresh=refresh)
        if suite == "tier-a":
            return select_tier_a(tasks)
        raise ValueError(f"Unsupported SWE-bench suite: {suite}")

    def _load_rows(self, refresh: bool) -> list[dict]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{self.dataset.replace('/', '__')}_{self.split}.json"
        if cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text())

        rows: list[dict] = []
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

    def _fetch_page(self, offset: int, length: int) -> dict:
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

    def _task_from_row(self, row: dict) -> SWEBenchTask:
        patch_files = parse_diff_files(row.get("patch", ""))
        test_files = parse_diff_files(row.get("test_patch", ""))
        test_directives = extract_test_directives(row.get("repo", ""), row.get("test_patch", ""))
        complexity = compute_complexity_score(
            len(patch_files),
            len(test_files),
            len(_parse_json_list(row.get("FAIL_TO_PASS", ""))),
        )
        return SWEBenchTask(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            version=row.get("version", ""),
            fail_to_pass=_parse_json_list(row.get("FAIL_TO_PASS", "")),
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


def select_tier_a(
    tasks: list[SWEBenchTask],
    target_size: int = DEFAULT_SUITE_SIZE,
    repo_cap: int = DEFAULT_REPO_CAP,
) -> list[SWEBenchTask]:
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

    selected: list[SWEBenchTask] = []
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


def resolve_test_command(task: SWEBenchTask) -> str:
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
