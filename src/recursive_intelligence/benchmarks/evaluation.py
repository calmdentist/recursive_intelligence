"""Benchmark scoring backends."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from recursive_intelligence.benchmarks.models import PatchScore, SWEBenchTask
from recursive_intelligence.benchmarks.swebench import resolve_python_requirement, resolve_test_command


class PatchEvaluator(Protocol):
    """Score a generated patch for one benchmark task."""

    def score_patch(self, task: SWEBenchTask, patch_text: str, task_dir: Path, mode: str) -> PatchScore:
        """Return a deterministic score record for the patch."""


class LocalPatchEvaluator:
    """Host-local patch application plus repo test execution.

    This backend remains useful for unit tests and local fixture repos.
    Production SWE-bench runs should use ``OfficialHarnessEvaluator``.
    """

    def __init__(self, timeout_seconds: int = 1800, keep_task_dirs: bool = False, cleanup_task_dirs: bool = True):
        self.timeout_seconds = timeout_seconds
        self.keep_task_dirs = keep_task_dirs
        self.cleanup_task_dirs = cleanup_task_dirs

    def score_patch(self, task: SWEBenchTask, patch_text: str, task_dir: Path, mode: str) -> PatchScore:
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
                self.timeout_seconds,
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


class OfficialHarnessEvaluator:
    """Score patches via the official SWE-bench Docker harness."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        timeout_seconds: int = 1800,
        namespace: str | None = None,
        python_executable: str | None = None,
        max_workers: int = 1,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.timeout_seconds = timeout_seconds
        self.namespace = _default_swebench_namespace() if namespace is None else namespace
        self.python_executable = python_executable or sys.executable
        self.max_workers = max_workers
        self._ensure_prerequisites()

    def score_patch(self, task: SWEBenchTask, patch_text: str, task_dir: Path, mode: str) -> PatchScore:
        if not patch_text.strip():
            return PatchScore(
                status="no_patch",
                patch_applied=False,
                tests_passed=False,
                exit_code=None,
                test_command=_harness_entrypoint(self.python_executable),
                error="Solver produced an empty patch",
            )

        eval_dir = task_dir / f"{mode}-eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_id = _official_run_id(task.instance_id, mode)
        model_name = f"recursive-intelligence/{mode}"
        sanitized_model = _sanitize_model_name(model_name)
        harness_log_path = eval_dir / "harness.log"
        predictions_path = eval_dir / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "instance_id": task.instance_id,
                    "model_name_or_path": model_name,
                    "model_patch": patch_text,
                }
            )
            + "\n"
        )

        command = [
            self.python_executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.dataset_name,
            "--split",
            self.split,
            "--instance_ids",
            task.instance_id,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(self.max_workers),
            "--run_id",
            run_id,
            "--timeout",
            str(self.timeout_seconds),
            "--report_dir",
            str(eval_dir),
            "--namespace",
            self.namespace,
        ]
        completed = subprocess.run(command, cwd=str(eval_dir), capture_output=True, text=True)
        harness_log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""))

        instance_dir = eval_dir / "logs" / "run_evaluation" / run_id / sanitized_model / task.instance_id
        report_path = instance_dir / "report.json"
        instance_log_path = instance_dir / "run_instance.log"
        test_output_path = instance_dir / "test_output.txt"
        selected_log_path = test_output_path if test_output_path.exists() else instance_log_path if instance_log_path.exists() else harness_log_path

        if report_path.exists():
            report = json.loads(report_path.read_text())
            resolved = bool(report.get(task.instance_id, {}).get("resolved", False))
            return PatchScore(
                status="passed" if resolved else "failed",
                patch_applied=True,
                tests_passed=resolved,
                exit_code=completed.returncode,
                test_command=shlex.join(command),
                log_path=str(selected_log_path),
                report_path=str(report_path),
                error=None if completed.returncode == 0 else _command_error_text(completed),
            )

        patch_applied = _detect_patch_applied(instance_log_path, harness_log_path)
        status = "patch_failed" if not patch_applied else "evaluation_error"
        return PatchScore(
            status=status,
            patch_applied=patch_applied,
            tests_passed=False,
            exit_code=completed.returncode,
            test_command=shlex.join(command),
            log_path=str(selected_log_path),
            error=_command_error_text(completed)
            or "SWE-bench evaluation did not produce an instance report.",
        )

    def _ensure_prerequisites(self) -> None:
        if importlib.util.find_spec("swebench") is None:
            raise RuntimeError(
                "The official SWE-bench harness is not installed. "
                "Install dependencies and reinstall `rari` so `swebench` is available."
            )
        if shutil.which("docker") is None:
            raise RuntimeError("Docker is required for official SWE-bench evaluation but was not found on PATH.")


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


def _default_swebench_namespace() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return ""
    return "swebench"


def _official_run_id(instance_id: str, mode: str) -> str:
    sanitized = instance_id.replace("/", "-").replace(":", "-")
    return f"eval-{mode}-{sanitized[:40]}-{uuid.uuid4().hex[:8]}"


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _command_error_text(completed: subprocess.CompletedProcess[str]) -> str | None:
    parts = [part.strip() for part in (completed.stderr or "", completed.stdout or "") if part.strip()]
    if not parts:
        return None
    return "\n".join(parts)


def _detect_patch_applied(instance_log_path: Path, harness_log_path: Path) -> bool:
    for path in (instance_log_path, harness_log_path):
        if not path.exists():
            continue
        content = path.read_text(errors="ignore")
        if "Patch Apply Failed" in content:
            return False
        if "Applied Patch" in content:
            return True
    return False


def _harness_entrypoint(python_executable: str) -> str:
    return f"{python_executable} -m swebench.harness.run_evaluation"


__all__ = [
    "LocalPatchEvaluator",
    "OfficialHarnessEvaluator",
    "PatchEvaluator",
    "_run_test_command",
    "_select_task_python",
]
