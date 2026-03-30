"""Helpers for running repo-local test commands consistently."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


def run_test_command(
    command: str,
    repo_dir: Path,
    timeout_seconds: int,
    python_executable: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a test command from a repository directory."""
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
