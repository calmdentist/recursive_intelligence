"""Git worktree creation, naming, and cleanup."""

from __future__ import annotations

import subprocess
from pathlib import Path

from recursive_intelligence.runtime.node_fsm import task_hash_short


class WorktreeError(Exception):
    pass


def branch_name(run_id: str, node_id: str, task_spec: str) -> str:
    """Generate a human-readable branch name: ri/<run_id_short>/<node_id_short>/<task_hash>."""
    run_short = run_id.replace("run-", "")[:8]
    node_short = node_id.replace("node-", "")[:8]
    t_hash = task_hash_short(task_spec)
    return f"ri/{run_short}/{node_short}/{t_hash}"


def create_worktree(
    repo_root: Path,
    worktrees_dir: Path,
    node_id: str,
    branch: str,
    base_ref: str = "HEAD",
) -> Path:
    """Create a git worktree for a node. Idempotent by path."""
    wt_path = worktrees_dir / node_id
    if wt_path.exists():
        return wt_path

    wt_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "worktree", "add", "-b", branch, str(wt_path), base_ref)
    return wt_path


def remove_worktree(repo_root: Path, worktree_path: Path) -> None:
    """Remove a git worktree and prune."""
    if worktree_path.exists():
        _git(repo_root, "worktree", "remove", "--force", str(worktree_path))
    _git(repo_root, "worktree", "prune")


def list_worktrees(repo_root: Path) -> list[dict[str, str]]:
    """List all git worktrees."""
    result = _git(repo_root, "worktree", "list", "--porcelain")
    worktrees = []
    current: dict[str, str] = {}
    for line in result.splitlines():
        if line.startswith("worktree "):
            if current:
                worktrees.append(current)
            current = {"path": line.split(" ", 1)[1]}
        elif line.startswith("HEAD "):
            current["head"] = line.split(" ", 1)[1]
        elif line.startswith("branch "):
            current["branch"] = line.split(" ", 1)[1]
    if current:
        worktrees.append(current)
    return worktrees


def ensure_clean_repo(repo_root: Path) -> None:
    """Raise if the repo has uncommitted changes."""
    status = _git(repo_root, "status", "--porcelain")
    if status.strip():
        raise WorktreeError(
            f"Repo at {repo_root} has uncommitted changes. Commit or stash before starting a run."
        )


def get_head_sha(repo_root: Path) -> str:
    return _git(repo_root, "rev-parse", "HEAD").strip()


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WorktreeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout
