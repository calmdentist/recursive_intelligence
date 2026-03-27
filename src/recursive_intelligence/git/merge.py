"""Cherry-pick integration for merging child work into parent worktrees."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class MergeError(Exception):
    pass


@dataclass
class IntegrationResult:
    status: str  # "integrated", "conflict"
    child_id: str
    commit_sha: str
    conflict_files: list[str] | None = None


def cherry_pick_child(
    parent_worktree: Path,
    child_commit_sha: str,
    child_id: str,
    base_sha: str | None = None,
) -> IntegrationResult:
    """Cherry-pick a child's work into the parent worktree.

    If base_sha is provided, cherry-picks the range (base_sha..child_commit_sha].
    Otherwise, cherry-picks just the single commit.
    """
    try:
        if base_sha and base_sha != child_commit_sha:
            # Cherry-pick the full range of commits from base to child HEAD
            _git(parent_worktree, "cherry-pick", f"{base_sha}..{child_commit_sha}")
        else:
            _git(parent_worktree, "cherry-pick", child_commit_sha)
        new_sha = _git(parent_worktree, "rev-parse", "HEAD").strip()
        return IntegrationResult(
            status="integrated",
            child_id=child_id,
            commit_sha=new_sha,
        )
    except MergeError as e:
        # Check for conflict
        conflict_files = _get_conflict_files(parent_worktree)
        if conflict_files:
            return IntegrationResult(
                status="conflict",
                child_id=child_id,
                commit_sha=child_commit_sha,
                conflict_files=conflict_files,
            )
        raise


def abort_cherry_pick(worktree: Path) -> None:
    """Abort an in-progress cherry-pick."""
    try:
        _git(worktree, "cherry-pick", "--abort")
    except MergeError:
        pass


def get_diff(worktree: Path, base_ref: str = "HEAD~1") -> str:
    """Get the diff for the latest commit."""
    try:
        return _git(worktree, "diff", base_ref, "HEAD")
    except MergeError:
        return ""


def get_changed_files(worktree: Path, base_ref: str = "HEAD~1") -> list[str]:
    """List files changed in the latest commit."""
    try:
        output = _git(worktree, "diff", "--name-only", base_ref, "HEAD")
        return [f for f in output.strip().splitlines() if f]
    except MergeError:
        return []


def _get_conflict_files(worktree: Path) -> list[str]:
    try:
        output = _git(worktree, "diff", "--name-only", "--diff-filter=U")
        return [f for f in output.strip().splitlines() if f]
    except MergeError:
        return []


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MergeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout
