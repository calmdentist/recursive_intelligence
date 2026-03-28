"""Tests for git worktree utilities."""

import subprocess
from pathlib import Path

import pytest

from recursive_intelligence.git.worktrees import (
    WorktreeError,
    branch_name,
    create_worktree,
    ensure_clean_repo,
    get_head_sha,
    list_worktrees,
    remove_worktree,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo for testing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo


class TestBranchNaming:
    def test_branch_name_format(self):
        name = branch_name("run-abc123def456", "node-xyz789012345", "implement feature X")
        assert name.startswith("ri/")
        parts = name.split("/")
        assert len(parts) == 4  # ri / run_short / node_short / task_hash

    def test_branch_name_deterministic(self):
        n1 = branch_name("run-abc", "node-xyz", "task")
        n2 = branch_name("run-abc", "node-xyz", "task")
        assert n1 == n2

    def test_branch_name_varies_with_task(self):
        n1 = branch_name("run-abc", "node-xyz", "task A")
        n2 = branch_name("run-abc", "node-xyz", "task B")
        assert n1 != n2


class TestWorktreeLifecycle:
    def test_create_and_remove(self, git_repo, tmp_path):
        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()
        wt_path = create_worktree(git_repo, wt_dir, "node-test", "ri/test/branch")
        assert wt_path.exists()
        assert (wt_path / "README.md").exists()

        remove_worktree(git_repo, wt_path)
        assert not wt_path.exists()

    def test_create_idempotent(self, git_repo, tmp_path):
        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()
        p1 = create_worktree(git_repo, wt_dir, "node-test", "ri/test/branch")
        p2 = create_worktree(git_repo, wt_dir, "node-test", "ri/test/branch")
        assert p1 == p2

    def test_list_worktrees(self, git_repo, tmp_path):
        wt_dir = tmp_path / "worktrees"
        wt_dir.mkdir()
        create_worktree(git_repo, wt_dir, "node-a", "ri/test/a")
        wts = list_worktrees(git_repo)
        # At least the main worktree + our new one
        assert len(wts) >= 2


class TestRepoChecks:
    def test_clean_repo_passes(self, git_repo):
        ensure_clean_repo(git_repo)  # should not raise

    def test_dirty_repo_raises(self, git_repo):
        # Modify a tracked file (untracked files are allowed)
        (git_repo / "README.md").write_text("modified")
        with pytest.raises(WorktreeError, match="uncommitted"):
            ensure_clean_repo(git_repo)

    def test_get_head_sha(self, git_repo):
        sha = get_head_sha(git_repo)
        assert len(sha) == 40
