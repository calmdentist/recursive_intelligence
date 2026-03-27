"""Diff and file-change utilities for artifact bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recursive_intelligence.git.merge import get_changed_files, get_diff


@dataclass
class ArtifactBundle:
    """Summary of a child node's work for parent review."""

    node_id: str
    diff: str
    changed_files: list[str]
    commit_sha: str | None
    summary: str = ""
    test_output: str = ""


def build_artifact_bundle(
    node_id: str,
    worktree: Path,
    base_ref: str = "HEAD~1",
    commit_sha: str | None = None,
    summary: str = "",
    test_output: str = "",
) -> ArtifactBundle:
    return ArtifactBundle(
        node_id=node_id,
        diff=get_diff(worktree, base_ref),
        changed_files=get_changed_files(worktree, base_ref),
        commit_sha=commit_sha,
        summary=summary,
        test_output=test_output,
    )
