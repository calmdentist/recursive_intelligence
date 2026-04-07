"""Diff and file-change utilities for artifact bundles."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
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


def find_scope_violations(changed_files: list[str], allowed_patterns: list[str] | None) -> list[str]:
    """Return changed files that do not match the declared ownership patterns."""
    if not allowed_patterns:
        return []
    normalized_patterns = [_normalize_pattern(pattern) for pattern in allowed_patterns if pattern]
    violations: list[str] = []
    for changed_file in changed_files:
        normalized_file = changed_file.lstrip("./")
        if not any(_matches_pattern(normalized_file, pattern) for pattern in normalized_patterns):
            violations.append(normalized_file)
    return violations


def _normalize_pattern(pattern: str) -> str:
    return pattern.lstrip("./")


def _matches_pattern(path: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(f"{prefix}/")
    return fnmatch(path, pattern)
