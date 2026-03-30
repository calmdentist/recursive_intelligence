"""Runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeConfig:
    """Configuration for a recursive intelligence run."""

    repo_root: Path
    max_parallel_children: int = 5  # 0 = unlimited
    max_verify_retries: int = 2
    ri_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    worktrees_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    benchmarks_dir: Path = field(init=False)
    datasets_dir: Path = field(init=False)
    tools_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.ri_dir = self.repo_root / ".ri"
        self.db_path = self.ri_dir / "state.db"
        self.worktrees_dir = self.ri_dir / "worktrees"
        self.artifacts_dir = self.ri_dir / "runs"
        self.benchmarks_dir = self.ri_dir / "benchmarks"
        self.datasets_dir = self.ri_dir / "datasets"
        self.tools_dir = self.ri_dir / "tools"

    def ensure_dirs(self) -> None:
        self.ri_dir.mkdir(exist_ok=True)
        self.worktrees_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.benchmarks_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)
        self.tools_dir.mkdir(exist_ok=True)

    @classmethod
    def from_cwd(cls) -> RuntimeConfig:
        return cls(repo_root=Path.cwd())
