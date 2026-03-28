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
    ri_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    worktrees_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.ri_dir = self.repo_root / ".ri"
        self.db_path = self.ri_dir / "state.db"
        self.worktrees_dir = self.ri_dir / "worktrees"
        self.artifacts_dir = self.ri_dir / "runs"

    def ensure_dirs(self) -> None:
        self.ri_dir.mkdir(exist_ok=True)
        self.worktrees_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

    @classmethod
    def from_cwd(cls) -> RuntimeConfig:
        return cls(repo_root=Path.cwd())
