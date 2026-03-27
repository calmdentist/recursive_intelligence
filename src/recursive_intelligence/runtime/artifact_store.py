"""Filesystem artifact storage under .ri/runs/<run-id>/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Write and read run artifacts to disk."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir

    def run_dir(self, run_id: str) -> Path:
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def node_dir(self, run_id: str, node_id: str) -> Path:
        d = self.run_dir(run_id) / node_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_transcript(self, run_id: str, node_id: str, session_id: str, transcript: Any) -> Path:
        p = self.node_dir(run_id, node_id) / f"transcript-{session_id}.json"
        p.write_text(json.dumps(transcript, indent=2))
        return p

    def save_diff(self, run_id: str, node_id: str, diff: str) -> Path:
        p = self.node_dir(run_id, node_id) / "diff.patch"
        p.write_text(diff)
        return p

    def save_report(self, run_id: str, report: dict[str, Any]) -> Path:
        p = self.run_dir(run_id) / "report.json"
        p.write_text(json.dumps(report, indent=2))
        return p

    def load_report(self, run_id: str) -> dict[str, Any] | None:
        p = self.run_dir(run_id) / "report.json"
        if p.exists():
            return json.loads(p.read_text())
        return None
