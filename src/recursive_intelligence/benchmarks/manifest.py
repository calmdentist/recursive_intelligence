"""Frozen benchmark manifest support."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from recursive_intelligence.benchmarks.models import BenchmarkTask
from recursive_intelligence.benchmarks.swebench import (
    DEFAULT_FEATUREBENCH_DATASET,
    DEFAULT_FEATUREBENCH_SPLIT,
    DEFAULT_SWEBENCH_PRO_DATASET,
    DEFAULT_SWEBENCH_PRO_SPLIT,
    build_loader,
    score_recursive_hard_task,
    select_recursive_hard,
)

DEFAULT_MANIFEST_ID = "recursive-hard-v1"
DEFAULT_MANIFEST_TITLE = "Recursive Hard v1"
DEFAULT_RUBRIC_VERSION = "recursive-hard-proxy-v1"


@dataclass
class ManifestTaskEntry:
    """Task reference in a frozen benchmark manifest."""

    benchmark: str
    dataset_name: str
    split: str
    instance_id: str
    repo: str | None = None
    selection_scores: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkManifest:
    """Versioned benchmark manifest."""

    manifest_id: str
    title: str
    description: str = ""
    rubric_version: str = DEFAULT_RUBRIC_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    tasks: list[ManifestTaskEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "title": self.title,
            "description": self.description,
            "rubric_version": self.rubric_version,
            "metadata": self.metadata,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def load_benchmark_manifest(path: Path) -> BenchmarkManifest:
    payload = json.loads(path.read_text())
    tasks = [
        ManifestTaskEntry(
            benchmark=entry["benchmark"],
            dataset_name=entry.get("dataset_name") or entry.get("dataset"),
            split=entry["split"],
            instance_id=entry["instance_id"],
            repo=entry.get("repo"),
            selection_scores=dict(entry.get("selection_scores", {})),
            notes=entry.get("notes"),
        )
        for entry in payload.get("tasks", [])
    ]
    return BenchmarkManifest(
        manifest_id=payload["manifest_id"],
        title=payload.get("title", payload["manifest_id"]),
        description=payload.get("description", ""),
        rubric_version=payload.get("rubric_version", DEFAULT_RUBRIC_VERSION),
        metadata=dict(payload.get("metadata", {})),
        tasks=tasks,
    )


def write_benchmark_manifest(manifest: BenchmarkManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return path


def resolve_manifest_tasks(
    manifest: BenchmarkManifest,
    cache_dir: Path,
    refresh: bool = False,
) -> list[BenchmarkTask]:
    """Materialize manifest task references into normalized benchmark tasks."""
    grouped_entries: dict[tuple[str, str, str], list[ManifestTaskEntry]] = {}
    for entry in manifest.tasks:
        key = (entry.benchmark, entry.dataset_name, entry.split)
        grouped_entries.setdefault(key, []).append(entry)

    loaded: dict[tuple[str, str, str], dict[str, BenchmarkTask]] = {}
    for key, entries in grouped_entries.items():
        benchmark, dataset_name, split = key
        loader = build_loader(benchmark, cache_dir, dataset=dataset_name, split=split)
        tasks = loader.load_tasks(refresh=refresh)
        loaded[key] = {task.instance_id: task for task in tasks}
        missing = [entry.instance_id for entry in entries if entry.instance_id not in loaded[key]]
        if missing:
            raise ValueError(
                f"Manifest references tasks missing from {benchmark} {dataset_name}/{split}: {', '.join(missing)}"
            )

    resolved: list[BenchmarkTask] = []
    for entry in manifest.tasks:
        key = (entry.benchmark, entry.dataset_name, entry.split)
        task = loaded[key][entry.instance_id]
        task.manifest_id = manifest.manifest_id
        if entry.selection_scores:
            task.selection_scores = dict(entry.selection_scores)
        if entry.notes:
            task.extra["manifest_notes"] = entry.notes
        resolved.append(task)
    return resolved


def build_recursive_hard_manifest(
    cache_dir: Path,
    output_path: Path | None = None,
    manifest_id: str = DEFAULT_MANIFEST_ID,
    title: str = DEFAULT_MANIFEST_TITLE,
    swebench_pro_count: int = 20,
    featurebench_count: int = 10,
    repo_cap: int = 2,
    refresh: bool = False,
) -> BenchmarkManifest:
    """Build a deterministic recursive-hard manifest from SWE-Bench Pro and FeatureBench."""
    swebench_pro_tasks = build_loader(
        "swebench_pro",
        cache_dir,
        dataset=DEFAULT_SWEBENCH_PRO_DATASET,
        split=DEFAULT_SWEBENCH_PRO_SPLIT,
    ).load_tasks(refresh=refresh)
    featurebench_tasks = build_loader(
        "featurebench",
        cache_dir,
        dataset=DEFAULT_FEATUREBENCH_DATASET,
        split=DEFAULT_FEATUREBENCH_SPLIT,
    ).load_tasks(refresh=refresh)

    selected = [
        *select_recursive_hard(swebench_pro_tasks, target_size=swebench_pro_count, repo_cap=repo_cap),
        *select_recursive_hard(featurebench_tasks, target_size=featurebench_count, repo_cap=repo_cap),
    ]

    manifest = BenchmarkManifest(
        manifest_id=manifest_id,
        title=title,
        description=(
            "Proxy-scored hard long-context suite spanning SWE-Bench Pro and FeatureBench. "
            "This first-pass manifest is deterministic and versioned so flat-vs-recursive comparisons "
            "can be repeated exactly."
        ),
        rubric_version=DEFAULT_RUBRIC_VERSION,
        metadata={
            "source_counts": {
                "swebench_pro": swebench_pro_count,
                "featurebench": featurebench_count,
            },
            "repo_cap": repo_cap,
            "sources": [
                f"swebench_pro:{DEFAULT_SWEBENCH_PRO_DATASET}:{DEFAULT_SWEBENCH_PRO_SPLIT}",
                f"featurebench:{DEFAULT_FEATUREBENCH_DATASET}:{DEFAULT_FEATUREBENCH_SPLIT}",
            ],
        },
        tasks=[
            ManifestTaskEntry(
                benchmark=task.benchmark,
                dataset_name=task.dataset_name or "",
                split=task.dataset_split or "",
                instance_id=task.instance_id,
                repo=task.repo,
                selection_scores=dict(task.selection_scores or score_recursive_hard_task(task)),
            )
            for task in selected
        ],
    )
    if output_path is not None:
        write_benchmark_manifest(manifest, output_path)
    return manifest
