"""Run-level delivery control plane helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.artifact_store import ArtifactStore
from recursive_intelligence.runtime.state_store import StateStore


class DeliveryController:
    """Persist and project previews, deployments, release status, and blockers."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.config.ensure_dirs()
        self._artifacts = ArtifactStore(self.config.artifacts_dir)

    def get_overview(self, run_id: str) -> dict[str, Any]:
        store = StateStore(self.config.db_path)
        run = store.get_run(run_id)
        if run is None:
            store.close()
            raise ValueError(f"Run {run_id} not found")

        readiness = store.get_run_readiness(run_id)
        delivery = store.get_run_delivery(run_id)
        store.close()

        return {
            "run": run,
            "readiness": readiness,
            "delivery": delivery,
            "release": dict(delivery.get("release", {})),
            "previews": list(delivery.get("previews", [])),
            "deployments": list(delivery.get("deployments", [])),
            "blockers": [
                blocker for blocker in delivery.get("blockers", [])
                if not blocker.get("resolved_at")
            ],
            "artifacts": {
                "run_dir": str(self._artifacts.run_dir(run_id)),
                "report_path": str(self._artifacts.run_dir(run_id) / "report.json"),
                "delivery_path": str(self._artifacts.run_dir(run_id) / "delivery.json"),
            },
        }

    def record_preview(
        self,
        run_id: str,
        url: str,
        *,
        label: str = "preview",
        status: str = "ready",
        note: str = "",
    ) -> dict[str, Any]:
        store = StateStore(self.config.db_path)
        preview = store.record_preview(run_id, url, label=label, status=status, note=note)
        self._append_root_event(store, run_id, "preview_recorded", preview)
        store.close()
        self._sync_delivery_snapshot(run_id)
        return preview

    def record_deployment(
        self,
        run_id: str,
        environment: str,
        url: str,
        *,
        status: str = "deployed",
        note: str = "",
        verification_status: str = "unknown",
    ) -> dict[str, Any]:
        store = StateStore(self.config.db_path)
        deployment = store.record_deployment(
            run_id,
            environment,
            url,
            status=status,
            note=note,
            verification_status=verification_status,
        )
        self._append_root_event(store, run_id, "deployment_recorded", deployment)
        store.close()
        self._sync_delivery_snapshot(run_id)
        return deployment

    def set_release_status(self, run_id: str, status: str, note: str = "") -> dict[str, Any]:
        store = StateStore(self.config.db_path)
        release = store.set_release_status(run_id, status, note=note)
        self._append_root_event(store, run_id, "release_status_updated", release)
        store.close()
        self._sync_delivery_snapshot(run_id)
        return release

    def add_blocker(
        self,
        run_id: str,
        *,
        kind: str,
        summary: str,
        details: str = "",
        action_requested: str = "",
    ) -> dict[str, Any]:
        store = StateStore(self.config.db_path)
        blocker = store.add_delivery_blocker(
            run_id,
            kind=kind,
            summary=summary,
            details=details,
            action_requested=action_requested,
        )
        self._append_root_event(store, run_id, "delivery_blocker_added", blocker)
        store.close()
        self._sync_delivery_snapshot(run_id)
        return blocker

    def resolve_blocker(self, run_id: str, blocker_id: str, note: str = "") -> dict[str, Any]:
        store = StateStore(self.config.db_path)
        blocker = store.resolve_delivery_blocker(run_id, blocker_id, note=note)
        self._append_root_event(store, run_id, "delivery_blocker_resolved", blocker)
        store.close()
        self._sync_delivery_snapshot(run_id)
        return blocker

    def _sync_delivery_snapshot(self, run_id: str) -> None:
        store = StateStore(self.config.db_path)
        delivery = store.get_run_delivery(run_id)
        store.close()
        snapshot = dict(delivery)
        snapshot["generated_at"] = _now()
        self._artifacts.save_delivery_state(run_id, snapshot)

    @staticmethod
    def _append_root_event(store: StateStore, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        run = store.get_run(run_id)
        if run is None or not run.root_node_id:
            return
        store.append_event(run_id, run.root_node_id, event_type, payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
