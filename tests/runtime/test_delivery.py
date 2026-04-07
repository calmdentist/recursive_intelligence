"""Tests for the delivery control plane."""

from __future__ import annotations

import json

import pytest

from recursive_intelligence.config import RuntimeConfig
from recursive_intelligence.runtime.delivery import DeliveryController
from recursive_intelligence.runtime.state_store import StateStore


@pytest.fixture
def config(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = RuntimeConfig(repo_root=repo)
    cfg.ensure_dirs()
    return cfg


def test_delivery_controller_persists_snapshot_and_events(config):
    store = StateStore(config.db_path)
    run = store.create_run(str(config.repo_root), "ship it")
    root = store.create_node(run.run_id, "root task")
    store.set_root_node(run.run_id, root.node_id)
    store.close()

    controller = DeliveryController(config)
    preview = controller.record_preview(run.run_id, "http://localhost:3000", label="local")
    deployment = controller.record_deployment(
        run.run_id,
        "production",
        "https://app.example.com",
        verification_status="pending",
    )
    release = controller.set_release_status(run.run_id, "blocked", note="Awaiting canary")
    blocker = controller.add_blocker(
        run.run_id,
        kind="canary",
        summary="Canary checks not started",
        action_requested="Run post-deploy verification.",
    )

    overview = controller.get_overview(run.run_id)
    assert overview["previews"][0]["preview_id"] == preview["preview_id"]
    assert overview["deployments"][0]["deployment_id"] == deployment["deployment_id"]
    assert overview["release"]["status"] == release["status"] == "blocked"
    assert overview["blockers"][0]["blocker_id"] == blocker["blocker_id"]

    delivery_path = config.artifacts_dir / run.run_id / "delivery.json"
    assert delivery_path.exists()
    snapshot = json.loads(delivery_path.read_text())
    assert snapshot["previews"][0]["url"] == "http://localhost:3000"
    assert snapshot["deployments"][0]["environment"] == "production"

    store = StateStore(config.db_path)
    events = store.get_node_events(root.node_id)
    store.close()
    assert {event.event_type for event in events} >= {
        "preview_recorded",
        "deployment_recorded",
        "release_status_updated",
        "delivery_blocker_added",
    }

    resolved = controller.resolve_blocker(run.run_id, blocker["blocker_id"], note="Canary passed")
    assert resolved["resolved_at"] is not None
    overview = controller.get_overview(run.run_id)
    assert overview["blockers"] == []
