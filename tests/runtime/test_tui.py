"""Unit tests for TUI-facing message filtering and state updates."""

import pytest

from recursive_intelligence.config import RuntimeConfig
pytest.importorskip("textual")

from recursive_intelligence.tui import RariApp


def test_tui_binds_run_id_from_run_created_status(tmp_path) -> None:
    app = RariApp(config=RuntimeConfig(repo_root=tmp_path), model="test-model")

    assert app.run_id is None

    app._consume_status_event({
        "event": "run_created",
        "run_id": "run-123",
        "root_node_id": "node-root",
    })

    assert app.run_id == "run-123"
