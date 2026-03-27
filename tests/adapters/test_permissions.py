"""Tests for mode-based permission config."""

import pytest

from recursive_intelligence.adapters.claude.permissions import get_mode_config


class TestModeConfig:
    def test_plan_mode_is_read_only(self):
        config = get_mode_config("plan")
        assert "Read" in config.allowed_tools
        assert "Edit" not in config.allowed_tools
        assert "Write" not in config.allowed_tools

    def test_execute_mode_has_edit(self):
        config = get_mode_config("execute")
        assert "Edit" in config.allowed_tools
        assert "Write" in config.allowed_tools
        assert "Bash" in config.allowed_tools
        assert config.permission_mode == "acceptEdits"

    def test_review_mode_is_read_only(self):
        config = get_mode_config("review")
        assert "Read" in config.allowed_tools
        assert "Edit" not in config.allowed_tools

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_mode_config("invalid")
