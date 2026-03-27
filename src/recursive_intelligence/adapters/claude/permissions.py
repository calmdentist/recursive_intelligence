"""Mode-based tool and permission configuration for Claude sessions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModeConfig:
    """Tool and permission config for a given node mode."""

    allowed_tools: list[str]
    permission_mode: str


# Plan mode: read-only exploration
PLAN_MODE = ModeConfig(
    allowed_tools=["Read", "Glob", "Grep", "Bash"],
    permission_mode="default",
)

# Execute mode: full edit + shell access
EXECUTE_MODE = ModeConfig(
    allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    permission_mode="acceptEdits",
)

# Review mode: read-only, inspecting child diffs
REVIEW_MODE = ModeConfig(
    allowed_tools=["Read", "Glob", "Grep"],
    permission_mode="default",
)


def get_mode_config(mode: str) -> ModeConfig:
    """Get tool/permission config for a node mode."""
    configs = {
        "plan": PLAN_MODE,
        "execute": EXECUTE_MODE,
        "review": REVIEW_MODE,
    }
    config = configs.get(mode)
    if config is None:
        raise ValueError(f"Unknown mode: {mode!r}. Expected one of: {list(configs)}")
    return config
