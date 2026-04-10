"""Tests for planner and system prompt policy guidance."""

from recursive_intelligence.adapters.claude.prompts import (
    ROOT_SYSTEM_CONTRACT,
    SYSTEM_CONTRACT,
    planning_prompt,
    routing_prompt,
)


def test_system_contract_emphasizes_context_aware_delegation() -> None:
    assert "Prefer direct execution" in SYSTEM_CONTRACT
    assert "mostly disjoint domains" in SYSTEM_CONTRACT
    assert "must not depend on sibling output" in SYSTEM_CONTRACT
    assert "Stop recursing" in SYSTEM_CONTRACT


def test_root_system_contract_keeps_parallelism_coarse_and_human_facing() -> None:
    assert "conversational and plain-English" in ROOT_SYSTEM_CONTRACT
    assert "small number of meaningful children" in ROOT_SYSTEM_CONTRACT


def test_planning_prompt_includes_decomposition_policy() -> None:
    prompt = planning_prompt(
        "build a product",
        snapshot_summary=["HEAD: abc12345"],
        ownership_context=[],
    )

    assert "## Decomposition Policy" in prompt
    assert "Solve directly if you can comfortably understand and complete the work" in prompt
    assert "Prefer a small number of meaningful children" in prompt
    assert "Children should own outcomes or domains" in prompt
    assert "Base the decision on your actual context needs" in prompt


def test_routing_prompt_limits_over_parallelization() -> None:
    prompt = routing_prompt(
        "extend auth flow",
        domains=[{
            "domain_name": "auth",
            "child_node_id": "node-auth",
            "availability": "merged into current snapshot",
            "file_patterns": ["auth.py"],
            "child_state": "completed",
            "last_summary": "built auth",
        }],
        pass_number=2,
        snapshot_summary=["HEAD: abc12345"],
        ownership_context=[],
        merged_work=[],
    )

    assert "route it back to that same child" in prompt
    assert "Do not spawn a second child" in prompt
    assert "Route at most one task to each child in a single wave" in prompt
    assert "Prefer the minimum number of child tasks needed" in prompt
