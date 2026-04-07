"""Parse structured JSON decisions from Claude agent responses."""

from __future__ import annotations

import json
import re
from typing import Any


class ParseError(Exception):
    """Failed to extract structured JSON from agent response."""


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a response string.

    Handles:
    - Raw JSON responses
    - JSON inside ```json fences
    - JSON embedded in surrounding text
    """
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try fenced code blocks
    fenced = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    for block in fenced:
        block = block.strip()
        if block.startswith("{"):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Try to find a JSON object anywhere in the text
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = None

    raise ParseError(f"No valid JSON object found in response: {text[:200]}...")


def parse_plan_decision(text: str) -> dict[str, Any]:
    """Parse a plan decision from agent output."""
    data = extract_json(text)
    if "action" not in data:
        raise ParseError(f"Plan decision missing 'action' field: {data}")
    valid_actions = {
        "solve_directly",
        "spawn_children",
        "route_to_children",
        "review_children",
        "integrate_and_finish",
        "pause",
        "done",
    }
    if data["action"] not in valid_actions:
        raise ParseError(f"Unknown plan action: {data['action']}. Expected one of: {valid_actions}")
    return data


def parse_execution_result(text: str) -> dict[str, Any]:
    """Parse an execution result from agent output."""
    data = extract_json(text)
    if "status" not in data:
        raise ParseError(f"Execution result missing 'status' field: {data}")
    return data


def parse_review_verdict(text: str) -> dict[str, Any]:
    """Parse a review verdict from agent output."""
    data = extract_json(text)
    if "verdict" not in data:
        raise ParseError(f"Review verdict missing 'verdict' field: {data}")
    valid_verdicts = {"accept", "revise", "reject"}
    if data["verdict"] not in valid_verdicts:
        raise ParseError(f"Unknown verdict: {data['verdict']}. Expected one of: {valid_verdicts}")
    return data
