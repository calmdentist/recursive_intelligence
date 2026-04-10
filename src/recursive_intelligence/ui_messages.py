"""Helpers for separating human-facing UI text from control-plane output."""

from __future__ import annotations

import json
import re


CONTROL_PLANE_KEYS = {
    "action",
    "status",
    "verdict",
    "children",
    "routes",
    "child_id",
    "commit_sha",
    "result_commit_sha",
    "follow_up",
}


FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _is_control_plane_payload(payload: object) -> bool:
    return isinstance(payload, dict) and bool(CONTROL_PLANE_KEYS & set(payload))


def _find_control_plane_span(text: str) -> tuple[int, int] | None:
    for match in FENCED_BLOCK_RE.finditer(text):
        block = match.group(1).strip()
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if _is_control_plane_payload(payload):
            return match.span()

    brace_depth = 0
    start: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if brace_depth == 0:
                start = index
            brace_depth += 1
        elif char == "}" and brace_depth > 0:
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                candidate = text[start : index + 1]
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if _is_control_plane_payload(payload):
                    return (start, index + 1)
                start = None
    return None


def looks_like_control_plane_json(text: str) -> bool:
    """Return True when a text block is only a runtime JSON envelope."""
    span = _find_control_plane_span(text)
    if span is None:
        return False
    start, end = span
    return not f"{text[:start]}{text[end:]}".strip()


def human_visible_text(text: str) -> str:
    """Strip machine-readable control-plane envelopes from a text block."""
    cleaned = text
    while True:
        span = _find_control_plane_span(cleaned)
        if span is None:
            break
        start, end = span
        cleaned = f"{cleaned[:start]}{cleaned[end:]}"
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
