"""Unit tests for UI-facing message cleanup helpers."""

from recursive_intelligence.ui_messages import human_visible_text, looks_like_control_plane_json


def test_human_visible_text_strips_control_plane_json() -> None:
    text = (
        "I am splitting the work into focused tracks.\n\n"
        '{"action":"spawn_children","rationale":"parallelize",'
        '"children":[{"idempotency_key":"ui","objective":"build UI","success_criteria":["done"]}]}'
    )

    assert human_visible_text(text) == "I am splitting the work into focused tracks."
    assert not looks_like_control_plane_json(text)


def test_human_visible_text_strips_fenced_control_plane_json() -> None:
    text = (
        "Quick update before I hand off implementation.\n\n"
        "```json\n"
        '{"status":"implemented","summary":"done","changed_files":["app.py"]}\n'
        "```"
    )

    assert human_visible_text(text) == "Quick update before I hand off implementation."


def test_looks_like_control_plane_json_detects_plain_envelope() -> None:
    text = '{"action":"solve_directly","rationale":"small task"}'

    assert looks_like_control_plane_json(text) is True
    assert human_visible_text(text) == ""
