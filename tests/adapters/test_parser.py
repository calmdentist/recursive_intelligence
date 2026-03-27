"""Tests for the JSON response parser."""

import pytest

from recursive_intelligence.adapters.claude.parser import (
    ParseError,
    extract_json,
    parse_execution_result,
    parse_plan_decision,
    parse_review_verdict,
)


class TestExtractJson:
    def test_raw_json(self):
        result = extract_json('{"action": "solve_directly"}')
        assert result["action"] == "solve_directly"

    def test_json_with_whitespace(self):
        result = extract_json('  \n{"action": "solve_directly"}  \n')
        assert result["action"] == "solve_directly"

    def test_fenced_json(self):
        text = 'Here is the plan:\n```json\n{"action": "solve_directly"}\n```'
        result = extract_json(text)
        assert result["action"] == "solve_directly"

    def test_fenced_no_lang(self):
        text = '```\n{"action": "solve_directly"}\n```'
        result = extract_json(text)
        assert result["action"] == "solve_directly"

    def test_embedded_json(self):
        text = "I've analyzed the repo and here's my plan: {\"action\": \"solve_directly\", \"rationale\": \"small task\"} Let me know."
        result = extract_json(text)
        assert result["action"] == "solve_directly"

    def test_nested_json(self):
        text = '{"action": "spawn_children", "children": [{"key": "a"}]}'
        result = extract_json(text)
        assert result["action"] == "spawn_children"
        assert len(result["children"]) == 1

    def test_no_json_raises(self):
        with pytest.raises(ParseError):
            extract_json("no json here")

    def test_invalid_json_raises(self):
        with pytest.raises(ParseError):
            extract_json("{invalid json content")


class TestParsePlanDecision:
    def test_solve_directly(self):
        text = '{"action": "solve_directly", "rationale": "Small fix"}'
        result = parse_plan_decision(text)
        assert result["action"] == "solve_directly"

    def test_spawn_children(self):
        text = '{"action": "spawn_children", "children": [{"idempotency_key": "a", "objective": "fix bug"}]}'
        result = parse_plan_decision(text)
        assert result["action"] == "spawn_children"

    def test_missing_action_raises(self):
        with pytest.raises(ParseError, match="missing 'action'"):
            parse_plan_decision('{"rationale": "no action"}')

    def test_unknown_action_raises(self):
        with pytest.raises(ParseError, match="Unknown plan action"):
            parse_plan_decision('{"action": "do_nothing"}')


class TestParseExecutionResult:
    def test_implemented(self):
        text = '{"status": "implemented", "summary": "Done", "commit_sha": "abc123"}'
        result = parse_execution_result(text)
        assert result["status"] == "implemented"

    def test_blocked(self):
        text = '{"status": "blocked", "kind": "missing_dep", "details": "need lib"}'
        result = parse_execution_result(text)
        assert result["status"] == "blocked"

    def test_missing_status_raises(self):
        with pytest.raises(ParseError, match="missing 'status'"):
            parse_execution_result('{"summary": "no status"}')


class TestParseReviewVerdict:
    def test_accept(self):
        text = '{"verdict": "accept", "child_id": "node-abc", "reason": "looks good"}'
        result = parse_review_verdict(text)
        assert result["verdict"] == "accept"

    def test_revise(self):
        text = '{"verdict": "revise", "child_id": "node-abc", "follow_up": "add tests"}'
        result = parse_review_verdict(text)
        assert result["verdict"] == "revise"

    def test_missing_verdict_raises(self):
        with pytest.raises(ParseError, match="missing 'verdict'"):
            parse_review_verdict('{"child_id": "node-abc"}')

    def test_unknown_verdict_raises(self):
        with pytest.raises(ParseError, match="Unknown verdict"):
            parse_review_verdict('{"verdict": "maybe"}')
