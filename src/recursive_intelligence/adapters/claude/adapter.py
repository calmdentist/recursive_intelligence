"""Claude Code adapter – Agent SDK integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult
from recursive_intelligence.adapters.claude.parser import extract_json, ParseError
from recursive_intelligence.adapters.claude.permissions import get_mode_config
from recursive_intelligence.adapters.claude.prompts import SYSTEM_CONTRACT

log = logging.getLogger(__name__)


class ClaudeAdapter(AgentAdapter):
    """Claude Code adapter using the Anthropic Agent SDK."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "claude"

    async def run(
        self,
        prompt: str,
        worktree: Path,
        mode: str,
        system_prompt: str | None = None,
        resume_session_id: str | None = None,
    ) -> NodeResult:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ResultMessage,
            SystemMessage,
            query,
        )

        mode_config = get_mode_config(mode)
        sys_prompt = system_prompt or SYSTEM_CONTRACT

        # Build options — cwd is always required (the CLI process needs it)
        if resume_session_id:
            options = ClaudeAgentOptions(
                cwd=str(worktree),
                resume=resume_session_id,
                model=self._model,
            )
        else:
            options = ClaudeAgentOptions(
                cwd=str(worktree),
                allowed_tools=mode_config.allowed_tools,
                permission_mode=mode_config.permission_mode,
                system_prompt=sys_prompt,
                model=self._model,
                setting_sources=["project"],
            )

        session_id: str | None = resume_session_id
        result_text = ""
        stop_reason = ""
        total_cost_usd: float = 0.0
        usage: dict[str, Any] = {}
        num_turns: int = 0
        duration_ms: int = 0
        duration_api_ms: int = 0

        log.info(
            "Running Claude session: mode=%s, resume=%s, cwd=%s",
            mode, resume_session_id or "(new)", worktree,
        )

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id", session_id)
                log.debug("Session ID: %s", session_id)

            elif isinstance(message, ResultMessage):
                result_text = message.result or ""
                stop_reason = message.stop_reason or ""
                total_cost_usd = message.total_cost_usd or 0.0
                usage = message.usage or {}
                num_turns = message.num_turns
                duration_ms = message.duration_ms
                duration_api_ms = message.duration_api_ms
                log.info(
                    "Session complete: stop_reason=%s, cost=$%.4f, turns=%d, duration=%dms",
                    stop_reason, total_cost_usd, num_turns, duration_ms,
                )

        if session_id is None:
            session_id = "unknown"

        cost = CostRecord(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_usd=total_cost_usd,
        )

        raw: dict[str, Any] = {}
        if result_text:
            try:
                raw = extract_json(result_text)
            except ParseError:
                log.warning("Could not parse JSON from result, using raw text")
                raw = {"_raw_text": result_text}

        raw["_num_turns"] = num_turns
        raw["_duration_ms"] = duration_ms
        raw["_duration_api_ms"] = duration_api_ms

        return NodeResult(
            session_id=session_id,
            raw=raw,
            result_text=result_text,
            cost=cost,
            stop_reason=stop_reason,
        )
