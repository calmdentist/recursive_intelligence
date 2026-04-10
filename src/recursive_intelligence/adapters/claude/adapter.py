"""Claude Code adapter – Agent SDK integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from recursive_intelligence.adapters.base import AgentAdapter, CostRecord, NodeResult, StreamCallback
from recursive_intelligence.adapters.claude.parser import extract_json, ParseError
from recursive_intelligence.adapters.claude.permissions import get_mode_config
from recursive_intelligence.adapters.claude.prompts import ROOT_SYSTEM_CONTRACT, SYSTEM_CONTRACT

log = logging.getLogger(__name__)


class ClaudeAdapter(AgentAdapter):
    """Claude Code adapter using the Anthropic Agent SDK."""

    def __init__(
        self,
        model: str | None = None,
        root_model: str | None = None,
        child_model: str | None = None,
    ) -> None:
        shared_model = model or "claude-sonnet-4-6"
        self._root_model = root_model or shared_model
        self._child_model = child_model or shared_model

    @property
    def name(self) -> str:
        return "claude"

    def _model_for_node(self, is_root: bool) -> str:
        return self._root_model if is_root else self._child_model

    async def run(
        self,
        prompt: str,
        worktree: Path,
        mode: str,
        system_prompt: str | None = None,
        resume_session_id: str | None = None,
        on_message: StreamCallback = None,
        is_root: bool = False,
    ) -> NodeResult:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            query,
        )

        mode_config = get_mode_config(mode)
        sys_prompt = system_prompt or (ROOT_SYSTEM_CONTRACT if is_root else SYSTEM_CONTRACT)

        model = self._model_for_node(is_root)
        betas = ["context-1m-2025-08-07"] if is_root else []

        if resume_session_id:
            options = ClaudeAgentOptions(
                cwd=str(worktree),
                resume=resume_session_id,
                model=model,
                thinking={"type": "adaptive"},
                **({"betas": betas} if betas else {}),
            )
        else:
            options = ClaudeAgentOptions(
                cwd=str(worktree),
                allowed_tools=mode_config.allowed_tools,
                permission_mode=mode_config.permission_mode,
                system_prompt=sys_prompt,
                model=model,
                setting_sources=["project"],
                thinking={"type": "adaptive"},
                **({"betas": betas} if betas else {}),
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

        def _emit(msg_type: str, data: dict[str, Any]) -> None:
            if on_message:
                try:
                    on_message(msg_type, data)
                except Exception:
                    pass  # never let UI errors kill the session

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id", session_id)
                log.debug("Session ID: %s", session_id)

            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        _emit("text", {"text": block.text})
                    elif isinstance(block, ThinkingBlock):
                        _emit("thinking", {"text": block.thinking})
                    elif isinstance(block, ToolUseBlock):
                        _emit("tool_use", {
                            "tool": block.name,
                            "input": block.input if hasattr(block, "input") else {},
                        })
                    elif isinstance(block, ToolResultBlock):
                        content = ""
                        if hasattr(block, "content") and block.content:
                            if isinstance(block.content, str):
                                content = block.content
                            elif isinstance(block.content, list):
                                content = " ".join(
                                    str(getattr(c, "text", c)) for c in block.content
                                )
                        _emit("tool_result", {"content": content[:500]})

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
