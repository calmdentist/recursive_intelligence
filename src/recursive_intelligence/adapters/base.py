"""Agent adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# Callback type for streaming messages to the UI.
# Called with (message_type: str, data: dict) for each intermediate event.
# Types: "text", "tool_use", "tool_result", "thinking", "status"
StreamCallback = Callable[[str, dict[str, Any]], None] | None


@dataclass
class NodeResult:
    """Structured result from an adapter session."""

    session_id: str
    raw: dict[str, Any]
    result_text: str = ""
    cost: CostRecord | None = None
    stop_reason: str = ""


@dataclass
class CostRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    total_usd: float = 0.0


class AgentAdapter(ABC):
    """Interface for coding agent adapters (Claude, Codex, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def run(
        self,
        prompt: str,
        worktree: Path,
        mode: str,
        system_prompt: str | None = None,
        resume_session_id: str | None = None,
        on_message: StreamCallback = None,
    ) -> NodeResult:
        """Run (or resume) an agent session to completion.

        Args:
            prompt: The task or follow-up instructions.
            worktree: Path to the node's git worktree (used as cwd).
            mode: One of "plan", "execute", "review".
            system_prompt: Optional system prompt override.
            resume_session_id: If set, resume this existing session.
            on_message: Optional callback for streaming intermediate events.

        Returns:
            NodeResult with session_id, parsed output, and cost info.
        """
        ...
