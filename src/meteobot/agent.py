"""The agent loop: one streaming code path that orchestrates a Turn.

All dependencies arrive by injection (LLM client, tool registry, settings);
the loop consumes only the registry's schema list and dispatch function, and
stays presentation-blind by forwarding text deltas to a renderer callback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from openai.types.responses import ResponseFunctionToolCall

from meteobot.config import Settings
from meteobot.llm import History, LLMClient
from meteobot.tools.registry import ToolRegistry
from meteobot.tools.results import ToolError, ToolResult

# One line per seam at INFO, keyed by nothing (single process, single
# conversation); configured to stderr at the composition root. See
# design-decisions.md #14.
logger = logging.getLogger(__name__)

ROUND_CAP_MESSAGE = (
    "I've hit my limit of weather lookups for this question, so I have to "
    "stop here. Try again with fewer places or a more specific question."
)

INSTRUCTIONS = (
    "You are meteobot, a concise CLI weather assistant. Answer questions "
    "about current weather using the get_weather tool, calling it once per "
    "city. Answer plainly in a sentence or two per city."
)


@dataclass(frozen=True)
class Agent:
    """Runs Turns against injected LLM, tools, and settings."""

    llm: LLMClient
    registry: ToolRegistry
    settings: Settings

    async def run_turn(
        self, user_input: str, history: History, render: Callable[[str], None]
    ) -> None:
        """Run one Turn: stream the answer, executing Tool Rounds as needed.

        Appends everything the Turn produces to the client-owned `history`.
        """
        logger.info("turn start: %r", user_input)
        history.append({"role": "user", "content": user_input})

        rounds_used = 0
        while True:
            text_parts: list[str] = []
            tool_calls: list[ResponseFunctionToolCall] = []
            async for event in self.llm.stream(
                model=self.settings.model,
                instructions=INSTRUCTIONS,
                tools=self.registry.schemas(),
                input=history,
            ):
                if event.type == "response.output_text.delta":
                    render(event.delta)
                    text_parts.append(event.delta)
                elif (
                    event.type == "response.output_item.done"
                    and event.item.type == "function_call"
                ):
                    tool_calls.append(event.item)

            if text_parts:
                history.append({"role": "assistant", "content": "".join(text_parts)})
            if not tool_calls:
                return

            # Cap check happens before the round runs, so History never holds
            # a function_call without its output.
            if rounds_used >= self.settings.tool_round_cap:
                logger.warning(
                    "tool round cap reached (%d); ending turn",
                    self.settings.tool_round_cap,
                )
                render(ROUND_CAP_MESSAGE)
                history.append({"role": "assistant", "content": ROUND_CAP_MESSAGE})
                return

            rounds_used += 1
            logger.info(
                "tool round %d: %d call(s) %s",
                rounds_used,
                len(tool_calls),
                [call.name for call in tool_calls],
            )
            await self._run_tool_round(tool_calls, history)

    async def _run_tool_round(
        self, tool_calls: list[ResponseFunctionToolCall], history: History
    ) -> None:
        """Execute one Tool Round: dispatch all Tool Calls concurrently and
        append each call and its result to History."""
        for call in tool_calls:
            history.append(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
        results = await asyncio.gather(
            *(self._dispatch(call) for call in tool_calls)
        )
        for call, result in zip(tool_calls, results, strict=True):
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

    async def _dispatch(self, call: ResponseFunctionToolCall) -> ToolResult:
        """Run one Tool Call, capturing any escaped exception as a Tool Error
        so one failing call never takes down its siblings or the Turn."""
        start = time.perf_counter()
        try:
            arguments: dict[str, object] = json.loads(call.arguments)
            result: ToolResult = await self.registry.dispatch(call.name, arguments)
        except Exception:
            result = ToolError(
                error="tool_execution_error",
                message=f"The {call.name} tool failed unexpectedly for this call.",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        status = f"error:{result['error']}" if "error" in result else "ok"
        logger.info(
            "tool call %s(%s) -> %s in %.1f ms",
            call.name,
            call.arguments,
            status,
            elapsed_ms,
        )
        return result
