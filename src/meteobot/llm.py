"""The LLM client seam.

The agent depends on `LLMClient`, a protocol with one method that mirrors
`responses.create(stream=True, ...)` and yields the SDK's own stream event
types. Tests supply a fake that replays scripted event sequences built from
captured wire fixtures; production wraps AsyncOpenAI in the thin adapter
below (exercised by the opt-in live e2e test).

History is the client-owned conversation state: a plain list of Responses API
input items (user/assistant messages, function_call and function_call_output
items), passed in full on every call. There is no server-side conversation
state, so History is always inspectable for debugging.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, cast

from openai import AsyncOpenAI
from openai.types.responses import ResponseStreamEvent

type HistoryItem = dict[str, object]
type History = list[HistoryItem]


class LLMClient(Protocol):
    """One streaming Turn step: full History in, SDK stream events out."""

    def stream(
        self,
        *,
        model: str,
        instructions: str,
        tools: list[dict[str, object]],
        input: History,
    ) -> AsyncIterator[ResponseStreamEvent]: ...


@dataclass(frozen=True)
class OpenAIResponsesLLM:
    """The production LLMClient: a thin wrapper over the Responses API."""

    client: AsyncOpenAI

    async def stream(
        self,
        *,
        model: str,
        instructions: str,
        tools: list[dict[str, object]],
        input: History,
    ) -> AsyncIterator[ResponseStreamEvent]:
        # The SDK's param types are TypedDict unions; our schemas and History
        # items are the same wire shapes as plain dicts.
        stream = await self.client.responses.create(
            model=model,
            instructions=instructions,
            tools=cast(Any, tools),
            input=cast(Any, input),
            stream=True,
        )
        async for event in stream:
            yield event
