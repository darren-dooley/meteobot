"""Agent construction and the Turn adapter.

v1 hand-wrote the model -> Tool Call -> model loop; v2 hands that surface to a
PydanticAI `Agent`, which owns streaming, concurrent tool dispatch, tool-output
validation, and the usage limit. Two things live here:

- `build_agent` — assembles the `Agent` from an injected model, the tool
  functions, and optional instrumentation capabilities. Tests inject a
  `TestModel`/`FunctionModel` and stub tools where production injects the real
  `OpenAIResponsesModel` and `get_weather`; the composition root is the only
  place tools are registered.
- `make_run_turn` — the thin adapter the REPL drives. It runs one streaming
  Turn, forwards text deltas to the renderer, carries the History forward as
  PydanticAI's own message list, and translates a tripped usage limit into the
  same honest, user-facing message v1's round cap produced. It stays
  presentation-blind: it renders text, nothing else.

Only two exception classes matter at this seam. `UsageLimitExceeded` is the
guardrail firing and is handled here as an in-Turn message. Anything else that
escapes the run is an Infrastructure Error (LLM API down, bad credentials,
transport failure) and is left to propagate to the REPL, which turns it into
one friendly line and survives.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Sequence

from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from meteobot.config import Settings
from meteobot.deps import Deps

# History is PydanticAI's own client-owned message list: the messages from one
# Turn are appended and carried into the next as `message_history`, so
# multi-turn follow-ups fall out for free and the record stays inspectable.
type History = list[ModelMessage]

# One Turn step: the user's text, the client-owned History (mutated in place),
# and a per-delta render callback.
type RunTurn = Callable[[str, History, Callable[[str], None]], Awaitable[None]]

AGENT_NAME = "meteobot"

INSTRUCTIONS = (
    "You are meteobot, a concise CLI weather assistant. Answer questions "
    "about current weather using the get_weather tool, calling it once per "
    "city. Answer plainly in a sentence or two per city."
)

# Rendered when a usage limit trips — either the request limit (the framework's
# Tool Round cap) or the token limit. Worded to be honest for both: the Turn did
# too much work to finish, whichever guardrail fired.
ROUND_CAP_MESSAGE = (
    "I've hit my limit for this question, so I have to stop here. Try again "
    "with fewer places or a more specific question."
)


def build_openai_model(
    settings: Settings, openai_client: AsyncOpenAI
) -> OpenAIResponsesModel:
    """The production model seam: OpenAI's Responses API over the shared client.

    Kept here so the composition root and the live e2e test build the model the
    same way; offline tests inject a `TestModel`/`FunctionModel` in its place.
    """
    return OpenAIResponsesModel(
        settings.model, provider=OpenAIProvider(openai_client=openai_client)
    )


def build_agent(
    model: Model,
    tools: Sequence[Callable[..., object]],
    *,
    capabilities: Sequence[AgentCapability[Deps]] = (),
) -> Agent[Deps]:
    """Assemble the Turn-running agent from injected parts.

    `tools` are typed async functions taking `RunContext[Deps]`; PydanticAI
    derives each tool's JSON schema from its type hints. `capabilities` carries
    optional instrumentation. This is the single composition point: adding a
    tool means writing one typed function and passing it here.
    """
    return Agent(
        model=model,
        deps_type=Deps,
        name=AGENT_NAME,
        instructions=INSTRUCTIONS,
        tools=list(tools),
        capabilities=list(capabilities),
    )


def make_run_turn(
    agent: Agent[Deps], deps: Deps, usage_limits: UsageLimits
) -> RunTurn:
    """Build the REPL's Turn adapter around a constructed agent and its deps."""

    async def run_turn(
        user_input: str, history: History, render: Callable[[str], None]
    ) -> None:
        """Run one streaming Turn, appending its messages to `history`.

        A tripped usage limit ends the Turn with an honest message rather than
        raising past the REPL; every other failure propagates as an
        Infrastructure Error for the REPL to catch.
        """
        try:
            async with agent.run_stream(
                user_input,
                message_history=list(history),
                deps=deps,
                usage_limits=usage_limits,
            ) as result:
                async for delta in result.stream_text(delta=True):
                    render(delta)
                new_messages = result.new_messages()
            history.extend(new_messages)
        except UsageLimitExceeded:
            # A capped Turn is abandoned, not recorded: the honest message is
            # rendered to the user, and this incomplete Turn's messages are
            # deliberately not carried into History so the next Turn starts clean.
            render(ROUND_CAP_MESSAGE)

    return run_turn


def usage_limits_from(settings: Settings) -> UsageLimits:
    """The Turn's guardrails as a declared PydanticAI policy.

    A request limit caps the model -> Tool Call -> model Tool Rounds (v1's round
    cap, now framework-enforced); a token limit bounds worst-case cost even
    within that request budget.
    """
    return UsageLimits(
        request_limit=settings.request_limit,
        total_tokens_limit=settings.total_tokens_limit,
    )
