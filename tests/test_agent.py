"""Agent-loop contract tests, driven through PydanticAI's model seam.

Where production injects an `OpenAIResponsesModel`, these tests inject a
`FunctionModel` (or `TestModel`) that replays scripted model behaviour: a
direct answer, one or more Tool Rounds, concurrent Tool Calls, a typed Tool
Error flowing back, and the usage-limit trip. Each test drives the real
`make_run_turn` adapter over a real `Agent`, so streaming, concurrent tool
dispatch, History growth, and the honest cap message are exercised as the REPL
would exercise them — only the model and the tool bodies are fakes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable

import httpx
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import UsageLimits

from meteobot.agent import ROUND_CAP_MESSAGE, History, build_agent, make_run_turn
from meteobot.config import Settings
from meteobot.deps import Deps
from meteobot.tools.weather import WeatherError, WeatherResult

# --- scripted model helpers -------------------------------------------------

StreamStep = Iterable[str | DeltaToolCalls]
StreamFn = Callable[[list[ModelMessage], AgentInfo], AsyncIterator[str | DeltaToolCalls]]


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(
        getattr(p, "part_kind", "") == "tool-return"
        for m in messages
        for p in getattr(m, "parts", [])
    )


def tool_calls(*calls: tuple[str, str, str]) -> DeltaToolCalls:
    """Build a one-round DeltaToolCalls from (call_id, name, json_args) triples."""
    return {
        index: DeltaToolCall(name=name, json_args=args, tool_call_id=call_id)
        for index, (call_id, name, args) in enumerate(calls)
    }


def text_only(*deltas: str) -> FunctionModel:
    """A model that answers directly, streaming text and never calling a tool."""

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        for delta in deltas:
            yield delta

    return FunctionModel(stream_function=stream)


def one_round_then_text(round_calls: DeltaToolCalls, *deltas: str) -> FunctionModel:
    """A model that issues one Tool Round, then answers with text."""

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        if not _has_tool_return(messages):
            yield round_calls
        else:
            for delta in deltas:
                yield delta

    return FunctionModel(stream_function=stream)


def always_calls_a_tool(call: DeltaToolCalls) -> FunctionModel:
    """A confused model that asks for another lookup on every request, so only
    the usage limit can end the Turn."""

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        yield call

    return FunctionModel(stream_function=stream)


# --- test fixtures / harness ------------------------------------------------

LONDON = WeatherResult(
    location="London, England, United Kingdom", latitude=51.5, longitude=-0.12,
    observed_at="2026-07-05T10:15", temperature_c=21.8, feels_like_c=21.2,
    humidity_percent=57, wind_speed_kmh=11.9, conditions="overcast",
)
PARIS = WeatherResult(
    location="Paris, Ile-de-France, France", latitude=48.85, longitude=2.35,
    observed_at="2026-07-05T10:15", temperature_c=25.4, feels_like_c=25.0,
    humidity_percent=45, wind_speed_kmh=9.0, conditions="clear sky",
)


def settings() -> Settings:
    return Settings(
        openai_api_key="test-key", model="gpt-4.1-mini", http_timeout_seconds=10.0,
        request_limit=6, total_tokens_limit=100_000, log_level="WARNING",
        tracing_enabled=False, langsmith_api_key=None,
        langsmith_endpoint="https://example.test/otel", langsmith_project="meteobot",
    )


def canned_weather(
    reports: dict[str, WeatherResult | WeatherError], dispatched: list[str] | None = None
) -> Callable[[RunContext[Deps], str], object]:
    """A get_weather stub returning canned reports, recording dispatch order."""

    async def get_weather(ctx: RunContext[Deps], city: str) -> WeatherResult | WeatherError:
        if dispatched is not None:
            dispatched.append(city)
        return reports[city]

    return get_weather


async def drive(
    model: FunctionModel,
    tool: Callable[..., object],
    question: str,
    *,
    request_limit: int = 6,
) -> tuple[History, list[str]]:
    """Run one Turn through the real adapter and return (History, rendered deltas)."""
    agent = build_agent(model, [tool])
    rendered: list[str] = []
    history: History = []
    async with httpx.AsyncClient() as http_client:
        deps = Deps(http_client=http_client, settings=settings())
        limits = UsageLimits(request_limit=request_limit, total_tokens_limit=100_000)
        run_turn = make_run_turn(agent, deps, limits)
        await run_turn(question, history, rendered.append)
    return history, rendered


def parts_of_kind(history: History, kind: str) -> list[object]:
    return [
        p
        for m in history
        for p in getattr(m, "parts", [])
        if getattr(p, "part_kind", "") == kind
    ]


# --- tests ------------------------------------------------------------------


async def test_direct_answer_turn_streams_text_and_makes_no_tool_calls() -> None:
    async def unused_tool(ctx: RunContext[Deps], city: str) -> WeatherResult | WeatherError:
        raise AssertionError("a direct-answer Turn must not call the tool")

    history, rendered = await drive(
        text_only("It's ", "sunny ", "everywhere."), unused_tool, "Say hello."
    )

    assert "".join(rendered) == "It's sunny everywhere."
    assert parts_of_kind(history, "tool-call") == []
    # History grew: the user's request and the assistant's response are recorded.
    assert _has_tool_return(history) is False
    texts = [getattr(p, "content", "") for p in parts_of_kind(history, "text")]
    assert "".join(texts) == "It's sunny everywhere."


async def test_tool_round_dispatches_calls_and_grows_history() -> None:
    dispatched: list[str] = []
    tool = canned_weather({"London": LONDON, "Paris": PARIS}, dispatched)
    model = one_round_then_text(
        tool_calls(
            ("call_london", "get_weather", '{"city": "London"}'),
            ("call_paris", "get_weather", '{"city": "Paris"}'),
        ),
        "London is overcast; Paris is clear.",
    )

    history, rendered = await drive(model, tool, "London and Paris?")

    assert sorted(dispatched) == ["London", "Paris"]
    # Both Tool Calls and both Tool Results are recorded in History, then the
    # assistant's final answer.
    calls = parts_of_kind(history, "tool-call")
    returns = parts_of_kind(history, "tool-return")
    assert [getattr(c, "tool_name") for c in calls] == ["get_weather", "get_weather"]
    returned_locations = {getattr(r, "content").location for r in returns}
    assert returned_locations == {LONDON.location, PARIS.location}
    assert "".join(rendered) == "London is overcast; Paris is clear."


async def test_tool_calls_in_one_round_run_concurrently() -> None:
    # The stub records how many calls are in flight at once; sequential dispatch
    # would never exceed one. Both must overlap for max_concurrent to reach two.
    active = 0
    max_concurrent = 0

    async def get_weather(ctx: RunContext[Deps], city: str) -> WeatherResult | WeatherError:
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.05)
        active -= 1
        return LONDON

    model = one_round_then_text(
        tool_calls(
            ("c1", "get_weather", '{"city": "London"}'),
            ("c2", "get_weather", '{"city": "Paris"}'),
        ),
        "Both done.",
    )

    await drive(model, get_weather, "London and Paris?")

    assert max_concurrent == 2


async def test_one_failing_city_returns_a_tool_error_leaving_siblings_intact() -> None:
    # Paris comes back as a typed WeatherError; London succeeds. The error is
    # structured data the model explains, the sibling result survives, and the
    # Turn still finishes with an answer.
    paris_error = WeatherError(error="unknown_city", message="No place named 'Paris'.")
    tool = canned_weather({"London": LONDON, "Paris": paris_error})
    model = one_round_then_text(
        tool_calls(
            ("c1", "get_weather", '{"city": "London"}'),
            ("c2", "get_weather", '{"city": "Paris"}'),
        ),
        "London is overcast; I couldn't find Paris.",
    )

    history, rendered = await drive(model, tool, "London and Paris?")

    returns = parts_of_kind(history, "tool-return")
    contents = [getattr(r, "content") for r in returns]
    assert LONDON in contents
    assert paris_error in contents
    assert "".join(rendered) == "London is overcast; I couldn't find Paris."


async def test_usage_limit_trip_ends_the_turn_with_an_honest_message() -> None:
    # The model asks for another lookup on every request, so only the usage
    # limit can end the Turn. With request_limit=2 it trips, and the Turn ends
    # with the honest cap message instead of hanging or burning credits.
    tool = canned_weather({"London": LONDON})
    model = always_calls_a_tool(
        tool_calls(("c", "get_weather", '{"city": "London"}'))
    )

    _, rendered = await drive(model, tool, "Weather in London?", request_limit=2)

    assert rendered[-1] == ROUND_CAP_MESSAGE


async def test_test_model_drives_the_whole_loop_offline() -> None:
    # A smoke check that TestModel (which auto-calls every tool then answers)
    # runs a full Turn through the adapter with no scripting and no API key.
    from pydantic_ai.models.test import TestModel

    tool = canned_weather({"London": LONDON, "a": LONDON})
    agent = build_agent(TestModel(), [tool])
    rendered: list[str] = []
    history: History = []
    async with httpx.AsyncClient() as http_client:
        deps = Deps(http_client=http_client, settings=settings())
        run_turn = make_run_turn(agent, deps, UsageLimits(request_limit=6))
        await run_turn("weather in London?", history, rendered.append)

    assert rendered, "TestModel should have streamed a final answer"
    assert parts_of_kind(history, "tool-call"), "TestModel should have called the tool"
