"""Agent loop contract tests, driven through the LLM client seam.

The fake client replays scripted event sequences as real SDK event objects.
Fixture-backed scripts come from tests/fixtures/responses_stream_*.jsonl,
captured from the real Responses API, so the fake's streaming surface cannot
drift from the wire format. Fabricated scripts reuse the same captured shapes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from openai._models import construct_type
from openai.types.responses import ResponseStreamEvent

from meteobot.agent import Agent
from meteobot.config import Settings
from meteobot.llm import History
from meteobot.tools.registry import Tool, ToolRegistry
from meteobot.tools.results import ToolResult

FIXTURES = Path(__file__).parent / "fixtures"


def parse_event(value: dict[str, object]) -> ResponseStreamEvent:
    """Build an SDK event the way the SDK's own stream parser does.

    The SDK constructs stream events leniently (construct_type); strict
    pydantic validation rejects real wire events, e.g. name=null on
    response.function_call_arguments.done.
    """
    return cast(
        ResponseStreamEvent,
        construct_type(type_=cast(Any, ResponseStreamEvent), value=value),
    )


def fixture_events(name: str) -> list[ResponseStreamEvent]:
    lines = (FIXTURES / name).read_text().splitlines()
    return [parse_event(json.loads(line)["event"]) for line in lines]


def tool_call_round(
    calls: list[tuple[str, str, str]],
) -> list[ResponseStreamEvent]:
    """A fabricated Tool Round script from (call_id, name, arguments) triples,
    shaped like the captured responses_stream_tool_call.jsonl events."""
    return [
        parse_event(
            {
                "type": "response.output_item.done",
                "output_index": index,
                "sequence_number": index,
                "item": {
                    "type": "function_call",
                    "id": f"fc_{call_id}",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                },
            }
        )
        for index, (call_id, name, arguments) in enumerate(calls)
    ]


def text_round(deltas: list[str]) -> list[ResponseStreamEvent]:
    """A fabricated direct-answer script, shaped like the captured
    responses_stream_text.jsonl delta events."""
    return [
        parse_event(
            {
                "type": "response.output_text.delta",
                "delta": delta,
                "item_id": "msg_fabricated",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": index,
                "logprobs": [],
            }
        )
        for index, delta in enumerate(deltas)
    ]


@dataclass
class ScriptedLLM:
    """Fake LLM client: same streaming surface, replays one script per call.

    Records every call's arguments (with a snapshot of `input`, since the
    agent owns and mutates History in place) for inspection.
    """

    scripts: list[list[ResponseStreamEvent]]
    calls: list[dict[str, object]] = field(default_factory=list)

    async def stream(
        self,
        *,
        model: str,
        instructions: str,
        tools: list[dict[str, object]],
        input: History,
    ) -> Any:
        self.calls.append(
            {
                "model": model,
                "instructions": instructions,
                "tools": tools,
                "input": [dict(item) for item in input],
            }
        )
        for event in self.scripts[len(self.calls) - 1]:
            yield event


LONDON_REPORT = {"location": "London, England, United Kingdom", "temperature_c": 21.8}
PARIS_REPORT = {"location": "Paris, Ile-de-France, France", "temperature_c": 25.4}

GET_WEATHER_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def canned_weather_registry(
    reports: dict[str, ToolResult],
    dispatched: list[str] | None = None,
) -> ToolRegistry:
    """A get_weather fake returning canned reports, recording dispatch order."""

    async def handler(city: str) -> ToolResult:
        if dispatched is not None:
            dispatched.append(city)
        return reports[city]

    return ToolRegistry(
        tools=(
            Tool(
                name="get_weather",
                description="Get the current weather for one city by name.",
                parameters=GET_WEATHER_PARAMETERS,
                handler=handler,
            ),
        )
    )


def settings(tool_round_cap: int = 5, log_level: str = "WARNING") -> Settings:
    return Settings(
        openai_api_key="test-key",
        model="gpt-4.1-mini",
        http_timeout_seconds=10.0,
        tool_round_cap=tool_round_cap,
        log_level=log_level,
    )


async def test_direct_answer_turn_streams_text_and_makes_no_tool_calls() -> None:
    events = fixture_events("responses_stream_text.jsonl")
    expected_deltas = [
        e.delta for e in events if e.type == "response.output_text.delta"
    ]
    llm = ScriptedLLM(scripts=[events])
    agent = Agent(llm=llm, registry=ToolRegistry(tools=()), settings=settings())

    history: History = []
    rendered: list[str] = []
    await agent.run_turn(
        "What's the weather in London and Paris right now?", history, rendered.append
    )

    assert rendered == expected_deltas
    assert len(llm.calls) == 1
    assert history == [
        {
            "role": "user",
            "content": "What's the weather in London and Paris right now?",
        },
        {"role": "assistant", "content": "".join(expected_deltas)},
    ]
    # The LLM saw the full client-owned History, the configured model, and the
    # registry's schemas.
    assert llm.calls[0]["input"] == [history[0]]
    assert llm.calls[0]["model"] == "gpt-4.1-mini"
    assert llm.calls[0]["tools"] == []


async def test_tool_round_dispatches_calls_and_grows_history() -> None:
    # Captured Turn: the model asks for London and Paris, then answers.
    llm = ScriptedLLM(
        scripts=[
            fixture_events("responses_stream_tool_call.jsonl"),
            fixture_events("responses_stream_text.jsonl"),
        ]
    )
    dispatched: list[str] = []
    registry = canned_weather_registry(
        {"London": LONDON_REPORT, "Paris": PARIS_REPORT}, dispatched
    )
    agent = Agent(llm=llm, registry=registry, settings=settings())

    history: History = []
    rendered: list[str] = []
    await agent.run_turn(
        "What's the weather in London and Paris right now?", history, rendered.append
    )

    assert sorted(dispatched) == ["London", "Paris"]
    assert len(llm.calls) == 2

    # History: user, one function_call + function_call_output pair per city
    # (call_ids from the captured fixture), then the assistant's answer.
    user, call_london, call_paris, out_london, out_paris, answer = history
    assert call_london == {
        "type": "function_call",
        "call_id": "call_qNxcWhcXv8BexP3pNwT5JVdF",
        "name": "get_weather",
        "arguments": '{"city":"London"}',
    }
    assert call_paris["call_id"] == "call_tLBNdsMwrfPE1wimnY9WpS6g"
    assert out_london["type"] == "function_call_output"
    assert out_london["call_id"] == call_london["call_id"]
    assert json.loads(cast(str, out_london["output"])) == LONDON_REPORT
    assert out_paris["call_id"] == call_paris["call_id"]
    assert json.loads(cast(str, out_paris["output"])) == PARIS_REPORT
    assert answer == {"role": "assistant", "content": "".join(rendered)}
    assert rendered, "the answer round should have streamed text deltas"

    # The second LLM call saw the full History up to that point.
    assert llm.calls[1]["input"] == [user, call_london, call_paris, out_london, out_paris]


async def test_tool_calls_in_one_round_run_concurrently() -> None:
    # Both handlers block on a shared barrier: sequential dispatch would
    # deadlock, so passing at all proves the calls were in flight together.
    barrier = asyncio.Barrier(2)

    async def handler(city: str) -> ToolResult:
        async with asyncio.timeout(1):
            await barrier.wait()
        return {"city": city}

    registry = ToolRegistry(
        tools=(
            Tool(
                name="get_weather",
                description="Get the current weather for one city by name.",
                parameters=GET_WEATHER_PARAMETERS,
                handler=handler,
            ),
        )
    )
    llm = ScriptedLLM(
        scripts=[
            tool_call_round(
                [
                    ("call_1", "get_weather", '{"city": "London"}'),
                    ("call_2", "get_weather", '{"city": "Paris"}'),
                ]
            ),
            text_round(["Both", " done."]),
        ]
    )
    agent = Agent(llm=llm, registry=registry, settings=settings())

    history: History = []
    await agent.run_turn("London and Paris?", history, lambda _: None)

    outputs = [item for item in history if item.get("type") == "function_call_output"]
    assert [json.loads(cast(str, o["output"])) for o in outputs] == [
        {"city": "London"},
        {"city": "Paris"},
    ]


async def test_one_failing_tool_call_leaves_sibling_results_intact() -> None:
    async def handler(city: str) -> ToolResult:
        if city == "Paris":
            raise RuntimeError("handler bug")
        return LONDON_REPORT

    registry = ToolRegistry(
        tools=(
            Tool(
                name="get_weather",
                description="Get the current weather for one city by name.",
                parameters=GET_WEATHER_PARAMETERS,
                handler=handler,
            ),
        )
    )
    llm = ScriptedLLM(
        scripts=[
            tool_call_round(
                [
                    ("call_1", "get_weather", '{"city": "London"}'),
                    ("call_2", "get_weather", '{"city": "Paris"}'),
                ]
            ),
            text_round(["London is fine; Paris lookup failed."]),
        ]
    )
    agent = Agent(llm=llm, registry=registry, settings=settings())

    history: History = []
    await agent.run_turn("London and Paris?", history, lambda _: None)

    outputs = {
        cast(str, item["call_id"]): json.loads(cast(str, item["output"]))
        for item in history
        if item.get("type") == "function_call_output"
    }
    # The sibling result survives; the failure goes back as a Tool Error the
    # LLM can explain, and the Turn still finishes with an answer.
    assert outputs["call_1"] == LONDON_REPORT
    assert outputs["call_2"] == {
        "error": "tool_execution_error",
        "message": "The get_weather tool failed unexpectedly for this call.",
    }
    assert len(llm.calls) == 2
    assert history[-1] == {
        "role": "assistant",
        "content": "London is fine; Paris lookup failed.",
    }


async def test_tool_round_cap_ends_the_turn_with_an_honest_message() -> None:
    # The scripted model asks for another lookup on every call, so only the
    # cap can end the Turn.
    llm = ScriptedLLM(
        scripts=[
            tool_call_round([(f"call_{n}", "get_weather", '{"city": "London"}')])
            for n in range(3)
        ]
    )
    registry = canned_weather_registry({"London": LONDON_REPORT})
    agent = Agent(llm=llm, registry=registry, settings=settings(tool_round_cap=2))

    history: History = []
    rendered: list[str] = []
    await agent.run_turn("Weather in London?", history, rendered.append)

    # Two Tool Rounds ran; the third request was refused, ending the Turn
    # with an honest message instead of hanging or lying.
    assert len(llm.calls) == 3
    executed = [item for item in history if item.get("type") == "function_call"]
    assert [item["call_id"] for item in executed] == ["call_0", "call_1"]
    final = history[-1]
    assert final["role"] == "assistant"
    message = cast(str, final["content"])
    assert rendered[-1] == message
    assert "lookup" in message and "limit" in message


def _agent_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "meteobot.agent"]


async def test_info_level_traces_each_step_of_the_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="meteobot.agent")
    llm = ScriptedLLM(
        scripts=[
            fixture_events("responses_stream_tool_call.jsonl"),
            fixture_events("responses_stream_text.jsonl"),
        ]
    )
    registry = canned_weather_registry({"London": LONDON_REPORT, "Paris": PARIS_REPORT})
    agent = Agent(llm=llm, registry=registry, settings=settings(log_level="INFO"))

    await agent.run_turn("London and Paris?", [], lambda _: None)

    messages = [r.getMessage() for r in _agent_logs(caplog)]
    # Turn start is logged once.
    assert any("turn start" in m for m in messages)
    # The chosen Tool Calls for the round are logged, naming the tool.
    assert any("tool round" in m and "get_weather" in m for m in messages)
    # Each Tool Call logs its name, an "ok" status, and a latency in ms. Two
    # cities → two per-call lines.
    per_call = [m for m in messages if "get_weather(" in m and "ms" in m]
    assert len(per_call) == 2
    assert all("ok" in m for m in per_call)


async def test_warning_level_emits_no_per_step_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # At the default level a normal Turn stays silent: the per-step trace is
    # opt-in, so it never pollutes an ordinary session.
    caplog.set_level(logging.WARNING, logger="meteobot.agent")
    llm = ScriptedLLM(
        scripts=[
            fixture_events("responses_stream_tool_call.jsonl"),
            fixture_events("responses_stream_text.jsonl"),
        ]
    )
    registry = canned_weather_registry({"London": LONDON_REPORT, "Paris": PARIS_REPORT})
    agent = Agent(llm=llm, registry=registry, settings=settings())

    await agent.run_turn("London and Paris?", [], lambda _: None)

    assert _agent_logs(caplog) == []


async def test_cap_hit_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    # A cap trip is an anomaly worth seeing even in a quiet session, so it logs
    # at WARNING rather than hiding in the opt-in INFO trace.
    caplog.set_level(logging.WARNING, logger="meteobot.agent")
    llm = ScriptedLLM(
        scripts=[
            tool_call_round([(f"call_{n}", "get_weather", '{"city": "London"}')])
            for n in range(3)
        ]
    )
    registry = canned_weather_registry({"London": LONDON_REPORT})
    agent = Agent(llm=llm, registry=registry, settings=settings(tool_round_cap=2))

    await agent.run_turn("Weather in London?", [], lambda _: None)

    warnings = [r for r in _agent_logs(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "cap" in warnings[0].getMessage().lower()
