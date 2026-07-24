"""Instrumentation tests: fail-open configuration and real span emission.

Two things are verified here. First, `build_instrumentation` makes the right
call from configuration alone: off when tracing is disabled, degrading to no
tracing (not a crash) when the LangSmith key is missing, and producing a
capability plus its owning provider when fully configured. Second, that wiring
actually works — running a Turn through an instrumented agent emits the
expected Turn and Tool Call spans, captured with an in-memory exporter so the
assertion never contacts LangSmith.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import InstrumentationSettings, RunContext
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import UsageLimits

from meteobot.agent import build_agent, make_run_turn
from meteobot.config import Settings
from meteobot.deps import Deps
from meteobot.tools.weather import WeatherError, WeatherResult
from meteobot.tracing import (
    _langsmith_headers,
    _traces_endpoint,
    build_instrumentation,
)


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        openai_api_key="test-key", model="gpt-4.1-mini", http_timeout_seconds=10.0,
        request_limit=6, total_tokens_limit=100_000, log_level="WARNING",
        tool_search_enabled=False, code_exec_enabled=False, executor="inprocess",
        sandbox_image="python:3.12-slim", code_exec_timeout_seconds=30.0,
        tracing_enabled=False, langsmith_api_key=None,
        langsmith_endpoint="https://example.test", langsmith_project="meteobot",
        langsmith_workspace_id=None,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_traces_endpoint_derives_the_otlp_url_from_the_api_base() -> None:
    full = "https://api.smith.langchain.com/otel/v1/traces"
    # The LangSmith API base URL gains both the /otel collector path and the
    # traces signal path; posting to a bare base or /otel is a 403.
    assert _traces_endpoint("https://api.smith.langchain.com") == full
    # Trailing slash tolerant, and idempotent whether given the base, the
    # collector path, or the already-complete URL.
    assert _traces_endpoint("https://api.smith.langchain.com/") == full
    assert _traces_endpoint("https://api.smith.langchain.com/otel") == full
    assert _traces_endpoint(full) == full


def test_headers_carry_key_and_project_but_omit_workspace_when_unset() -> None:
    headers = _langsmith_headers(
        settings(langsmith_api_key="ls-secret", langsmith_workspace_id=None)
    )
    assert headers == {"x-api-key": "ls-secret", "Langsmith-Project": "meteobot"}


def test_multi_workspace_key_adds_the_workspace_id_header() -> None:
    # A key linked to more than one workspace is forbidden (403) without this;
    # the header is what routes the trace to the right workspace.
    headers = _langsmith_headers(
        settings(langsmith_api_key="ls-secret", langsmith_workspace_id="ws-123")
    )
    assert headers["Langsmith-Workspace-Id"] == "ws-123"


def test_tracing_off_returns_no_capability() -> None:
    caps, provider = build_instrumentation(settings(tracing_enabled=False))
    assert caps == ()
    assert provider is None


def test_tracing_on_without_key_fails_open_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="meteobot.tracing")
    caps, provider = build_instrumentation(
        settings(tracing_enabled=True, langsmith_api_key=None)
    )
    # Degrades to a normal untraced session rather than crashing.
    assert caps == ()
    assert provider is None
    assert any("LANGSMITH_API_KEY" in r.getMessage() for r in caplog.records)


def test_tracing_fully_configured_returns_capability_and_provider() -> None:
    caps, provider = build_instrumentation(
        settings(tracing_enabled=True, langsmith_api_key="ls-secret")
    )
    try:
        assert len(caps) == 1
        assert isinstance(caps[0], Instrumentation)
        assert provider is not None
    finally:
        if provider is not None:
            provider.shutdown()


LONDON = WeatherResult(
    location="London, England, United Kingdom", latitude=51.5, longitude=-0.12,
    observed_at="2026-07-05T10:15", temperature_c=21.8, feels_like_c=21.2,
    humidity_percent=57, wind_speed_kmh=11.9, conditions="overcast",
)


def _model() -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo):
        has_return = any(
            getattr(p, "part_kind", "") == "tool-return"
            for m in messages
            for p in getattr(m, "parts", [])
        )
        if not has_return:
            yield {
                0: DeltaToolCall(
                    name="get_weather", json_args='{"city": "London"}', tool_call_id="c1"
                )
            }
        else:
            yield "London is overcast."

    return FunctionModel(stream_function=stream)


async def test_a_turn_emits_turn_and_tool_call_spans() -> None:
    # In-memory exporter: assert instrumentation is wired without a network hop.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    capability = Instrumentation(
        settings=InstrumentationSettings(tracer_provider=provider)
    )

    async def get_weather(
        ctx: RunContext[Deps], city: str
    ) -> WeatherResult | WeatherError:
        return LONDON

    agent = build_agent(_model(), [get_weather], capabilities=[capability])
    async with httpx.AsyncClient() as http_client:
        deps = Deps(http_client=http_client, settings=settings())
        run_turn = make_run_turn(agent, deps, UsageLimits(request_limit=6))
        await run_turn("weather in London?", [], lambda _: None)

    names = [span.name for span in exporter.get_finished_spans()]
    # The Turn is a span, and the Tool Call it made is a nested span.
    assert any(name.startswith("invoke_agent") for name in names), names
    assert "execute_tool get_weather" in names, names
