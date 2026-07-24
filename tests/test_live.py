"""Opt-in live end-to-end test against the real Open-Meteo and OpenAI APIs.

Excluded from the default suite (`addopts = -m 'not live'`); run it explicitly
with `uv run pytest -m live`. It skips cleanly when no API key is present, so
the default offline suite needs no key and no network.

Assertions are deliberately loose. Exact model wording is not a contract, so
the test proves the wiring end to end — a real Turn drove both APIs through the
PydanticAI agent, an answer streamed back, both named cities appear, and the
History grew through a real Tool Round — rather than pinning a phrasing that
model updates would break.
"""

from __future__ import annotations

import httpx
import pytest
from openai import AsyncOpenAI

from meteobot.agent import (
    History,
    build_agent,
    build_openai_model,
    make_run_turn,
    usage_limits_from,
)
from meteobot.config import MissingAPIKeyError, load_settings
from meteobot.deps import Deps
from meteobot.tools.weather import get_weather

pytestmark = pytest.mark.live

QUESTION = "What's the weather in London and Tokyo right now?"


async def test_multi_city_question_streams_a_real_answer() -> None:
    try:
        settings = load_settings()
    except MissingAPIKeyError:
        pytest.skip("no OPENAI_API_KEY present; live e2e test skipped")

    # Assemble the real stack exactly as the composition root does.
    transport = httpx.AsyncHTTPTransport(retries=1)
    async with (
        httpx.AsyncClient(
            timeout=settings.http_timeout_seconds, transport=transport
        ) as http_client,
        AsyncOpenAI(
            api_key=settings.openai_api_key, http_client=http_client
        ) as openai_client,
    ):
        model = build_openai_model(settings, openai_client)
        agent = build_agent(model, [get_weather])
        deps = Deps(http_client=http_client, settings=settings)
        run_turn = make_run_turn(agent, deps, usage_limits_from(settings))

        history: History = []
        rendered: list[str] = []
        await run_turn(QUESTION, history, rendered.append)

    answer = "".join(rendered)
    # An answer actually streamed back, token by token.
    assert rendered, "the live Turn should have streamed at least one text delta"
    # Both cities the question named surface in the streamed answer.
    assert "London" in answer
    assert "Tokyo" in answer
    # History grew through a real Tool Round: at least one get_weather Tool Call.
    tool_calls = [
        p
        for m in history
        for p in getattr(m, "parts", [])
        if getattr(p, "part_kind", "") == "tool-call"
    ]
    assert tool_calls, "a live weather question should have called get_weather"
    assert all(getattr(p, "tool_name") == "get_weather" for p in tool_calls)
