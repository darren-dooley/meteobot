"""Opt-in live end-to-end test against the real Open-Meteo and OpenAI APIs.

Excluded from the default suite (`addopts = -m 'not live'`); run it explicitly
with `uv run pytest -m live`. It skips cleanly when no API key is present, so
the default offline suite needs no key and no network.

Assertions are deliberately loose. Exact model wording is not a contract, so
the test proves the wiring end to end — a real Turn drove both APIs, an answer
streamed back, both named cities appear, and History grew through a real Tool
Round — rather than pinning a phrasing that model updates would break.
"""

from __future__ import annotations

import httpx
import pytest
from openai import AsyncOpenAI

from meteobot.agent import Agent
from meteobot.config import MissingAPIKeyError, load_settings
from meteobot.llm import History, OpenAIResponsesLLM
from meteobot.tools.registry import ToolRegistry
from meteobot.tools.weather import weather_tool

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
        AsyncOpenAI(api_key=settings.openai_api_key) as openai_client,
    ):
        llm = OpenAIResponsesLLM(client=openai_client)
        registry = ToolRegistry(tools=(weather_tool(http_client),))
        agent = Agent(llm=llm, registry=registry, settings=settings)

        history: History = []
        rendered: list[str] = []
        await agent.run_turn(QUESTION, history, rendered.append)

    answer = "".join(rendered)
    # An answer actually streamed back, token by token.
    assert rendered, "the live Turn should have streamed at least one text delta"
    # Both cities the question named surface in the streamed answer.
    assert "London" in answer
    assert "Tokyo" in answer
    # History grew through a real Tool Round: the user message, at least one
    # get_weather Tool Call with an output, then the assistant's answer.
    assert history[0] == {"role": "user", "content": QUESTION}
    tool_calls = [item for item in history if item.get("type") == "function_call"]
    assert tool_calls, "a live weather question should have called get_weather"
    assert all(item["name"] == "get_weather" for item in tool_calls)
    assert any(item.get("type") == "function_call_output" for item in history)
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"], "the Turn should end with a non-empty answer"
