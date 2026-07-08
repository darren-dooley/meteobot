"""Composition root: settings → shared HTTP client → LLM client → registry → agent → CLI.

Everything is assembled here and injected; no module has import-time side effects.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import httpx
from openai import AsyncOpenAI

from meteobot.agent import Agent
from meteobot.cli import start_repl
from meteobot.config import ConfigError, Settings, load_settings
from meteobot.llm import History, OpenAIResponsesLLM
from meteobot.tools.registry import ToolRegistry
from meteobot.tools.weather import weather_tool


async def _run(settings: Settings) -> None:
    """Assemble the app around the shared httpx client and hand off to the CLI."""
    # One shared async client for all tools: connection pooling across the
    # concurrent fan-out, timeout and connect-retry policy configured once.
    # httpx does not retry by default; retries=1 covers connect failures only.
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
        await start_repl(agent, history)


def main() -> int:
    """Console-script entry point. Returns the process exit code."""
    try:
        settings = load_settings()
    except ConfigError as err:
        print(str(err), file=sys.stderr)
        return 1
    # Diagnostics go to stderr so they never interleave with the streamed
    # answer on stdout; level-gated by METEOBOT_LOG_LEVEL (default WARNING).
    logging.basicConfig(
        level=settings.log_level,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_run(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
