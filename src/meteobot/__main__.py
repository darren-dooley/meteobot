"""Composition root: settings -> tracing -> shared HTTP client -> model -> agent -> CLI.

Everything is assembled here and injected; no module has import-time side
effects. The shared httpx client, the OpenAI Responses model, the tool
functions, and (when enabled) the LangSmith instrumentation are wired together
into one PydanticAI agent, and the streaming Turn adapter is handed to the REPL.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import httpx
from openai import AsyncOpenAI

from pydantic_ai import Tool

from meteobot.agent import (
    History,
    build_agent,
    build_openai_model,
    instructions_for,
    make_run_turn,
    usage_limits_from,
)
from meteobot.cli import start_repl
from meteobot.config import ConfigError, Settings, load_settings
from meteobot.deps import Deps
from meteobot.features import build_advanced_tooling, merge_capabilities
from meteobot.tool_examples import with_examples
from meteobot.tools.weather import get_weather
from meteobot.tracing import build_instrumentation

# get_weather with tool-use examples folded into its description: two calls that
# show a plain city and a multi-word city, so the model sees the expected shape.
WEATHER_TOOL = Tool(
    get_weather,
    name="get_weather",
    description=with_examples(
        "Get the current weather for one city by name. Call once per city when "
        "several cities are asked about.",
        "get_weather",
        [{"city": "London"}, {"city": "San Francisco"}],
    ),
)


async def _run(settings: Settings) -> None:
    """Assemble the app around the shared httpx client and hand off to the CLI."""
    capabilities, tracer_provider = build_instrumentation(settings)

    # One shared async client for all tools: connection pooling across the
    # concurrent fan-out, timeout and connect-retry policy configured once.
    # httpx does not retry by default; retries=1 covers connect failures only.
    # The same client backs the OpenAI SDK, so every outbound call shares it.
    transport = httpx.AsyncHTTPTransport(retries=1)
    try:
        async with (
            httpx.AsyncClient(
                timeout=settings.http_timeout_seconds, transport=transport
            ) as http_client,
            AsyncOpenAI(
                api_key=settings.openai_api_key, http_client=http_client
            ) as openai_client,
            build_advanced_tooling(settings) as advanced,
        ):
            model = build_openai_model(settings, openai_client)
            agent = build_agent(
                model,
                [WEATHER_TOOL, *advanced.extra_tools],
                toolsets=advanced.toolsets,
                capabilities=merge_capabilities(capabilities, advanced.capabilities),
                instructions=instructions_for(capabilities_enabled=advanced.enabled),
            )
            deps = Deps(
                http_client=http_client,
                settings=settings,
                sandbox=advanced.sandbox,
            )
            run_turn = make_run_turn(agent, deps, usage_limits_from(settings))

            history: History = []
            await start_repl(run_turn, history)
    finally:
        # Flush the batch span exporter so the last Turn's trace reaches
        # LangSmith before the process exits.
        if tracer_provider is not None:
            tracer_provider.shutdown()


def main() -> int:
    """Console-script entry point. Returns the process exit code."""
    try:
        settings = load_settings()
    except ConfigError as err:
        print(str(err), file=sys.stderr)
        return 1
    # Diagnostics go to stderr so they never interleave with the streamed
    # answer on stdout; level-gated by METEOBOT_LOG_LEVEL (default WARNING).
    # LangSmith traces are the primary observability path; these lines remain
    # as an offline fallback.
    logging.basicConfig(
        level=settings.log_level,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_run(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
