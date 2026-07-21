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

from meteobot.agent import (
    History,
    build_agent,
    build_openai_model,
    make_run_turn,
    usage_limits_from,
)
from meteobot.cli import start_repl
from meteobot.config import ConfigError, Settings, load_settings
from meteobot.deps import Deps
from meteobot.tools.weather import get_weather
from meteobot.tracing import build_instrumentation


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
        ):
            model = build_openai_model(settings, openai_client)
            agent = build_agent(model, [get_weather], capabilities=capabilities)
            deps = Deps(http_client=http_client, settings=settings)
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
