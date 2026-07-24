"""The Docker executor, against a real container. Skipped when Docker is absent.

These tests spin up an actual `--network none` container per run, so they are
opt-in on Docker being usable: `preflight()` gates the whole module. The MCP
tools run in-process in the parent (the test process); the container reaches
them only over the stdio RPC bridge, which is exactly the production split and
is what lets a network-isolated container still call tools.
"""

from __future__ import annotations

import asyncio

import pytest

from meteobot.codeexec import McpTools
from meteobot.codeexec.docker_executor import DockerExecutor, preflight
from meteobot.mcp_toolbox import server


def _docker_available() -> bool:
    return asyncio.run(preflight()) is None


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker is not available"
)


async def test_preflight_reports_docker_usable() -> None:
    assert await preflight() is None


async def test_docker_executor_runs_code_calling_mcp_tools() -> None:
    code = (
        "temps = [0.0, 100.0]\n"
        "out = await asyncio.gather("
        "*[celsius_to_fahrenheit(celsius=t) for t in temps])\n"
        "print(json.dumps([round(x, 2) for x in out]))\n"
    )
    async with McpTools.connect(server) as tools:
        result = await DockerExecutor(timeout_seconds=60).run(code, tools)
    assert result.error is None, result.error
    assert result.stdout.strip() == "[32.0, 212.0]"
    assert result.tool_calls == 2


async def test_docker_executor_returns_snippet_errors_as_data() -> None:
    async with McpTools.connect(server) as tools:
        result = await DockerExecutor(timeout_seconds=60).run(
            "raise ValueError('boom')", tools
        )
    assert result.error is not None
    assert "boom" in result.error


async def test_docker_executor_has_no_network() -> None:
    # The container is --network none, so an outbound socket must fail. The
    # snippet catches the failure and prints a marker either way so the result
    # is deterministic.
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('REACHABLE')\n"
        "except Exception:\n"
        "    print('BLOCKED')\n"
    )
    async with McpTools.connect(server) as tools:
        result = await DockerExecutor(timeout_seconds=60).run(code, tools)
    assert result.stdout.strip() == "BLOCKED"


async def test_docker_executor_enforces_a_timeout() -> None:
    async with McpTools.connect(server) as tools:
        result = await DockerExecutor(timeout_seconds=2).run(
            "await asyncio.sleep(30)", tools
        )
    assert result.error is not None
    assert "time limit" in result.error
