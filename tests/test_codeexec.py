"""Code execution over MCP tools, driven through the in-process demo server.

`McpTools` connects to the real `converter` FastMCP instance in-process, so the
in-process executor runs against the same tools production uses. The Docker
executor has its own module (test_docker_executor.py) so this file needs no
container and stays fast.
"""

from __future__ import annotations

import pytest

from meteobot.codeexec import InProcessExecutor, McpTools, Sandbox
from meteobot.codeexec.executor import ExecResult
from meteobot.codeexec.tool import build_description, run_python
from meteobot.deps import Deps
from meteobot.mcp_toolbox import server


async def test_mcp_tools_lists_and_calls_unwrapping_scalar_results() -> None:
    async with McpTools.connect(server) as tools:
        assert "celsius_to_fahrenheit" in tools.names
        # FastMCP wraps a scalar return as {"result": ...}; McpTools unwraps it
        # so model code sees a plain float.
        value = await tools.call("celsius_to_fahrenheit", {"celsius": 100.0})
    assert value == pytest.approx(212.0)


async def test_in_process_executor_runs_code_and_returns_only_stdout() -> None:
    code = (
        "f = await celsius_to_fahrenheit(celsius=21.8)\n"
        "print(round(f, 2))\n"
    )
    async with McpTools.connect(server) as tools:
        result = await InProcessExecutor().run(code, tools)
    assert isinstance(result, ExecResult)
    assert result.stdout.strip() == "71.24"
    assert result.error is None
    assert result.tool_calls == 1


async def test_in_process_executor_supports_parallel_tool_calls() -> None:
    code = (
        "temps = [0.0, 100.0, 21.8]\n"
        "out = await asyncio.gather("
        "*[celsius_to_fahrenheit(celsius=t) for t in temps])\n"
        "print(json.dumps([round(x, 2) for x in out]))\n"
    )
    async with McpTools.connect(server) as tools:
        result = await InProcessExecutor().run(code, tools)
    assert result.error is None
    assert result.stdout.strip() == "[32.0, 212.0, 71.24]"
    assert result.tool_calls == 3


async def test_in_process_executor_reports_errors_as_data_not_raises() -> None:
    async with McpTools.connect(server) as tools:
        result = await InProcessExecutor().run("raise ValueError('boom')", tools)
    assert result.error is not None
    assert "ValueError" in result.error
    assert "boom" in result.error


async def test_in_process_executor_enforces_a_timeout() -> None:
    async with McpTools.connect(server) as tools:
        result = await InProcessExecutor(timeout_seconds=0.1).run(
            "await asyncio.sleep(5)", tools
        )
    assert result.error is not None
    assert "time limit" in result.error


async def test_sandbox_delegates_to_its_executor() -> None:
    async with McpTools.connect(server) as tools:
        sandbox = Sandbox(executor=InProcessExecutor(), tools=tools)
        result = await sandbox.run("print(await kmh_to_mph(kmh=1.609344))")
    assert result.stdout.strip() == "1.0"


async def test_run_python_tool_without_a_sandbox_returns_a_tool_error() -> None:
    # No sandbox on Deps => code execution disabled; the tool returns a readable
    # error rather than raising, keeping it in the Tool Error tier.
    ctx = _ctx(Deps(http_client=None, settings=None, sandbox=None))  # type: ignore[arg-type]
    result = await run_python(ctx, "print(1)")
    assert result.stdout == ""
    assert result.error is not None
    assert "not enabled" in result.error


async def test_run_python_description_lists_the_live_tool_catalogue() -> None:
    async with McpTools.connect(server) as tools:
        description = build_description(tools)
    assert "celsius_to_fahrenheit(celsius: number)" in description
    assert "Examples" in description


def _ctx(deps: Deps):
    """A minimal RunContext carrying deps for a direct tool-function call."""
    from pydantic_ai import RunContext

    return RunContext(deps=deps, model=None, usage=None)  # type: ignore[call-arg]
