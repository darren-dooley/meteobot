"""The execution seam: run model-written code, return only its stdout.

`Executor` is the one interface the rest of the app depends on, so the isolation
strategy is swappable without touching the agent or the MCP wiring. Two
implementations ship:

- `InProcessExecutor` here — runs code in this process with tool proxies wired
  straight to `McpTools`. It is NOT a security boundary: model code shares the
  interpreter and can reach anything this process can. It exists as the
  Docker-absent fallback and as the test double, both single-user local paths.
- `DockerExecutor` (sibling module) — the default, a real `--network none`
  container boundary.

`Sandbox` binds a chosen executor to a connected `McpTools`, giving the
`run_python` tool a single `run(code)` to call. Every result comes back as a
typed `ExecResult`; an executor reports failure in `ExecResult.error` rather than
raising, so a broken snippet is a Tool Error the model can read and retry, never
something that aborts the Turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from meteobot.codeexec.mcp_tools import McpTools

# Wall-clock ceiling for one code-execution run, in seconds. Bounds a runaway
# loop in the model's code the way the usage limit bounds a runaway Turn.
DEFAULT_TIMEOUT_SECONDS = 30.0


class ExecResult(BaseModel):
    """The outcome of one code-execution run, as the model sees it."""

    stdout: str
    error: str | None = None
    tool_calls: int = 0


class Executor(Protocol):
    """Runs a snippet against a set of MCP tools; returns only its stdout."""

    async def run(self, code: str, tools: McpTools) -> ExecResult: ...


@dataclass(frozen=True)
class Sandbox:
    """An executor bound to its MCP tools — the `run_python` tool's one handle."""

    executor: Executor
    tools: McpTools

    async def run(self, code: str) -> ExecResult:
        """Run `code` with the bound tools and executor."""
        return await self.executor.run(code, self.tools)


def wrap_as_async_main(code: str) -> str:
    """Wrap a snippet as an `async def __main__()` body so it can `await`.

    Indenting the user's lines under one async function lets the snippet use
    `await tool(...)` and `asyncio.gather(...)` at top level, which is the whole
    point of programmatic tool calling.
    """
    body = "".join("    " + line + "\n" for line in code.splitlines())
    return "async def __main__():\n" + (body or "    pass\n")


class InProcessExecutor:
    """Runs code in this process. Convenient and unsandboxed; see module docs."""

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout_seconds

    async def run(self, code: str, tools: McpTools) -> ExecResult:
        """Execute `code` with tool proxies that call `tools` directly."""
        calls = 0

        def make_proxy(name: str):
            async def proxy(**kwargs: object) -> object:
                nonlocal calls
                calls += 1
                return await tools.call(name, dict(kwargs))

            return proxy

        namespace: dict[str, object] = {"asyncio": asyncio, "json": __import__("json")}
        for name in tools.names:
            namespace[name] = make_proxy(name)

        buffer = io.StringIO()
        try:
            exec(wrap_as_async_main(code), namespace)  # noqa: S102 - see module docs
            main = namespace["__main__"]
            with contextlib.redirect_stdout(buffer):
                await asyncio.wait_for(main(), timeout=self._timeout)  # type: ignore[operator]
        except asyncio.TimeoutError:
            return ExecResult(
                stdout=buffer.getvalue(),
                error=f"Code execution exceeded the {self._timeout:g}s time limit.",
                tool_calls=calls,
            )
        except Exception as exc:
            return ExecResult(
                stdout=buffer.getvalue(),
                error=f"{type(exc).__name__}: {exc}",
                tool_calls=calls,
            )
        return ExecResult(stdout=buffer.getvalue(), tool_calls=calls)
