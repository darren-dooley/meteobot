"""Programmatic tool calling: a code-execution environment that calls MCP tools.

This package is meteobot's provider-agnostic take on the "Programmatic Tool
Calling" idea — the model writes Python that calls MCP tools, the code runs in a
sandbox, and only its stdout returns to the model, keeping bulk intermediate
data out of the context window. It has four parts:

- `McpTools` — the trusted side: a connected FastMCP client the parent uses to
  actually invoke MCP tools by name. Both executors call back into this.
- `Executor` / `ExecResult` — the swappable execution seam and its typed result.
- `DockerExecutor` — the default: model code runs in a locked-down `--network
  none` container and reaches `McpTools` only over a stdio RPC bridge.
- `InProcessExecutor` — the no-Docker fallback and the test double: the same
  code runs in-process with tool proxies wired straight to `McpTools`.

`Sandbox` binds an executor to an `McpTools` so the `run_python` tool has one
handle to call. See `meteobot.codeexec.tool` for the PydanticAI adapter.
"""

from __future__ import annotations

from meteobot.codeexec.executor import (
    ExecResult,
    Executor,
    InProcessExecutor,
    Sandbox,
)
from meteobot.codeexec.mcp_tools import McpTools

__all__ = [
    "ExecResult",
    "Executor",
    "InProcessExecutor",
    "McpTools",
    "Sandbox",
]
