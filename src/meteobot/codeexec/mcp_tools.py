"""The trusted side of code execution: a connected client to the MCP tools.

`McpTools` wraps a FastMCP `Client` over the demo MCP server. It is the only
component that actually invokes an MCP tool, and it always runs in the parent
process — never inside the sandbox. Both executors treat it as the callback
target: the in-process executor calls `call()` directly from tool proxies, and
the Docker executor calls it while servicing the container's RPC requests. That
split is the whole safety story — untrusted model code never holds the client,
it can only ask the parent to make a named call on its behalf.

The client is opened once at the composition root (`async with McpTools.connect
(...)`) and injected, mirroring how the shared httpx client is managed. A tool's
return value is normalised to a JSON-serialisable form so it can cross the
Docker RPC boundary and land back in the model's code as an ordinary value.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import Client

import meteobot.mcp_toolbox as mcp_toolbox


def server_script_path() -> Path:
    """Filesystem path to the demo MCP server, run as a stdio subprocess.

    FastMCP builds a stdio transport from a `.py` path by launching it with the
    current interpreter; the server module has no intra-package imports, so
    running it as a plain script (rather than `-m`) resolves cleanly.
    """
    return Path(mcp_toolbox.__file__)


@dataclass(frozen=True)
class ToolInfo:
    """One MCP tool's model-facing surface: how code-exec advertises it."""

    name: str
    description: str
    parameters: dict[str, Any]


class McpTools:
    """A connected view over the MCP server: list tools, call a tool by name."""

    def __init__(self, client: Client, tools: list[ToolInfo]) -> None:
        self._client = client
        self._tools = tools

    @classmethod
    @asynccontextmanager
    async def connect(cls, transport: Any) -> AsyncIterator[McpTools]:
        """Open a client to `transport` and prefetch the tool catalogue.

        `transport` is anything FastMCP accepts: the server script `Path` in
        production, or an in-process `FastMCP` instance in tests. The catalogue
        is read once on connect; the demo server's tool set is static.
        """
        async with Client(transport) as client:
            listed = await client.list_tools()
            tools = [
                ToolInfo(
                    name=t.name,
                    description=t.description or "",
                    parameters=dict(t.inputSchema or {}),
                )
                for t in listed
            ]
            yield cls(client, tools)

    @property
    def tools(self) -> list[ToolInfo]:
        """The MCP tool catalogue, prefetched at connect time."""
        return list(self._tools)

    @property
    def names(self) -> list[str]:
        """Just the tool names — injected into the sandbox as callables."""
        return [t.name for t in self._tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke one MCP tool and return its result as a JSON-able value.

        FastMCP wraps a scalar tool return as `{"result": value}`; that single
        wrapper is unwrapped so model code sees `celsius_to_fahrenheit(...)` as a
        plain float rather than a dict. Structured (multi-field) returns pass
        through unchanged.
        """
        result = await self._client.call_tool(name, arguments)
        content = result.structured_content
        if isinstance(content, dict) and set(content) == {"result"}:
            return content["result"]
        return content
