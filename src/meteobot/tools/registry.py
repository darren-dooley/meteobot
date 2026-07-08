"""The tool registry: the extensibility seam.

Each tool is a declaration (name, description, parameter schema, async
handler). The registry is a plain value assembled explicitly at the
composition root and injected into the agent — no import-time
self-registration, no global singleton. The agent consumes exactly two
things: the schema list (handed to the LLM) and the async dispatch function.

Future middleware (per-tool timeouts, retries, tracing, MCP-backed handlers)
attaches at the dispatch seam.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from meteobot.tools.results import ToolError, ToolResult


@dataclass(frozen=True)
class Tool:
    """One tool declaration; `parameters` is a JSON Schema for the arguments."""

    name: str
    description: str
    parameters: dict[str, object]
    handler: Callable[..., Awaitable[ToolResult]]


@dataclass(frozen=True)
class ToolRegistry:
    """A plain value: the toolset one agent can use."""

    tools: tuple[Tool, ...]

    def schemas(self) -> list[dict[str, object]]:
        """Tool schemas in the Responses API function-tool shape."""
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": True,
            }
            for tool in self.tools
        ]

    async def dispatch(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        """Run the named tool with the LLM-supplied arguments."""
        for tool in self.tools:
            if tool.name == name:
                return await tool.handler(**arguments)
        return ToolError(
            error="unknown_tool",
            message=f"No tool named {name!r} is registered.",
        )
