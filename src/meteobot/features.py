"""Assembly of the advanced tool-use features, with their async lifecycle.

`build_advanced_tooling` is an async context manager the composition root enters
inside its `async with` block. It turns settings into the parts `build_agent`
needs — extra tools, deferred toolsets, capabilities — and the `Sandbox` that
goes on `Deps`, opening and closing the code-execution MCP client for the
lifetime of the app.

Two independent features are wired here, each gated by its own setting:

- Tool search: the demo MCP server attached as a deferred `MCPToolset` plus the
  `ToolSearch` capability, so its conversion tools stay out of the prompt until
  the model searches for them. The agent manages this toolset's connection.
- Code execution: a second, separately-managed client to the same server backs
  the `run_python` tool, whose executor is Docker by default and falls back to
  in-process when Docker is unavailable (logged, never fatal).

Keeping this out of `__main__` leaves the composition root a flat, readable
sequence and puts the one piece of real conditional lifecycle in a tested unit.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

from pydantic_ai import Tool
from pydantic_ai.capabilities import AgentCapability, ToolSearch
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

from meteobot.codeexec import InProcessExecutor, McpTools, Sandbox
from meteobot.codeexec.docker_executor import DockerExecutor, preflight
from meteobot.codeexec.executor import Executor
from meteobot.codeexec.mcp_tools import server_script_path
from meteobot.codeexec.tool import build_run_python
from meteobot.config import Settings
from meteobot.deps import Deps

logger = logging.getLogger(__name__)


@dataclass
class AdvancedTooling:
    """The parts the composition root folds into the agent and its deps."""

    extra_tools: list[Tool[Deps] | Callable[..., object]] = field(default_factory=list)
    toolsets: list[AbstractToolset[Deps]] = field(default_factory=list)
    capabilities: list[AgentCapability[Deps]] = field(default_factory=list)
    sandbox: Sandbox | None = None

    @property
    def enabled(self) -> bool:
        """True if any feature is active — drives the capability instructions."""
        return bool(self.extra_tools or self.toolsets)


async def _build_executor(settings: Settings) -> Executor:
    """Pick the code-execution executor, falling back off Docker when unusable."""
    if settings.executor == "inprocess":
        return InProcessExecutor(timeout_seconds=settings.code_exec_timeout_seconds)
    reason = await preflight()
    if reason is not None:
        logger.warning(
            "Docker executor unavailable (%s); falling back to in-process "
            "execution, which is not sandboxed.",
            reason,
        )
        return InProcessExecutor(timeout_seconds=settings.code_exec_timeout_seconds)
    return DockerExecutor(
        image=settings.sandbox_image,
        timeout_seconds=settings.code_exec_timeout_seconds,
    )


@asynccontextmanager
async def build_advanced_tooling(
    settings: Settings,
) -> AsyncIterator[AdvancedTooling]:
    """Assemble tool search and code execution for the app's lifetime."""
    tooling = AdvancedTooling()

    if settings.tool_search_enabled:
        # The agent owns this toolset's connection; marking it deferred hides its
        # tools until `search_tools` discovers them. `ToolSearch` provides the
        # local keyword search that backs discovery on non-native providers.
        mcp_toolset = MCPToolset(str(server_script_path()))
        tooling.toolsets.append(mcp_toolset.defer_loading())
        tooling.capabilities.append(ToolSearch())

    async with AsyncExitStack() as stack:
        if settings.code_exec_enabled:
            # A second client to the same server, managed here rather than by the
            # agent, because the executor calls it outside any agent run.
            tools = await stack.enter_async_context(
                McpTools.connect(server_script_path())
            )
            executor = await _build_executor(settings)
            tooling.sandbox = Sandbox(executor=executor, tools=tools)
            tooling.extra_tools.append(build_run_python(tools))
        yield tooling


def merge_capabilities(
    base: Sequence[AgentCapability[Deps]], extra: Sequence[AgentCapability[Deps]]
) -> list[AgentCapability[Deps]]:
    """Combine instrumentation capabilities with the advanced-tooling ones."""
    return [*base, *extra]
