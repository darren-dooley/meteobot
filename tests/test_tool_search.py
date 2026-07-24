"""Tool search: deferred MCP tools stay hidden until the model discovers them.

Driven offline with a `FunctionModel` scripting the two-step discovery dance —
call `search_tools`, then call the discovered tool — against the real demo
server attached in-process and marked deferred. No API key, no network.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from meteobot.mcp_toolbox import server


def _returns(result) -> dict[str, object]:
    return {
        part.tool_name: part.content
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolReturnPart)
    }


async def test_search_discovers_a_deferred_tool_and_makes_it_callable() -> None:
    step = {"n": 0}

    def model(messages, info: AgentInfo) -> ModelResponse:
        if step["n"] == 0:
            step["n"] = 1
            return ModelResponse(
                parts=[ToolCallPart("search_tools", {"queries": ["celsius to fahrenheit"]})]
            )
        if step["n"] == 1:
            step["n"] = 2
            return ModelResponse(
                parts=[ToolCallPart("celsius_to_fahrenheit", {"celsius": 100.0})]
            )
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(model),
        toolsets=[MCPToolset(server).defer_loading()],
        capabilities=[ToolSearch()],
    )
    async with agent:
        result = await agent.run("convert 100C to F")

    returns = _returns(result)
    # search_tools surfaced the conversion tool...
    discovered = returns["search_tools"]["discovered_tools"]
    assert any(t["name"] == "celsius_to_fahrenheit" for t in discovered)
    # ...and the discovered tool then executed (212F for 100C). The MCP toolset
    # unwraps the scalar result to a plain float.
    assert returns["celsius_to_fahrenheit"] == 212.0


async def test_search_with_no_match_reports_nothing_found() -> None:
    # A query that matches no conversion tool returns an empty discovery set
    # rather than erroring, so the model can adjust or give up cleanly.
    def model(messages, info: AgentInfo) -> ModelResponse:
        if not any(
            isinstance(p, ToolReturnPart)
            for m in messages
            for p in getattr(m, "parts", [])
        ):
            return ModelResponse(
                parts=[ToolCallPart("search_tools", {"queries": ["send an email"]})]
            )
        return ModelResponse(parts=[TextPart("no such tool")])

    agent = Agent(
        FunctionModel(model),
        toolsets=[MCPToolset(server).defer_loading()],
        capabilities=[ToolSearch()],
    )
    async with agent:
        result = await agent.run("email someone")

    discovered = _returns(result)["search_tools"]["discovered_tools"]
    assert discovered == []
