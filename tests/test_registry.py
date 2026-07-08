"""Tool registry contract: schema list + async dispatch over explicit declarations.

Fake tools are supplied by constructing a registry value directly — no
monkeypatching, no global registration.
"""

from __future__ import annotations

from meteobot.tools.registry import Tool, ToolRegistry
from meteobot.tools.results import ToolResult

ECHO_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


async def echo_handler(text: str) -> ToolResult:
    return {"echoed": text}


def echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echoes the text back.",
        parameters=ECHO_PARAMETERS,
        handler=echo_handler,
    )


def test_schemas_render_declarations_in_responses_api_shape() -> None:
    registry = ToolRegistry(tools=(echo_tool(),))
    assert registry.schemas() == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echoes the text back.",
            "parameters": ECHO_PARAMETERS,
            "strict": True,
        }
    ]


async def test_dispatch_runs_the_named_handler_with_arguments() -> None:
    registry = ToolRegistry(tools=(echo_tool(),))
    result = await registry.dispatch("echo", {"text": "hello"})
    assert result == {"echoed": "hello"}


async def test_dispatch_of_unknown_tool_returns_structured_error() -> None:
    registry = ToolRegistry(tools=(echo_tool(),))
    result = await registry.dispatch("does_not_exist", {})
    assert result == {
        "error": "unknown_tool",
        "message": "No tool named 'does_not_exist' is registered.",
    }


async def test_get_weather_registers_and_dispatches_through_the_registry() -> None:
    import httpx

    from meteobot.tools.weather import weather_tool

    from test_weather_tool import client_serving, open_meteo_handler

    async with client_serving(open_meteo_handler) as http_client:
        registry = ToolRegistry(tools=(weather_tool(http_client),))

        (schema,) = registry.schemas()
        assert schema["name"] == "get_weather"
        parameters = schema["parameters"]
        assert isinstance(parameters, dict)
        assert parameters["required"] == ["city"]

        result = await registry.dispatch("get_weather", {"city": "London"})

    assert isinstance(result, dict)
    assert result["location"] == "London, England, United Kingdom"
    assert result["temperature_c"] == 21.8
