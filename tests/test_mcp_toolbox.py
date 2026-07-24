"""The demo MCP server's conversion tools, exercised through a FastMCP client.

The server object is a real FastMCP instance, so a client connects to it
in-process — no subprocess, no ports — and the same tools the agent discovers
are the ones asserted here.
"""

from __future__ import annotations

import pytest
from fastmcp import Client

from meteobot.mcp_toolbox import server


async def call(name: str, **arguments: float) -> float:
    async with Client(server) as client:
        result = await client.call_tool(name, arguments)
    return result.structured_content["result"]


async def test_server_advertises_the_conversion_tools() -> None:
    async with Client(server) as client:
        names = {t.name for t in await client.list_tools()}
    assert names == {
        "celsius_to_fahrenheit",
        "fahrenheit_to_celsius",
        "celsius_to_kelvin",
        "kmh_to_mph",
        "mph_to_kmh",
    }


@pytest.mark.parametrize(
    ("name", "arg", "value", "expected"),
    [
        ("celsius_to_fahrenheit", "celsius", 21.8, 71.24),
        ("celsius_to_fahrenheit", "celsius", 0.0, 32.0),
        ("fahrenheit_to_celsius", "fahrenheit", 32.0, 0.0),
        ("celsius_to_kelvin", "celsius", 0.0, 273.15),
        ("kmh_to_mph", "kmh", 1.609344, 1.0),
        ("mph_to_kmh", "mph", 1.0, 1.609344),
    ],
)
async def test_conversions(name: str, arg: str, value: float, expected: float) -> None:
    assert await call(name, **{arg: value}) == pytest.approx(expected)
