"""A small, hermetic MCP server: unit conversions the weather answer can use.

This is the demo corpus for the two advanced tool-use features. It is a real
[Model Context Protocol](https://modelcontextprotocol.io) server (FastMCP over
stdio), not a stub: meteobot connects to it the same way it would connect to any
third-party MCP server, so the wiring you see here is the wiring a real
integration uses.

The tools are deliberately pure and deterministic — temperature and speed
conversions over floats, no network, no clock, no state. That keeps the demo
hermetic (tests need no live server behind it) and gives the model a corpus that
pairs naturally with `get_weather`, which reports °C and km/h: "what's that in
Fahrenheit and mph?" becomes a tool-search discovery plus a conversion call.

Run as a stdio server with `python -m meteobot.mcp_toolbox`; that is exactly how
`MCPToolset(server_script_path())` and the code-execution client spawn it.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("converter")


@server.tool()
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature in degrees Celsius to degrees Fahrenheit."""
    return celsius * 9 / 5 + 32


@server.tool()
def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert a temperature in degrees Fahrenheit to degrees Celsius."""
    return (fahrenheit - 32) * 5 / 9


@server.tool()
def celsius_to_kelvin(celsius: float) -> float:
    """Convert a temperature in degrees Celsius to kelvin."""
    return celsius + 273.15


@server.tool()
def kmh_to_mph(kmh: float) -> float:
    """Convert a speed in kilometres per hour to miles per hour."""
    return kmh / 1.609344


@server.tool()
def mph_to_kmh(mph: float) -> float:
    """Convert a speed in miles per hour to kilometres per hour."""
    return mph * 1.609344


def main() -> None:
    """Entry point for `python -m meteobot.mcp_toolbox`: serve over stdio."""
    server.run()


if __name__ == "__main__":
    main()
