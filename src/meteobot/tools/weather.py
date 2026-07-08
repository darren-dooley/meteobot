"""The get_weather tool: geocode a city name, then fetch current conditions.

One composed tool at the level of user intent — the model asks for a city and
gets one structured result; it never sees coordinates or API plumbing.
"""

from __future__ import annotations

from functools import partial
from typing import Any, TypedDict

import httpx

from meteobot.tools.registry import Tool
from meteobot.tools.results import ToolError

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "weather_code",
    "wind_speed_10m",
)

# WMO weather interpretation codes, per the Open-Meteo docs.
WMO_CONDITIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class WeatherReport(TypedDict):
    """Current conditions for one resolved location."""

    location: str
    latitude: float
    longitude: float
    observed_at: str
    temperature_c: float
    feels_like_c: float
    humidity_percent: int
    wind_speed_kmh: float
    conditions: str


def weather_tool(http_client: httpx.AsyncClient) -> Tool:
    """The get_weather declaration, with the shared HTTP client bound in."""
    return Tool(
        name="get_weather",
        description=(
            "Get the current weather for one city by name. "
            "Call once per city when several cities are asked about."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name as the user gave it, e.g. 'London'.",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        handler=partial(get_weather, http_client=http_client),
    )


def _display_name(place: dict[str, Any]) -> str:
    parts = [place["name"], place.get("admin1"), place.get("country")]
    return ", ".join(p for p in parts if p)


async def get_weather(
    city: str, *, http_client: httpx.AsyncClient
) -> WeatherReport | ToolError:
    """Geocode `city` and return its current conditions as one structured result.

    Failures the LLM can act on come back as a structured ToolError, never
    as a raised exception.
    """
    try:
        return await _geocode_and_fetch(city, http_client)
    except httpx.TimeoutException:
        return ToolError(
            error="weather_service_error",
            message=f"The weather service did not respond in time for {city!r}.",
        )
    except httpx.HTTPError:
        return ToolError(
            error="weather_service_error",
            message=f"The weather service failed while looking up {city!r}.",
        )


async def _geocode_and_fetch(
    city: str, http_client: httpx.AsyncClient
) -> WeatherReport | ToolError:
    geo_response = await http_client.get(
        GEOCODING_URL,
        params={"name": city, "count": 5, "language": "en", "format": "json"},
    )
    geo_response.raise_for_status()
    # Captured wire format: the "results" key is absent when nothing matches.
    candidates: list[dict[str, Any]] = geo_response.json().get("results", [])

    # The API fuzzy-matches, so a misspelling still returns candidates; only an
    # exact name match counts as resolved. Candidates are ordered by relevance,
    # so the first exact match is the best one.
    place = next(
        (c for c in candidates if c["name"].casefold() == city.strip().casefold()),
        None,
    )
    if place is None:
        alternatives = list(dict.fromkeys(_display_name(c) for c in candidates))
        return ToolError(
            error="unknown_city",
            message=f"No place named {city!r} was found."
            + (" Did you mean one of these?" if alternatives else ""),
            alternatives=alternatives,
        )

    forecast_response = await http_client.get(
        FORECAST_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": ",".join(CURRENT_VARIABLES),
        },
    )
    forecast_response.raise_for_status()
    current = forecast_response.json()["current"]

    code = current["weather_code"]
    return WeatherReport(
        location=_display_name(place),
        latitude=place["latitude"],
        longitude=place["longitude"],
        observed_at=current["time"],
        temperature_c=current["temperature_2m"],
        feels_like_c=current["apparent_temperature"],
        humidity_percent=current["relative_humidity_2m"],
        wind_speed_kmh=current["wind_speed_10m"],
        conditions=WMO_CONDITIONS.get(code, f"unknown conditions (code {code})"),
    )
