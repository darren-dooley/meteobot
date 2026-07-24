"""The get_weather tool: geocode a city name, then fetch current conditions.

One composed tool at the level of user intent — the model asks for a city and
gets one structured result; it never sees coordinates or API plumbing.

`fetch_weather` holds all the logic and is the tested unit: a plain async
function over the shared httpx client, returning a typed `WeatherResult |
WeatherError` union. `get_weather` is the thin PydanticAI adapter that reaches
the shared client through the run's dependency-injection context and delegates.
Every failure the model can act on comes back as a `WeatherError` value; the
tool never raises, so one failing city can never abort the Turn or its siblings.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import BaseModel
from pydantic_ai import RunContext

from meteobot.deps import Deps

# The closed set of Tool Error codes get_weather can return. Named so the model
# and the tests share one explicit vocabulary rather than free-form strings.
WeatherErrorCode = Literal[
    "unknown_city", "weather_service_error", "tool_execution_error"
]

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


class WeatherResult(BaseModel):
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


class WeatherError(BaseModel):
    """A structured failure the model explains conversationally (a Tool Error).

    Returned as data, never raised across the tool boundary, so the model can
    tell the user what went wrong and a failed city leaves its siblings intact.
    """

    error: WeatherErrorCode
    message: str
    alternatives: list[str] = []


async def get_weather(
    ctx: RunContext[Deps], city: str
) -> WeatherResult | WeatherError:
    """Get the current weather for one city by name.

    Call once per city when several cities are asked about.
    """
    return await fetch_weather(city, ctx.deps.http_client)


def _display_name(place: dict[str, Any]) -> str:
    parts = [place["name"], place.get("admin1"), place.get("country")]
    return ", ".join(p for p in parts if p)


async def fetch_weather(
    city: str, http_client: httpx.AsyncClient
) -> WeatherResult | WeatherError:
    """Geocode `city` and return its current conditions as one structured result.

    Every failure the LLM can act on — an unknown city, a weather-service
    timeout or outage, or any unexpected error looking up this one city — comes
    back as a structured WeatherError, never a raised exception. That keeps one
    failing city from aborting the Turn or taking down its concurrent siblings.
    """
    try:
        return await _geocode_and_fetch(city, http_client)
    except httpx.TimeoutException:
        return WeatherError(
            error="weather_service_error",
            message=f"The weather service did not respond in time for {city!r}.",
        )
    except httpx.HTTPError:
        return WeatherError(
            error="weather_service_error",
            message=f"The weather service failed while looking up {city!r}.",
        )
    except Exception:
        # A bug looking up one city is still just that city's failure: report it
        # as a Tool Error so the model can move on rather than aborting the Turn.
        return WeatherError(
            error="tool_execution_error",
            message=f"Something went wrong looking up the weather for {city!r}.",
        )


async def _geocode_and_fetch(
    city: str, http_client: httpx.AsyncClient
) -> WeatherResult | WeatherError:
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
        return WeatherError(
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
    return WeatherResult(
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
