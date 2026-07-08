"""get_weather contract tests, driven through the httpx mock-transport seam.

All canned responses are fixtures captured from the real Open-Meteo APIs
(see tests/fixtures/), so the fakes cannot drift from the wire format.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx

from meteobot.tools.weather import get_weather

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def client_serving(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def open_meteo_handler(request: httpx.Request) -> httpx.Response:
    """Route requests the way the real Open-Meteo hosts would."""
    if request.url.host == "geocoding-api.open-meteo.com":
        return httpx.Response(200, content=fixture("geocoding_london.json"))
    if request.url.host == "api.open-meteo.com":
        return httpx.Response(200, content=fixture("forecast_london.json"))
    raise AssertionError(f"unexpected host: {request.url.host}")


async def test_happy_path_composes_geocoding_and_current_conditions() -> None:
    async with client_serving(open_meteo_handler) as http_client:
        result = await get_weather("London", http_client=http_client)

    # Values come from the captured fixtures: London GB is the top geocoding
    # match; the forecast fixture reports 21.8°C, code 3 (overcast).
    assert result == {
        "location": "London, England, United Kingdom",
        "latitude": 51.50853,
        "longitude": -0.12574,
        "observed_at": "2026-07-05T10:15",
        "temperature_c": 21.8,
        "feels_like_c": 21.2,
        "humidity_percent": 57,
        "wind_speed_kmh": 11.9,
        "conditions": "overcast",
    }
    # The result must be JSON-serializable: it is sent back to the LLM verbatim.
    json.dumps(result)


async def test_unknown_city_returns_structured_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Captured wire format: no "results" key at all when nothing matches.
        assert request.url.host == "geocoding-api.open-meteo.com"
        return httpx.Response(200, content=fixture("geocoding_unknown.json"))

    async with client_serving(handler) as http_client:
        result = await get_weather("Zzyzxqux", http_client=http_client)

    assert result == {
        "error": "unknown_city",
        "message": "No place named 'Zzyzxqux' was found.",
        "alternatives": [],
    }


async def test_misspelled_city_returns_error_with_likely_alternatives() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Captured wire format for "Lundon": fuzzy candidates, no exact match.
        assert request.url.host == "geocoding-api.open-meteo.com"
        return httpx.Response(200, content=fixture("geocoding_misspelled.json"))

    async with client_serving(handler) as http_client:
        result = await get_weather("Lundon", http_client=http_client)

    assert result == {
        "error": "unknown_city",
        "message": "No place named 'Lundon' was found. Did you mean one of these?",
        "alternatives": [
            "Lundong, Guangxi, China",
            "Lundongcun, Guangxi, China",
            "Lundongtun, Guangxi, China",
            "Lundong, Guizhou, China",
        ],
    }


async def test_timeout_returns_structured_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with client_serving(handler) as http_client:
        result = await get_weather("London", http_client=http_client)

    assert result == {
        "error": "weather_service_error",
        "message": "The weather service did not respond in time for 'London'.",
    }


async def test_service_failure_returns_structured_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, content=fixture("geocoding_london.json"))
        return httpx.Response(500, json={"reason": "internal error"})

    async with client_serving(handler) as http_client:
        result = await get_weather("London", http_client=http_client)

    assert result == {
        "error": "weather_service_error",
        "message": "The weather service failed while looking up 'London'.",
    }
