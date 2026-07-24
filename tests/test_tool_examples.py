"""The tool-use-examples helper: example calls appended to a description."""

from __future__ import annotations

from meteobot.tool_examples import with_examples


def test_appends_rendered_example_calls() -> None:
    result = with_examples(
        "Get the current weather for one city.",
        "get_weather",
        [{"city": "London"}, {"city": "San Francisco"}],
    )
    assert "Get the current weather for one city." in result
    assert "Examples:" in result
    assert "get_weather(city='London')" in result
    assert "get_weather(city='San Francisco')" in result


def test_no_examples_returns_the_description_unchanged() -> None:
    assert with_examples("desc", "get_weather", []) == "desc"
