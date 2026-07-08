"""Structured tool results: the JSON-serializable contract between tools and the LLM.

A Tool Error is a failure the LLM can act on (unknown city, weather-service
timeout). It is returned as data, never raised across the tool boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict


class ToolError(TypedDict):
    """A structured failure the model explains conversationally."""

    error: str
    message: str
    alternatives: NotRequired[list[str]]


# What any tool handler resolves to; always JSON-serializable.
type ToolResult = Mapping[str, object]
