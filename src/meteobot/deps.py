"""The dependency-injection payload carried through a PydanticAI run.

A `Deps` value is assembled once at the composition root and passed on every
agent run; tools reach it through PydanticAI's `RunContext`. It preserves v1's
decision to share one pooled httpx client across the concurrent multi-city
fan-out, with its timeout and connect-retry policy configured in one place, and
to make the injected `Settings` reachable wherever a tool needs a tunable.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from meteobot.config import Settings


@dataclass
class Deps:
    """Shared runtime dependencies injected into every tool via `RunContext`."""

    http_client: httpx.AsyncClient
    settings: Settings
