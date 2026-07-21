"""LangSmith tracing over OpenTelemetry — the production observability path.

PydanticAI's native instrumentation emits OpenTelemetry spans; this module
points an OTLP/HTTP exporter at LangSmith's OTel endpoint and hands the wiring
back as a PydanticAI instrumentation capability plus the `TracerProvider` that
owns the exporter (so the composition root can flush it on shutdown). No
LangChain dependency and no double-instrumentation.

Tracing is fail-open by design: it is off unless explicitly enabled, a missing
LangSmith key or any error building the exporter degrades to a warning and a
normal local session, and the API key travels only in the request headers —
never in a span attribute — so turning tracing on is safe.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import InstrumentationSettings
from pydantic_ai.capabilities import AgentCapability, Instrumentation

from meteobot.config import Settings
from meteobot.deps import Deps

logger = logging.getLogger(__name__)

# LangSmith documents its OTel collector as the base `.../otel` URL, which the
# OTel SDK's env-var path expands to `.../otel/v1/traces`. The exporter's
# `endpoint=` constructor argument is used verbatim, though, so we append the
# traces signal path ourselves — configuring the base URL and posting to `/otel`
# is a 403.
_TRACES_SIGNAL_PATH = "/v1/traces"


def _traces_endpoint(base: str) -> str:
    """Resolve the configured base URL to the OTLP traces endpoint."""
    trimmed = base.rstrip("/")
    if trimmed.endswith(_TRACES_SIGNAL_PATH):
        return trimmed
    return trimmed + _TRACES_SIGNAL_PATH


def build_instrumentation(
    settings: Settings,
) -> tuple[Sequence[AgentCapability[Deps]], TracerProvider | None]:
    """Return the instrumentation capabilities and the owning TracerProvider.

    Empty capabilities and `None` mean "run untraced": either tracing is off, or
    it is on but misconfigured, in which case this logs a warning rather than
    failing the session. The returned provider, when present, must be
    `shutdown()` on exit to flush the batch exporter.
    """
    if not settings.tracing_enabled:
        return (), None

    if settings.langsmith_api_key is None:
        logger.warning(
            "METEOBOT_TRACING is on but LANGSMITH_API_KEY is not set; "
            "continuing without tracing."
        )
        return (), None

    try:
        exporter = OTLPSpanExporter(
            endpoint=_traces_endpoint(settings.langsmith_endpoint),
            headers={
                "x-api-key": settings.langsmith_api_key,
                "Langsmith-Project": settings.langsmith_project,
            },
        )
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception:
        logger.warning(
            "failed to configure LangSmith tracing; continuing without it.",
            exc_info=True,
        )
        return (), None

    capability = Instrumentation(
        settings=InstrumentationSettings(tracer_provider=provider)
    )
    logger.info(
        "tracing enabled: exporting spans to %s (project %r)",
        settings.langsmith_endpoint,
        settings.langsmith_project,
    )
    return (capability,), provider
