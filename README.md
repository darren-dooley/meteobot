# meteobot

A natural-language weather CLI. Ask a question, an LLM calls weather APIs, and the answer streams back. The agent loop, tool dispatch, and streaming are owned by [PydanticAI](https://ai.pydantic.dev); every Turn can be traced in [LangSmith](https://smith.langchain.com) over OpenTelemetry.

## Quick start

```bash
cp .env.example .env      # paste your OpenAI key
uv run meteobot           # start the REPL
```

Or inline: `OPENAI_API_KEY=sk-... uv run meteobot`

## Configuration

All tunables are environment variables (loaded from `.env` without overriding your shell):

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required. The app fails fast with one line if it is missing. |
| `METEOBOT_MODEL` | `gpt-4.1-mini` | OpenAI model, used via the Responses API. |
| `METEOBOT_HTTP_TIMEOUT` | `10.0` | Per-request HTTP timeout, seconds. |
| `METEOBOT_REQUEST_LIMIT` | `6` | Max model requests per Turn (the framework-enforced round cap). |
| `METEOBOT_TOTAL_TOKENS_LIMIT` | `100000` | Per-Turn token ceiling, a second cost guardrail. |
| `METEOBOT_LOG_LEVEL` | `WARNING` | stderr diagnostics level. |
| `METEOBOT_TRACING` | `false` | Turn LangSmith tracing on. Fails open if misconfigured. |
| `LANGSMITH_API_KEY` | — | Needed only when tracing is on. |
| `LANGSMITH_PROJECT` | `meteobot` | LangSmith project name for the traces. |
| `LANGSMITH_OTEL_ENDPOINT` | `https://api.smith.langchain.com/otel` | OTLP/HTTP endpoint. |

## Tracing

With `METEOBOT_TRACING=true` and a `LANGSMITH_API_KEY`, PydanticAI's native instrumentation emits OpenTelemetry spans that ship to LangSmith over OTLP/HTTP — one searchable trace per Turn, with its Tool Calls, latency, and tokens. There is no LangChain dependency. Tracing is fail-open: with tracing on but no key, or if the exporter cannot be built, meteobot logs a warning and runs a normal local session.

## Folder structure

```
src/meteobot/
  __main__.py     composition root + console entry point
  config.py       settings, .env loading, fail-fast on missing key
  cli.py          REPL loop (control flow + terminal I/O edge)
  agent.py        PydanticAI agent construction + the streaming Turn adapter
  deps.py         injected run dependencies (shared httpx client + settings)
  tracing.py      LangSmith-over-OpenTelemetry wiring (fail-open)
  tools/
    weather.py    get_weather tool + fetch_weather logic + typed result models
```

## Tests

```bash
uv run pytest          # offline suite: no key, no network
uv run pytest -m live  # end-to-end test: needs key, hits real APIs
```

The agent loop is tested offline through PydanticAI's `TestModel`/`FunctionModel` (scripted Tool Rounds, concurrency, the usage-limit trip); the weather tool is tested against captured Open-Meteo fixtures through an httpx mock transport; a trace test asserts span emission via an in-memory OpenTelemetry exporter. The LLM's judgement beyond the live smoke check is evals territory and out of scope here.

## Design decisions

- **PydanticAI, not LangGraph**: the Turn is a linear single-tool loop, so the framework owns the model → Tool Call → model cycle, streaming, typed tools, and usage limits; a graph/checkpointer engine would be over-sized. LangGraph would earn its place only with genuinely graph-shaped control flow.
- **Typed union tool output**: `get_weather` returns `WeatherResult | WeatherError`; a failed city is structured data the model explains, and its siblings survive.
- **Two-tier errors**: Tool Errors are model input; Infrastructure Errors (LLM API down, bad credentials) surface to the human at the REPL and the session survives.
- **Usage limits as policy**: a request limit and a token limit, enforced by the framework, replace a hand-counted round cap.
- **Client-owned History**: PydanticAI's message list is carried across Turns and stays inspectable; no server-side conversation state.
- **Shared httpx client**: one pooled client, one timeout/connect-retry policy, injected into tools and the OpenAI SDK alike.
- **Observability as traces**: every Turn is a LangSmith trace, not just ephemeral stderr lines.
