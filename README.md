# meteobot

A natural-language weather CLI. Ask a question, the LLM calls weather APIs, and the answer streams back.

## Quick start

```bash
cp .env.example .env      # paste your OpenAI key
uv run meteobot           # start the REPL
```

Or inline: `OPENAI_API_KEY=sk-... uv run meteobot`

## Folder structure

```
src/meteobot/
  __main__.py     composition root + console entry point
  config.py       settings, .env loading, fail-fast on missing key
  cli.py          REPL loop
  agent.py        streaming agent loop
  llm.py          LLM client adapter
  tools/
    registry.py   tool registry
    weather.py    get_weather tool
    results.py    tool result/error shapes
```

## Tests

```bash
uv run pytest          # offline suite: no key, no network
uv run pytest -m live  # end-to-end test: needs key, hits real APIs
```

## Design decisions

- **Small package**: Focused modules with explicit injection.
- **One streaming path**: Bounded loop with tool-round cap guarantees termination.
- **Client-owned History**: Explicit list passed on every call, enables multi-turn easily.
- **One composed tool**: `get_weather(city)` at user intent level, parallel multi-city calls.
- **Shared httpx client**: Connection pooling, one timeout/retry policy injected.
- **Two-tier errors**: Tool Errors are model input; Infrastructure Errors surface to human.
- **Explicit registry**: Visible in one place, testable in isolation.
- **Model as config**: `gpt-4.1-mini` default, overridable via `METEOBOT_MODEL`.

