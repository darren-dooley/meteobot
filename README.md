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
| `METEOBOT_TOOL_SEARCH` | `true` | Attach the demo MCP server as a deferred, searchable toolset. |
| `METEOBOT_CODE_EXEC` | `true` | Enable the `run_python` code-execution tool. |
| `METEOBOT_EXECUTOR` | `docker` | Code-exec sandbox: `docker` (isolated) or `inprocess` (no boundary). |
| `METEOBOT_SANDBOX_IMAGE` | `python:3.12-slim` | Image for the Docker executor. |
| `METEOBOT_CODE_EXEC_TIMEOUT` | `30.0` | Wall-clock ceiling for one code-exec run, seconds. |
| `METEOBOT_TRACING` | `false` | Turn LangSmith tracing on. Fails open if misconfigured. |
| `LANGSMITH_API_KEY` | — | Needed only when tracing is on. |
| `LANGSMITH_PROJECT` | `meteobot` | LangSmith project name for the traces. |
| `LANGSMITH_OTEL_ENDPOINT` | `https://api.smith.langchain.com/otel` | OTLP/HTTP endpoint. |

## Advanced tool use

meteobot ships a provider-agnostic take on Anthropic's [advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) patterns, built on PydanticAI so they work on the OpenAI model:

- **Tool search** — a small MCP server (`mcp_toolbox.py`, unit conversions) is attached as a *deferred* toolset: its tools stay out of the prompt until the model calls `search_tools` to discover them. On `gpt-4.1-mini` this uses PydanticAI's local keyword search (native server-side search is Claude / GPT-5.4+ only), so there is no provider lock-in.
- **Programmatic tool calling** — a `run_python` tool runs a model-written script and returns only what it prints, so bulk intermediate data never enters the context window. The script calls MCP tools as ordinary `await` functions and can `asyncio.gather` them.
- **Tool-use examples** — few-shot example calls are embedded in tool descriptions (2.14 has no `input_examples` field), demonstrating the expected argument shapes.

The code-execution sandbox is a swappable `Executor` seam. The default `DockerExecutor` runs the script in an ephemeral `--network none --cap-drop ALL` container; the model's code reaches MCP tools only over a stdio RPC bridge back to the trusted parent process, so a network-isolated container can still call tools that need the network. When Docker is unavailable the app logs a reason and falls back to the unsandboxed `InProcessExecutor` (fine for a local single-user REPL). Both features can be turned off independently via the env vars above.

## Tracing

With `METEOBOT_TRACING=true` and a `LANGSMITH_API_KEY`, PydanticAI's native instrumentation emits OpenTelemetry spans that ship to LangSmith over OTLP/HTTP — one searchable trace per Turn, with its Tool Calls, latency, and tokens. There is no LangChain dependency. Tracing is fail-open: with tracing on but no key, or if the exporter cannot be built, meteobot logs a warning and runs a normal local session.

## Folder structure

```
src/meteobot/
  __main__.py     composition root + console entry point
  config.py       settings, .env loading, fail-fast on missing key
  cli.py          REPL loop (control flow + terminal I/O edge)
  agent.py        PydanticAI agent construction + the streaming Turn adapter
  deps.py         injected run dependencies (shared httpx client + settings + sandbox)
  tracing.py      LangSmith-over-OpenTelemetry wiring (fail-open)
  features.py     assembles tool search + code execution and their async lifecycle
  tool_examples.py  helper: fold few-shot example calls into a tool description
  mcp_toolbox.py  the demo MCP server (unit conversions) over stdio
  tools/
    weather.py    get_weather tool + fetch_weather logic + typed result models
  codeexec/
    mcp_tools.py       trusted-side FastMCP client (lists + calls MCP tools)
    executor.py        Executor seam, ExecResult, Sandbox, InProcessExecutor
    docker_executor.py default sandbox: network-isolated container + RPC bridge
    runner.py          in-container entry point (stdlib only; never runs in-process)
    tool.py            the run_python PydanticAI tool + generated description
```

## Tests

```bash
uv run pytest          # offline suite: no key, no network
uv run pytest -m live  # end-to-end test: needs key, hits real APIs
```

The agent loop is tested offline through PydanticAI's `TestModel`/`FunctionModel` (scripted Tool Rounds, concurrency, the usage-limit trip); the weather tool is tested against captured Open-Meteo fixtures through an httpx mock transport; a trace test asserts span emission via an in-memory OpenTelemetry exporter. Tool search and code execution are tested offline against the demo MCP server attached in-process (a `FunctionModel` scripts the search-then-call dance; the in-process executor runs real snippets); the Docker executor has its own module that auto-skips when Docker is unavailable. The LLM's judgement beyond the live smoke check is evals territory and out of scope here.

## Design decisions

- **PydanticAI, not LangGraph**: the Turn is a linear single-tool loop, so the framework owns the model → Tool Call → model cycle, streaming, typed tools, and usage limits; a graph/checkpointer engine would be over-sized. LangGraph would earn its place only with genuinely graph-shaped control flow.
- **Typed union tool output**: `get_weather` returns `WeatherResult | WeatherError`; a failed city is structured data the model explains, and its siblings survive.
- **Two-tier errors**: Tool Errors are model input; Infrastructure Errors (LLM API down, bad credentials) surface to the human at the REPL and the session survives.
- **Usage limits as policy**: a request limit and a token limit, enforced by the framework, replace a hand-counted round cap.
- **Client-owned History**: PydanticAI's message list is carried across Turns and stays inspectable; no server-side conversation state.
- **Shared httpx client**: one pooled client, one timeout/connect-retry policy, injected into tools and the OpenAI SDK alike.
- **Provider-agnostic advanced tool use**: the blog's features are Anthropic beta APIs, but the same capabilities are built on PydanticAI (`ToolSearch`, `MCPToolset(...).defer_loading()`, a local `run_python` tool) so they run on the OpenAI model with no provider swap.
- **Sandbox as a swappable seam**: `run_python` depends only on the `Executor` interface; Docker is the default, in-process is the fallback and test double, and a stronger boundary (e.g. a hosted microVM service) can drop in without touching the agent or MCP wiring. Untrusted model code never holds the MCP client — it can only ask the trusted parent to make a named call.
- **Observability as traces**: every Turn is a LangSmith trace, not just ephemeral stderr lines.
