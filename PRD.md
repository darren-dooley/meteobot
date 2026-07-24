---
title: meteobot v2 — productionised on PydanticAI + LangSmith
labels: [ready-for-agent]
---

# PRD: meteobot v2 (PydanticAI + LangSmith)

Vocabulary used below is defined in `CONTEXT.md` (Turn, Tool Round, Tool Call, History, Tool Error, Infrastructure Error).

This is a rewrite of meteobot's core, not an incremental change. v1 hand-wrote the agent loop, an `LLMClient` Protocol seam, and a bespoke `ToolRegistry`. v2 replaces all three with [PydanticAI](https://ai.pydantic.dev) so the loop is a maintained dependency rather than owned code, and instruments every Turn in [LangSmith](https://smith.langchain.com) over OpenTelemetry. The user-facing behaviour of v1 is preserved exactly; the machinery underneath changes.

## Problem Statement

meteobot v1 works, but its agent loop, LLM adapter, and tool registry are all hand-written. That is fine as a reference POC and wrong as a production posture: the loop, streaming accumulation, round cap, tool dispatch, and error capture are meteobot's code to maintain and get right, when a mature framework already owns exactly that surface. There is also no production-grade observability — the only window into the probabilistic agent loop is level-gated `logging` lines on stderr, which do not survive as traces, cannot be searched, and give no token or latency aggregation across Turns. To run this as a real assistant, and to be able to answer "what did the model actually do on that Turn?" after the fact, the orchestration should be a maintained dependency and every Turn should be a searchable trace.

## Solution

meteobot v2 keeps the same interactive terminal experience — ask a plain-English weather question, get one live streamed answer, multi-turn follow-ups, survives bad input and provider failure, exits on `quit`/`exit` — but rebuilt on PydanticAI. A single PydanticAI `Agent` owns the Turn: it runs the model → Tool Call → model cycle, validates Tool Call arguments and tool outputs against Pydantic models, streams the final answer token-by-token, and enforces a usage limit in place of the hand-counted round cap. The `get_weather` tool becomes a typed async function reached through the framework's dependency-injection context. Every Turn emits OpenTelemetry spans that ship to LangSmith, giving per-Turn, per-Tool-Call, per-token traces with latency and cost. Setup is unchanged in spirit: clone, provide an OpenAI key (and optionally a LangSmith key), run one command.

## User Stories

### Preserved user-facing behaviour (from v1)

1. As a terminal user, I want to ask weather questions in natural language, so that I don't have to learn a query syntax.
2. As a terminal user, I want the assistant to fetch live weather data when my question needs it, so that answers reflect current conditions rather than the model's training data.
3. As a terminal user, I want to mention several cities in one message and get one unified answer, so that I don't have to ask city by city.
4. As a terminal user, I want multi-city lookups to run concurrently, so that three cities don't take three times as long.
5. As a terminal user, I want the answer to appear token-by-token as it is generated, so that I see progress immediately.
6. As a terminal user, I want follow-up questions to use conversation context, so that "what about tomorrow?" or "and in Paris?" work naturally.
7. As a terminal user, I want a misspelled or unknown city explained conversationally, with likely alternatives when the name is ambiguous, so that I can correct myself.
8. As a terminal user, I want one invalid city to leave the other cities' results intact, so that a single typo doesn't spoil the whole question.
9. As a terminal user, I want a weather-service timeout or failure for one city reported in plain language within the answer, so that I know what happened without a stack trace.
10. As a terminal user, I want the session to survive an LLM provider outage or network failure with a one-line message, so that a transient failure doesn't kill my conversation.
11. As a terminal user, I want questions that don't need weather data answered directly without spurious lookups.
12. As a terminal user, I want to exit by typing `quit` or `exit`.
13. As a terminal user, I want arrow-key line editing and in-session input history.
14. As a terminal user, I want answers in plain text suited to a terminal.
15. As a terminal user, I want a clear startup message when my API key is missing, so that I can fix my environment first.
16. As a terminal user, I want a runaway sequence of Tool Rounds cut off with an honest message, so that a confused model can't hang my session or burn credits.
17. As a new user, I want to clone the repo, provide a key via `.env` or an environment variable, and start with one documented command.

### New in v2 — framework and observability

18. As the maintainer, I want the Turn's model → Tool Call → model loop owned by PydanticAI, so that streaming, tool dispatch, and iteration control are a maintained dependency rather than meteobot's code.
19. As the maintainer, I want the round cap expressed as a PydanticAI usage limit, so that "won't hang or burn credits" is a declared policy the framework enforces, not a hand-counted loop variable.
20. As the maintainer, I want `get_weather` to be a typed async function whose arguments and result are validated by Pydantic, so that malformed Tool Calls are rejected at the boundary before my code runs.
21. As the maintainer, I want the Tool Error tier expressed as a typed union result (`WeatherResult | WeatherError`), so that a failed city is structured data the model explains and my tests assert on, with siblings preserved.
22. As the maintainer, I want the Infrastructure Error tier still caught at the REPL, so that a provider outage or bad key surfaces as one friendly line and the REPL survives.
23. As an operator, I want every Turn to emit OpenTelemetry spans that reach LangSmith, so that I can see each Turn, its Tool Calls, arguments, latency, tokens, and cost as a searchable trace instead of ephemeral stderr lines.
24. As an operator, I want tracing to be toggleable by configuration and to fail open, so that a missing LangSmith key or an unreachable collector degrades to a normal local session rather than breaking the app.
25. As an operator, I want no secrets (API keys) present in trace payloads or span attributes, so that turning on tracing is safe.
26. As a developer, I want all tunables — model, HTTP timeout, usage limits, log level, LangSmith key/endpoint/project, tracing toggle — in one frozen injected `Settings` object with environment overrides and fail-fast validation, so that operational knobs are discoverable and a bad value names itself.
27. As a developer, I want the History carried as PydanticAI's message list and still fully client-owned and inspectable, so that "what did the model see?" is answerable and multi-turn falls out for free.
28. As a developer, I want the OpenAI Responses API preserved via PydanticAI's `OpenAIResponsesModel`, so that v1's deliberate choice (dependable parallel function calling, no reasoning preamble) carries forward.
29. As a developer, I want to test the whole agent loop offline with PydanticAI's `TestModel`/`FunctionModel`, so that I can verify concurrent Tool Call dispatch, History growth, the usage-limit trip, and direct-answer Turns without an API key or spend.
30. As a developer, I want the weather tool tested against captured Open-Meteo wire fixtures through the shared client's mock transport, so that my fakes can't drift from the real format.
31. As a developer, I want one opt-in live end-to-end test against both real APIs, so that I can confirm the integration genuinely works.
32. As a developer extending meteobot, I want to add a tool by writing one typed async function and registering it at the composition root, so that the agent construction is the only place that changes.

## Implementation Decisions

- **Framework choice — PydanticAI, not LangGraph.** The Turn is a linear single-tool loop; a graph/checkpointer engine would be over-sized machinery for it. PydanticAI is the right-sized tool: it owns the model → Tool Call → model loop, typed tools, streaming, and usage limits, which is exactly the surface v1 hand-wrote. LangGraph is deliberately out of scope (see Out of Scope) and the rationale is part of the design's defensible story.

- **Agent construction.** A single PydanticAI `Agent` is built at the composition root with `OpenAIResponsesModel` (model `gpt-4.1-mini`, env-overridable), the `get_weather` tool, an instructions/system prompt equivalent to v1's, and `instrument=True`. The bespoke agent loop (`agent.py`) and the `LLMClient` Protocol + `OpenAIResponsesLLM` adapter (`llm.py`) are removed; PydanticAI's own `Model` abstraction is the new seam.

- **Tool shape — PydanticAI idiom.** `ToolRegistry` and the `Tool` dataclass are removed. `get_weather` becomes a typed async function registered via the agent's tool list at the composition root; PydanticAI generates the JSON schema from the type hints. Extensibility is preserved as "add one typed function and register it at assembly," at the same single composition point v1 used. It remains one composed tool at the level of user intent (geocode + current-conditions fetch inside the tool); the model never sees coordinates.

- **Dependency injection.** A `Deps` dataclass carries the shared `httpx.AsyncClient` and the `Settings`, passed on the agent run and reached inside `get_weather` through PydanticAI's `RunContext`. This preserves v1's "one shared pooled client, timeout and connect-retry configured once, injected into tools" decision. The httpx client is still created once at the composition root inside an `async with`.

- **Tool Error tier — typed union output.** `get_weather` returns `WeatherResult | WeatherError`, both Pydantic models (replacing the `WeatherReport`/`ToolError` TypedDicts). The model receives the structured error in the tool result and explains it conversationally; per-call capture inside the tool means one failing city never takes down its siblings. `ModelRetry` is deliberately not used here — an unknown city is a fact to report, not a prompt to retry, and retrying would burn a round for the wrong reason.

- **Infrastructure Error tier — unchanged routing.** Failures only the user can act on (LLM API down, bad credentials, transport failure) escape the agent run and are caught at the REPL: one friendly line, the Turn is abandoned, the REPL survives. Never shown to the model.

- **Runtime guardrails.** The hand-counted `tool_round_cap` is replaced by PydanticAI `UsageLimits` (a request limit, plus a token limit), enforced by the framework; when the limit trips, the Turn ends with an honest message to the user, matching v1's cap behaviour. Per-tool timeout and connect-retry remain a property of the shared httpx client (httpx does not retry by default; connect-only retry preserved). Mechanical retries stay confined to the transport/SDK layer; the model is allowed to re-plan after a Tool Error but is never the retry mechanism.

- **Conversation state / History.** Client-owned PydanticAI message history: the message list from one run is passed as `message_history` into the next, giving multi-turn follow-ups for free and keeping the record fully inspectable client-side. No server-side conversation state.

- **Rendering / REPL.** `cli.py`'s structure is kept: `run_repl` (the tested control-flow and error-routing loop) and `start_repl` (the untested terminal-I/O edge — worker-thread blocking read, `readline` line editing, flushed per-delta writes). Only the injected `run_turn` adapter changes: it drives the PydanticAI agent's streaming run and forwards text deltas to the existing renderer callback. The agent stays presentation-blind. It remains a local interactive REPL — not a server; "deploy" here means installable/runnable, not hosted.

- **Configuration.** The frozen `Settings` object is extended: existing fields (OpenAI key with fail-fast message, model, HTTP timeout, log level) plus usage-limit tunables, LangSmith API key, OTLP endpoint, project name, and a tracing on/off toggle. `.env` loaded via python-dotenv without overriding exported variables; invalid tunables fail fast naming the offending variable. Tracing config is optional: absent LangSmith key with tracing on degrades to a warning and a normal local session, never a crash.

- **LangSmith over OpenTelemetry.** PydanticAI's native instrumentation (`instrument=True`) emits OTel spans; an OTLP/HTTP exporter is pointed at LangSmith's OTel endpoint (`https://api.smith.langchain.com/otel`) with the API key and project supplied via headers. No LangChain dependency and no double-instrumentation. This is the production observability path; the v1 stderr `logging` lines may remain as an offline fallback but are no longer the primary trace. Secrets are never placed in span attributes.

## Testing Decisions

- **What makes a good test here.** Assert on external behaviour at a seam — the typed result a tool returns, the sequence of Tool Calls the model seam is driven to make, the History/message shape after a Turn — never on framework internals or private helpers. Fakes stand in for vendors; the code under test (tool logic, REPL control flow, config parsing, agent wiring) is always real.

- **Model seam — PydanticAI `TestModel`/`FunctionModel`.** The agent-loop tests inject a `TestModel` or a `FunctionModel` where production injects `OpenAIResponsesModel`, replaying scripted model behaviour: concurrent Tool Call dispatch across multiple cities, History/message growth across a Turn, the `UsageLimits` trip surfacing the honest message, and a direct-answer Turn with no Tool Call. This replaces v1's hand-rolled fake `LLMClient`; it is the highest seam for loop mechanics and the same seam count as before.

- **HTTP transport seam — unchanged.** The shared injected `httpx.AsyncClient` accepts a `MockTransport`, so `get_weather` runs real code against canned Open-Meteo responses: happy path, unknown/misspelled city → structured `WeatherError` with alternatives, timeout → structured `WeatherError`.

- **Realistic fixtures.** The Open-Meteo geocoding/forecast fixtures carry over unchanged (captured from the real API). Scripted model behaviour is expressed through PydanticAI's test models rather than replayed raw Responses wire events; where captured Responses fixtures still encode a useful shape they inform the scripts.

- **Live e2e test.** One opt-in test drives the real `Agent` against both real APIs (multi-city question → streamed answer), marked so it runs only on explicit request with a key present, excluded from the default offline suite.

- **Trace emission.** A test asserts that running a Turn with tracing enabled produces the expected spans (Turn → Tool Call), captured via an in-memory OTel span exporter — verifying instrumentation is wired without contacting LangSmith.

- **Deliberately untested.** The terminal I/O edge in `start_repl` (fake-TTY fiddliness, near-zero signal) and the LLM's judgment beyond the e2e smoke check (evals territory, named as such in the README).

## Out of Scope

- **LangGraph.** Considered and rejected for this problem: the Turn is a linear single-tool loop, so a graph/checkpointer engine adds machinery without a job. It would earn its place only if meteobot grew genuinely graph-shaped control flow (branching, durable resume across sessions, human-in-the-loop pauses). Recorded here as a deliberate decision, not an omission.
- **Packaging & CI hardening.** No Dockerfile, no GitHub Actions pipeline in this PRD. The console-script entry point and committed uv lockfile from v1 remain; broader shipping infrastructure is deferred.
- **Server / hosted deployment.** meteobot stays a local REPL CLI. No HTTP API, no LangGraph Platform, no long-running service.
- **Multi-provider abstraction.** OpenAI stays the single provider. PydanticAI makes provider-swapping cheap later, but v2 does not build or test that surface.
- **Service-grade observability beyond traces.** Metrics dashboards, alerting, log aggregation pipelines, and cost budgets in LangSmith are left for later; v2 ships the traces themselves.

## Further Notes

- **Dependencies added:** `pydantic-ai-slim[openai]` (or `pydantic-ai`), plus the OpenTelemetry OTLP/HTTP exporter libraries used to ship spans to LangSmith. The OpenAI SDK and httpx remain. Removed code: `agent.py` (loop), `llm.py` (Protocol + adapter), `tools/registry.py`, and the TypedDict results in `tools/results.py` (superseded by Pydantic models).
- **Interview framing.** This work exists to demonstrate fluency and judgment with a PydanticAI + LangSmith stack. The defensible narrative is as much about what was *not* used (LangGraph) and *why* as about what was built.
- **Vocabulary.** `CONTEXT.md` is unchanged and still governs: Turn, Tool Round, Tool Call, History, Tool Error, Infrastructure Error. In v2, "Tool Round" maps onto one PydanticAI request iteration that issues Tool Calls, and the round cap maps onto the request usage limit.
