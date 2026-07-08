---
title: meteobot — CLI weather assistant
labels: [ready-for-agent]
---

# PRD: meteobot

Vocabulary used below is defined in `CONTEXT.md` (Turn, Tool Round, Tool Call, History, Tool Error, Infrastructure Error). Decision rationale lives in `design-decisions.md`; deferred work in `what-happens-next.md`.

## Problem Statement

Checking the weather for one or several places means leaving the terminal, opening a browser or app, and running one search per city. There is no way to ask a plain-English question like "what's the weather in London and Tokyo right now?" from the command line and get one live, readable answer. meteobot is also a small, runnable reference for structuring an LLM agent with real tool calls, async concurrency, and streaming.

## Solution

meteobot: an interactive terminal assistant. The user types a natural-language question; an LLM decides whether it needs live data and issues one Tool Call per city, which fetch current conditions from Open-Meteo concurrently; the LLM then streams a single friendly answer token-by-token. The conversation is multi-turn (follow-ups like "and Paris?" work), survives bad input and provider failures, and exits on `quit` or `exit`. Setup is clone, provide an OpenAI key, run one command.

## User Stories

1. As a terminal user, I want to ask weather questions in natural language, so that I don't have to learn a query syntax.
2. As a terminal user, I want the assistant to fetch live weather data when my question needs it, so that answers reflect current conditions rather than the model's training data.
3. As a terminal user, I want to mention several cities in one message and get one unified answer, so that I don't have to ask city by city.
4. As a terminal user, I want multi-city lookups to run concurrently, so that three cities don't take three times as long.
5. As a terminal user, I want the answer to appear token-by-token as it is generated, so that I see progress immediately instead of staring at a silent prompt.
6. As a terminal user, I want follow-up questions to use conversation context, so that "what about tomorrow?" or "and in Paris?" work naturally.
7. As a terminal user, I want a misspelled or unknown city explained conversationally, with likely alternatives when the name is ambiguous, so that I can correct myself and move on.
8. As a terminal user, I want one invalid city to leave the other cities' results intact, so that a single typo doesn't spoil the whole question.
9. As a terminal user, I want a weather-service timeout or failure for one city reported in plain language within the answer, so that I know what happened without seeing a stack trace.
10. As a terminal user, I want the session to survive an LLM provider outage or network failure with a one-line message, so that a transient failure doesn't kill my conversation.
11. As a terminal user, I want questions that don't need weather data answered directly without spurious lookups, so that the assistant feels sensible rather than mechanical.
12. As a terminal user, I want to exit by typing `quit` or `exit`, so that leaving the app is obvious.
13. As a terminal user, I want arrow-key line editing and in-session input history, so that correcting and repeating questions is painless.
14. As a terminal user, I want answers in plain text suited to a terminal, so that markdown symbols don't litter the output.
15. As a terminal user, I want a clear startup message when my API key is missing, so that I can fix my environment before anything else happens.
16. As a terminal user, I want a runaway sequence of Tool Rounds cut off with an honest message, so that a confused model can't hang my session or burn my credits indefinitely.
17. As a new user, I want to clone the repo, provide a key via `.env` or an environment variable, and start the app with a single documented command, so that first run succeeds without troubleshooting.
18. As a reader of the codebase, I want a README explaining the design decisions and the named scope cuts with reasoning, so that I can understand the judgment calls, not just the code.
19. As a contributor, I want to run the test suite offline with no API key, so that I can verify behavior without spending money.
20. As a contributor, I want an opt-in live end-to-end test against the real APIs, so that I can confirm the integration genuinely works.
21. As a contributor, I want type hints and clear function signatures throughout, so that the code reads without archaeology.
22. As a developer extending meteobot, I want to add a new tool by registering it in one place, so that the agent loop and CLI never change when capabilities grow.
23. As a developer, I want all tunables (model, timeouts, round cap) in one injected configuration object with environment overrides, so that operational knobs are discoverable and changeable without code edits.
24. As a developer, I want Tool Errors returned as structured data rather than raised across the tool boundary, so that failure handling is a contract I can test.
25. As a developer, I want mechanical retries confined to the transport and SDK layers, so that transient network blips never become LLM re-planning or user-visible noise.
26. As a developer, I want the History inspectable client-side, so that "what did the model actually see?" is answerable during debugging.
27. As a developer, I want mocked tests built on fixtures captured from the real APIs, so that my fakes can't drift from the wire formats they imitate.
28. As a developer, I want the agent's steps — Turn start, each Tool Round's chosen Tool Calls, each Tool Call's name, arguments, latency, and error status, and cap hits — logged to stderr at a configurable level, so that I can debug the probabilistic agent loop without a tracing platform and without the logs polluting the streamed answer on stdout.

## Implementation Decisions

- **Name and packaging**: the project is named `meteobot` — package name, console script, and README title. Distributed as a uv-managed Python package with a committed lockfile and a console-script entry point; the install path is clone → provide key → `uv run meteobot`.
- **Module decomposition**: four concerns, separately housed — CLI (input loop and rendering), agent (LLM orchestration), tools (registry plus the weather tool), and config. No hexagonal layering; it would be over-engineering at this scale.
- **Agent loop**: one streaming code path with a bounded loop. Every LLM call streams; text deltas go to the renderer as they arrive, Tool Calls accumulate. If a Turn's response contains Tool Calls, they execute concurrently, results append to the History, and the loop calls again — capped at ~5 Tool Rounds per Turn, with the cap surfaced honestly to the user if hit. No special-casing of tool-only turns, chained rounds, or direct answers.
- **LLM interface**: OpenAI Responses API via the official SDK, model `gpt-4.1-mini` by default — chosen for dependable parallel function calling and no reasoning preamble (first token streams immediately). Model is configuration, env-overridable, never a hardcoded constant.
- **Conversation state**: a client-owned History list passed in full on every call; no server-side state (`previous_response_id` rejected: opaque, vendor-locked, no cost saving). Multi-turn support falls out of this for free.
- **Tool shape**: a single composed `get_weather(city)` tool — geocoding then current-conditions fetch happen inside the tool, which returns one structured result. Multiple cities in one message become multiple parallel Tool Calls. Tools sit at the level of user intent, not API plumbing; the model is never given coordinates to hallucinate.
- **Tool registry**: the extensibility seam. Each tool is a declaration (name, description, parameter schema, async handler); the registry is a plain value assembled explicitly at the composition root and injected into the agent, which consumes only the registry's schema list and dispatch function and never knows which tools exist. No import-time self-registration: adding a tool means adding one entry at assembly. This is also where future middleware (per-tool timeouts, retries, tracing, MCP-backed handlers) would attach.
- **HTTP layer**: httpx with one shared async client created at the composition root and injected into tools — connection pooling across the concurrent fan-out, timeout and connect-retry policy configured once. The SDK retries its own calls; the LLM is allowed to re-plan after a Tool Error but is never the retry mechanism.
- **Input loop**: the blocking terminal read is pushed onto a worker thread and awaited, keeping the event loop free of blocking calls; readline gives line editing and in-session history for free.
- **Rendering**: plain flushed writes per text delta behind a tiny renderer function; the agent loop is presentation-blind. Decided by prototype: a transient status line during Tool Rounds costs a spinner thread, cursor-control escape codes, and real teardown for negligible value. The system prompt asks the model for plain text.
- **Error routing**: two tiers. Tool Errors (unknown city, weather-service timeout or failure) are caught inside the tool and returned as structured results the model explains conversationally; per-call capture around concurrent Tool Calls preserves sibling results. Infrastructure Errors (LLM API failure, bad credentials) are caught at the CLI loop: one friendly line, the Turn is abandoned, the REPL survives.
- **Configuration**: a frozen settings object read once at startup and injected — API key (fail-fast with a clear message if absent, passed explicitly to the SDK client), model, HTTP timeout, round cap, and log level. `.env` loaded via python-dotenv without overriding exported variables; README documents both the `.env` path and an inline-env fallback command. Invalid tunables (non-numeric or out-of-range timeout/cap) fail fast with the offending variable named, not a raw parse traceback.
- **Debug logging**: stdlib `logging` to stderr, level-gated via `METEOBOT_LOG_LEVEL` (default `WARNING`, quiet in a normal session; `INFO` surfaces the per-step trace). One line at each existing seam — Turn start, the Tool Calls a Tool Round chose, and each Tool Call's name/args/latency/error — all at `INFO`. A cap hit is the exception: it logs at `WARNING`, since a runaway sequence tripping the cap is an anomaly worth seeing even in a quiet default session. So the default level shows only cap trips; `INFO` adds the full per-step trace. stderr, never stdout, so logs never interleave with the streamed answer or break a `> answer.txt` redirect; the app writes no file, since a file is a redirect (`2> run.log`) the operator owns, not app machinery. This is the minimal in-scope slice for debugging the agent loop; the service-grade version (structured JSON keyed by conversation/turn, OTel tracing spans at the dispatch seam, token/cost and latency metrics, log aggregation) stays deferred in `what-happens-next.md`.

## Testing Decisions

- **What makes a good test here**: assert on external behavior at a seam — the structured result a tool returns, the sequence of calls a fake LLM sees, the History shape after a Turn — never on internals like private helpers or call counts of implementation details. Fakes stand in for vendors; the code under test is always real.
- **Methodology**: TDD (red-green-refactor) for the deterministic layers, driving implementation of the tool contract and agent loop mechanics.
- **Two seams, both pre-existing in the design**:
  - *HTTP transport seam*: the shared injected client accepts a mock transport, so weather-tool tests exercise real tool code against canned HTTP responses — happy path, unknown city → structured Tool Error, timeout → structured Tool Error.
  - *LLM client seam*: the agent accepts a fake client exposing the same streaming surface as the real SDK, replaying scripted event sequences — verifying concurrent Tool Call dispatch, History growth, round-cap trip, and direct-answer Turns.
- **Realistic fixtures**: mock responses and streamed event sequences are captured from the real Open-Meteo and Responses APIs, not invented, so fixtures can't drift from the wire format.
- **Live e2e test**: one opt-in test against both real APIs (multi-city question → streamed answer), marked so it runs only on explicit request with a key present and is excluded from the default suite.
- **Deliberately untested**: CLI input/rendering (fake-TTY fiddliness, near-zero signal) and the LLM's judgment beyond the e2e smoke check (evals territory, named as such in the README).
- **Prior art**: none — greenfield repo; these tests establish the project's patterns.

