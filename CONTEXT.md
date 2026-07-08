# Weather Assistant

A command-line weather assistant: the user asks natural-language weather questions, an LLM decides when to call a weather tool, and the answer streams back token-by-token.

## Language

**Turn**:
One complete exchange: a user message through to the assistant's final streamed answer, including any tool rounds in between.
_Avoid_: request, exchange, interaction

**Tool Round**:
One LLM call within a turn whose output includes tool calls. A turn contains zero or more tool rounds, bounded by a fixed cap, followed by a final answer.
_Avoid_: iteration, cycle

**Tool Call**:
A single request from the LLM to execute one named tool with arguments. Multiple tool calls in one tool round execute concurrently.
_Avoid_: function call, tool invocation

**History**:
The client-owned record of a conversation: every user message, model output, and tool result so far, carried into each subsequent LLM call.
_Avoid_: context, state, memory

**Tool Error**:
A failure the LLM can act on (unknown city, weather service timeout), returned to it as a structured result rather than raised.
_Avoid_: exception, tool failure

**Infrastructure Error**:
A failure only the user can act on (LLM API down, bad credentials), surfaced as a terminal message; never shown to the LLM.
_Avoid_: system error, fatal error
