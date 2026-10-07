# forge-ops Terminal UI and Event Layer — Design

Date: 2026-10-06
Status: Draft, awaiting user review

## Goal

Give a human a single Rich-based terminal entry point (`forge`) that runs the full
pipeline (Architect → TeamLead → parallel Workers), lets them answer the
Architect's questions, and shows what every agent is doing, with token usage,
estimated cost and status, live.

## Success criteria

- `python forge.py "<idea>"` (or prompting for the idea) runs the whole pipeline.
- Architect questions and the accept/change/reject decision are answered inside
  the UI, including on Windows (no reliance on `select` on stdin).
- During execution the UI shows, live: task table, one block per parallel worker
  (current node, current tool, rework count, review score, tokens), per-agent
  token totals, estimated cost, and a rolling event log.
- Token usage is real, taken from `response.usage` of every Claude call.
- Every event is persisted to the existing `TelemetryStore`.
- The existing test suite (`tests/`) keeps passing unchanged.

## Non-goals

- No web UI. Rich only.
- No remote/standalone monitor process (a `forge monitor` tailing JSONL is a
  possible later addition, not part of this work).
- No change to agent node logic, prompts, or graph topology.
- No kill/pause control of running workers.

## Constraints found in the code

- Every module creates its own `Anthropic()` client; tests patch
  `ArchitectAgent.Anthropic`, `WorkerAgent.Anthropic`, `TeamLeadAgent.Anthropic`.
  Instrumentation must wrap whatever that returns, not replace it.
- Test mocks return responses without a `.usage` attribute; instrumentation must
  tolerate that.
- Workers run in a `ThreadPoolExecutor` (`TeamLeadAgent.dispatch_workers`).
  `contextvars` do not propagate into pool threads, so worker identity must be set
  explicitly at the top of each worker thread.
- The Architect graph pauses via LangGraph `interrupt()`; resume is
  `Command(resume=...)`.

## Architecture

Three units with narrow interfaces:

1. `forge_events.py`: event bus, client tracking, node tracing, pricing. No Rich.
2. `forge_ui.py`: Rich rendering, state reducer, prompts. Depends only on the bus.
3. `forge.py`: entry point; wires bus, UI, telemetry subscriber and the pipeline.

### 1. forge_events.py

- `EventBus`: thread-safe `publish(event)` / `subscribe(callback)`; bounded history
  (default 1000) replayed to late subscribers.
- Event: extends the existing `AgentEvent` fields with `kind`, one of
  `node_start`, `node_end`, `llm_call`, `tool_call`, `task_status`,
  `prompt_user`, `run_end`. Token fields (`input_tokens`, `output_tokens`,
  `cost_usd`, `model`) live in `metadata`.
- `track(client, agent)`: returns a proxy around an `Anthropic` client. It forwards
  `messages.create` unchanged, then publishes an `llm_call` event with model,
  latency, tokens and cost. Usage is read with `getattr`; missing usage records
  zeros. Call sites change from `Anthropic()` to `track(Anthropic(), "<agent>")`.
- `@traced_node(agent)`: decorator for LangGraph node functions. Publishes
  `node_start`/`node_end` (duration, success, error) and sets a context variable
  with `agent`, `node`, `worker_id`, `task_id` so `llm_call` events inherit the
  attribution.
- Worker context: `run_worker` and `_run_one` set the context variable at thread
  start.
- `PRICES`: per-model USD per million input/output tokens, configurable; unknown
  models cost 0 and are flagged in the event.
- Telemetry subscriber: writes every event through `TelemetryStore.log_event`.
- Housekeeping: remove duplicate `Question`, `UserAnswer`, `RequirementsReview`,
  `ArchitectureSpec` definitions in `models.py`.

Edits to existing code: about 10 `Anthropic()` call sites, 15–20 node decorators,
2 worker-context sites. No node logic changes.

### 2. forge_ui.py

Threading: the pipeline runs in a background thread; `Live` runs on the main
thread. The UI subscribes to the bus and folds events into a state object under a
lock; `Live` refreshes about 4 times per second from that state. Worker threads
never write to the terminal.

Phase 1 — Architect (interactive). `interrupt()` payloads reach the UI as
`prompt_user` events. `Live` is stopped; the UI renders the question panel and asks
via Rich (`Prompt.ask(choices=...)` for multiple choice, `Confirm.ask` for yes/no,
`Prompt.ask` for text). The understanding summary renders as Markdown; the decision
(accept / change with notes / reject) is also asked through Rich. The answer is
returned through `Command(resume=...)` and `Live` restarts. The existing stdin-based
`run_architect_cli` stays for tests and is not used by the UI path.

Phase 2 — execution dashboard, one `Layout`:

- Header: project, phase, elapsed, tokens in/out, estimated cost.
- Tasks table: id, status (color-coded), dependencies, retry count
  (from `task_status` events).
- Workers panel: one block per active worker with current node, current tool,
  rework count, last review score, per-worker tokens.
- Token panel: totals per agent and model, running cost.
- Event log: rolling tail of the last ~8 events.

End state: summary panel with the `ProjectReport`, per-batch review scores, total
tokens and cost, touched files and the telemetry location.

### 3. forge.py

Creates the bus, UI and telemetry subscriber; runs Architect then TeamLead in the
background thread; passes the Architect's `ArchitectureSpec.spec_text` to
`run_teamlead`; prints the summary. `forge.py` also exposes the idea as a CLI
argument.

## Error handling

- Pipeline exception: shown in a red panel, terminal restored, exception re-raised
  after cleanup.
- Ctrl+C: stop `Live`, exit cleanly; in-flight LLM calls finish in their threads
  (threads cannot be killed safely).
- Not a TTY (CI, pipes): fall back to plain one-line-per-event output, no `Live`.
- Missing `usage` on a response: record zeros, never raise.
- Subscriber failure: a failing subscriber is logged and skipped; it must not break
  an agent call.

## Testing

- Unit: `EventBus` (ordering, thread safety, replay), `track()` (usage present,
  usage absent, passthrough of return value and exceptions), cost math.
- UI state reducer: feed event sequences, assert state (no terminal).
- Render smoke test with `Console(record=True)`.
- Regression: the full existing suite must pass unchanged.

## Open items

- Exact per-model prices for the price table are filled in at implementation time
  from Anthropic's published pricing.
- Any Rich dependency (`rich`) is added to `requirements.txt` at implementation time.
