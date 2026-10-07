# forge-ops

A multi-agent system that turns a rough project idea into working code. Three
LangGraph agents run as a pipeline and call Claude through the `anthropic` SDK,
behind a live Rich terminal dashboard that shows what every agent is doing, plus
token usage, estimated cost and status.

```
idea → ArchitectAgent → architecture spec → TeamLeadAgent → WorkerAgent(s) → code in projects/<id>/
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
```

Put your key in a `.env` file in the repo root (or export it):

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then run:

```bash
python forge.py                       # opens an interactive forge> prompt
python forge.py "a todo CLI in Python"   # one-shot: run a single project and exit
python forge.py "idea" --project-id my-project
```

### The `forge>` prompt

Type an idea to start a project, or use a command:

| Command | What it does |
|---|---|
| `/help` | show the command list |
| `/projects` | list projects built so far, with their final status |
| `/usage` | token and estimated cost totals for this session |
| `/exit` | leave forge (also `/quit` or Ctrl+D) |

Ctrl+C during a run cancels that run and keeps the session. Workers stop at their
next step; calls already in flight to the API finish first.

The prompt builds **new projects** only. Asking the agents to change an existing
project is not supported yet.

## What you will see

1. **Architect phase (interactive).** The Architect asks up to a few clarifying
   questions, then shows its understanding of your requirements. Answer the
   questions in the terminal, then `accept`, `change` (with notes) or `reject`.
2. **Execution phase (live dashboard).** The TeamLead splits the spec into tasks
   and runs them in parallel workers. The dashboard shows:
   - a header with project, phase, elapsed time, tokens and estimated cost
   - a task table (status, dependencies, retries, review score)
   - one block per active worker (current step, current tool, rework count, tokens)
   - token usage per agent, and a rolling event log
3. **Summary.** Final status, tasks completed, average review score, tokens, cost
   and the files produced.

Output goes to `projects/<id>/`. Agent logging goes to `projects/<id>/forge.log`
so it never corrupts the live display. When stdout is not a terminal (CI, pipes),
the UI falls back to plain one-line-per-event output.

## How it works

| Piece | File | Role |
|---|---|---|
| ArchitectAgent | `ArchitectAgent.py` | Turns the idea into a structured spec: analyse, ask questions, check completeness, human review, write the spec. Pauses with LangGraph `interrupt()` for human input. |
| TeamLeadAgent | `TeamLeadAgent.py` | Decomposes the spec into tasks with dependencies, schedules ready tasks, dispatches workers in parallel (default cap 5), reviews each batch, retries failed tasks (default 2 retries). |
| WorkerAgent | `WorkerAgent.py` | Implements one task: plan, execute with tools, review, rework (default max 2 cycles), generate tests, report. |
| Worker tools | `WorkerTools.py`, `utils.py` | `create_file`, `read_file`, `write_file`, project schema, and file locks so parallel workers do not collide. |
| Memory and telemetry | `forge_memory.py` | JSONL telemetry under `projects/_telemetry/` plus an optional mem0 memory layer that agents consult before planning. |
| Event layer | `forge_events.py` | Thread-safe event bus, a proxy around the Anthropic client that records tokens, latency and cost on every call, and a decorator that traces every graph node. |
| UI | `forge_ui.py` | Folds events into state and renders the dashboard and prompts with [Rich](https://github.com/Textualize/rich). |
| Session | `forge_session.py` | The `forge>` prompt and its commands. |
| Entry point | `forge.py` | Wires the bus, UI and pipeline together. |

### Observability

Every Claude call goes through a tracking proxy, so token counts come from the
API's own `usage` field. Each event (node start/end, LLM call, tool call, task
status) is also appended to `projects/_telemetry/agent_events.jsonl`. Cost is an
estimate from a small price table in `forge_events.py` (`PRICES_USD_PER_MTOK`);
models not in the table are counted as zero cost, so check the table before
relying on the numbers.

## Configuration

Defaults live next to the code that uses them:

| Setting | File | Default |
|---|---|---|
| Model | `ArchitectAgent.py`, `WorkerAgent.py`, `TeamLeadAgent.py` (`MODEL`) | `claude-haiku-4-5` |
| Parallel workers | `TeamLeadAgent.py` (`MAX_WORKER_PARALLELISM`) | 5 |
| Task retry cycles | `TeamLeadAgent.py` (`MAX_RETRY_CYCLES`) | 2 |
| Worker rework cycles | `WorkerAgent.py` (`MAX_REWORK_CYCLES`) | 2 |
| Clarification rounds | `ArchitectAgent.py` (`MAX_CLARIFICATION_ROUNDS`) | 4 |

## Tests

```bash
python -m pytest tests -q
```

Run it as `python -m pytest tests`, not bare `pytest`: the repo also contains
generated projects under `projects/` that are not part of the suite.

The tests mock the Anthropic client, so they need no API key and cost nothing.

## Design docs

- Spec: `docs/superpowers/specs/2026-10-06-forge-terminal-ui-design.md`
- Implementation plan: `docs/superpowers/plans/2026-10-06-forge-terminal-ui.md`

## Known limitations

- Only new projects can be started; there is no "modify an existing project" flow.
- Only `messages.create` is tracked for token usage (no agent uses streaming).
- The dashboard has been exercised with a simulated run and the test suite; it has
  not been verified against a long real run on every terminal.
