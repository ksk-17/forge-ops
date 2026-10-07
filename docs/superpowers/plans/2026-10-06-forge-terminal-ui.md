# forge Terminal UI and Event Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One `python forge.py "<idea>"` command that runs Architect → TeamLead → parallel Workers behind a live Rich dashboard (tasks, per-worker progress, real token usage and cost) with the Architect's questions answered inside the UI.

**Architecture:** A thread-safe `EventBus` (`forge_events.py`) receives events from a proxy around each `Anthropic` client (tokens, latency, cost), from a decorator on every LangGraph node, and from task-status/tool-call hooks. A Rich `ForgeUI` (`forge_ui.py`) folds events into state and renders a dashboard on the main thread while the pipeline runs in a background thread; human prompts cross threads through a request queue. `forge.py` wires everything together.

**Tech Stack:** Python, Rich, LangGraph, anthropic SDK, pytest, portalocker.

**Spec:** `docs/superpowers/specs/2026-10-06-forge-terminal-ui-design.md`

## Global Constraints

- Existing tests patch `ArchitectAgent.Anthropic`, `WorkerAgent.Anthropic`, `TeamLeadAgent.Anthropic`; instrumentation must wrap whatever those return. Do not replace the `Anthropic()` call sites with a shared function.
- Test mocks return responses with no `.usage` (or a `MagicMock` one); token reading must never raise.
- `contextvars` do not cross into `ThreadPoolExecutor` threads; worker identity is derived from node `state` (`worker_id`, `task.task_id`) inside the traced-node wrapper, not set at thread start. (Deviation from the spec's "set at thread start": same effect, more robust across LangGraph's own executor.)
- No change to node logic, prompts, or graph topology.
- `forge_events.py` must not import Rich. `forge_ui.py` must import only the bus/event helpers, never agents.
- Prices (USD per million tokens, input/output): `claude-haiku-4-5` = 1.00 / 5.00. Verify against Anthropic's published pricing when implementing Task 1.
- Run tests with `python -m pytest tests -q` (not bare `pytest`, which collects `projects/` demo output and errors).
- **Git hygiene:** the working tree has the user's uncommitted edits to `ArchitectAgent.py`, `TeamLeadAgent.py`, `WorkerAgent.py`, `models.py`, `requirements.txt` and an untracked `forge_memory.py`. Before Task 3, ask the user to commit or stash those, otherwise `git add` on those files sweeps their work into this plan's commits. Commit only files named in each task.

## Review Focus

1. Blank free-text answer in the UI must not shift later answers (`ArchitectAgent.ask_user` drops blank lines) → UI sends a placeholder (Task 8).
2. LLM response without `usage`, or with a `MagicMock` usage → zero tokens, no exception (Task 2).
3. A subscriber that raises must not break an agent's LLM call or other subscribers (Task 1).
4. Question/error text containing Rich markup such as `[/oops]` or `[red` must render literally, not crash or restyle (Task 7).
5. Pipeline exception or Ctrl+C must restore the terminal and surface the error; non-TTY runs must fall back to plain lines (Task 8).

---

### Task 0: Make the environment runnable on Windows

`utils.py` imports `fcntl` (Unix only) although `requirements.txt` already lists `portalocker` "for fcntl". On this Windows venv the existing tests cannot even be collected (`ModuleNotFoundError: fcntl`, and `anthropic` is not installed).

**Files:**
- Modify: `utils.py:3,29,46,55,74`
- Test: `tests/test_worker_tools.py` (existing)

**Interfaces:**
- Consumes: nothing.
- Produces: importable `utils`, `WorkerTools`, and agent modules on Windows; a known baseline test result.

- [ ] **Step 1: Install dependencies**

Run: `.venv\Scripts\python -m pip install -r requirements.txt rich`
Expected: installs `anthropic`, `portalocker`, `rich`, etc. without error.

- [ ] **Step 2: Confirm the failure**

Run: `.venv\Scripts\python -m pytest tests -q`
Expected: collection error `ModuleNotFoundError: No module named 'fcntl'`.

- [ ] **Step 3: Swap fcntl for portalocker in utils.py**

Replace `import fcntl` with `import portalocker`, then replace the four calls:

```python
        fcntl.flock(f, fcntl.LOCK_EX)   ->   portalocker.lock(f, portalocker.LOCK_EX)
            fcntl.flock(f, fcntl.LOCK_UN)   ->   portalocker.unlock(f)
```

(Both `acquire_lock` and `remove_lock` have one lock and one unlock each.)

- [ ] **Step 4: Record the baseline**

Run: `.venv\Scripts\python -m pytest tests -q`
Expected: collection succeeds. Write down the pass/fail counts; every later "run the full suite" step must match or improve on this baseline. If tests fail for reasons unrelated to this plan, list them to the user and do not fix them here.

- [ ] **Step 5: Commit**

```bash
git add utils.py
git commit -m "fix: use portalocker instead of fcntl so locking works on Windows"
```

---

### Task 1: Event bus, context, pricing, telemetry subscriber

**Files:**
- Create: `forge_events.py`
- Create: `tests/test_forge_events.py`
- Modify: `tests/conftest.py` (add `fresh_bus` fixture)

**Interfaces:**
- Consumes: `forge_memory.TelemetryStore.log_event(event)`.
- Produces (used by every later task):
  - `EventBus(history_size=1000)` with `.run_id: str`, `.project_id: str`, `.subscribe(callback, replay=False)`, `.publish(event)`, `.history() -> list`.
  - `get_bus() -> EventBus`, `set_bus(bus)`.
  - `emit(kind, *, agent=None, node=None, task_id=None, worker_id=None, duration_seconds=0.0, success=True, error=None, metadata=None) -> dict` (fills `event_id`, `run_id`, `project_id`, `timestamp`, and agent/node/task_id/worker_id from the current context when not given).
  - `set_context(**fields) -> Token`, `reset_context(token)`, `current_context() -> dict`.
  - `estimate_cost(model, input_tokens, output_tokens) -> tuple[float, bool]`.
  - `telemetry_subscriber(event)`.
  - Event kinds: `node_start`, `node_end`, `llm_call`, `tool_call`, `task_status`, `prompt_user`, `run_end`.

- [ ] **Step 1: Add the fixture to tests/conftest.py (append)**

```python
import pytest


@pytest.fixture
def fresh_bus():
    """Install an isolated EventBus as the global bus for one test."""
    from forge_events import EventBus, get_bus, set_bus

    previous = get_bus()
    bus = EventBus()
    set_bus(bus)
    yield bus
    set_bus(previous)
```

- [ ] **Step 2: Write the failing tests (`tests/test_forge_events.py`)**

```python
import json
import threading

import pytest

import forge_events as fe
from forge_events import EventBus, emit, estimate_cost, reset_context, set_context


def test_publish_delivers_in_order(fresh_bus):
    seen = []
    fresh_bus.subscribe(seen.append)
    for i in range(3):
        emit("node_start", node=f"n{i}")
    assert [e["node"] for e in seen] == ["n0", "n1", "n2"]


def test_history_is_bounded():
    bus = EventBus(history_size=3)
    for i in range(5):
        bus.publish({"i": i})
    assert [e["i"] for e in bus.history()] == [2, 3, 4]


def test_replay_gives_late_subscriber_past_events(fresh_bus):
    emit("node_start", node="a")
    seen = []
    fresh_bus.subscribe(seen.append, replay=True)
    assert [e["node"] for e in seen] == ["a"]


def test_failing_subscriber_does_not_break_publish_or_others(fresh_bus):
    good = []

    def bad(_event):
        raise RuntimeError("boom")

    fresh_bus.subscribe(bad)
    fresh_bus.subscribe(good.append)
    emit("node_start", node="a")  # must not raise
    assert len(good) == 1


def test_publish_is_thread_safe(fresh_bus):
    seen = []
    fresh_bus.subscribe(seen.append)

    def work():
        for _ in range(100):
            emit("llm_call")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 800
    assert len({e["event_id"] for e in seen}) == 800


def test_emit_inherits_context_and_context_resets(fresh_bus):
    token = set_context(agent="worker", node="plan_task", worker_id="worker-a", task_id="a")
    try:
        e = emit("llm_call")
    finally:
        reset_context(token)
    assert (e["agent"], e["node"], e["worker_id"], e["task_id"]) == (
        "worker", "plan_task", "worker-a", "a",
    )
    assert emit("llm_call")["worker_id"] is None


def test_emit_stamps_run_and_project(fresh_bus):
    fresh_bus.project_id = "proj-1"
    e = emit("run_end")
    assert e["run_id"] == fresh_bus.run_id
    assert e["project_id"] == "proj-1"
    assert e["timestamp"].endswith("+00:00") or e["timestamp"].endswith("Z")


def test_estimate_cost_matches_model_prefix():
    assert estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == (6.0, True)


def test_estimate_cost_unknown_model_is_zero_and_flagged():
    assert estimate_cost("some-future-model", 1000, 1000) == (0.0, False)


def test_telemetry_subscriber_writes_agent_event_shape(tmp_path, monkeypatch, fresh_bus):
    import forge_memory

    monkeypatch.setattr(forge_memory, "_TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(forge_memory, "_EVENTS_FILE", tmp_path / "agent_events.jsonl")
    fresh_bus.subscribe(fe.telemetry_subscriber)

    emit("llm_call", agent="worker", node="plan_task", metadata={"input_tokens": 5})

    rec = json.loads((tmp_path / "agent_events.jsonl").read_text().splitlines()[0])
    assert rec["agent"] == "worker"
    assert rec["metadata"]["kind"] == "llm_call"
    assert rec["metadata"]["input_tokens"] == 5
    assert {
        "event_id", "run_id", "project_id", "agent", "node", "task_id",
        "duration_seconds", "success", "error", "metadata", "timestamp",
    } <= set(rec)
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_forge_events.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'forge_events'`.

- [ ] **Step 4: Implement `forge_events.py`**

```python
"""
forge_events.py — event bus, LLM-call tracking and node tracing for forge-ops.

No Rich dependency. Agents publish events; the UI and the telemetry store
subscribe. Public API: EventBus, get_bus, set_bus, emit, set_context,
reset_context, current_context, estimate_cost, telemetry_subscriber,
track, traced_node.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple

from typing_extensions import TypedDict

from forge_memory import TelemetryStore

logger = logging.getLogger(__name__)

EVENT_KINDS = (
    "node_start", "node_end", "llm_call", "tool_call",
    "task_status", "prompt_user", "run_end",
)

# USD per million tokens: (input, output). Matched by model-name prefix.
PRICES_USD_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
}


class ForgeEvent(TypedDict):
    event_id: str
    run_id: str
    kind: str
    project_id: str
    agent: str
    node: str
    task_id: Optional[str]
    worker_id: Optional[str]
    duration_seconds: float
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]
    timestamp: str


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Tuple[float, bool]:
    """Return (cost_usd, price_known)."""
    for prefix, (in_price, out_price) in PRICES_USD_PER_MTOK.items():
        if (model or "").startswith(prefix):
            return (input_tokens * in_price + output_tokens * out_price) / 1_000_000, True
    return 0.0, False


# ── context ────────────────────────────────────────────────────────────────

_ctx: contextvars.ContextVar = contextvars.ContextVar("forge_event_ctx", default=None)


def set_context(**fields: Any) -> contextvars.Token:
    current = _ctx.get() or {}
    return _ctx.set({**current, **fields})


def reset_context(token: contextvars.Token) -> None:
    _ctx.reset(token)


def current_context() -> Dict[str, Any]:
    return dict(_ctx.get() or {})


# ── bus ────────────────────────────────────────────────────────────────────

class EventBus:
    def __init__(self, history_size: int = 1000) -> None:
        self.run_id: str = uuid.uuid4().hex[:12]
        self.project_id: str = ""
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)

    def subscribe(self, callback: Callable[[Dict[str, Any]], None], replay: bool = False) -> None:
        with self._lock:
            past = list(self._history) if replay else []
            self._subscribers.append(callback)
        for event in past:
            self._deliver(callback, event)

    def publish(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            self._deliver(callback, event)

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    @staticmethod
    def _deliver(callback: Callable[[Dict[str, Any]], None], event: Dict[str, Any]) -> None:
        try:
            callback(event)
        except Exception:  # a broken subscriber must never break an agent
            logger.exception("forge_events: subscriber failed")


_bus = EventBus()


def get_bus() -> EventBus:
    return _bus


def set_bus(bus: EventBus) -> None:
    global _bus
    _bus = bus


def emit(
    kind: str,
    *,
    agent: Optional[str] = None,
    node: Optional[str] = None,
    task_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    duration_seconds: float = 0.0,
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ForgeEvent:
    bus = get_bus()
    ctx = current_context()
    event: ForgeEvent = {
        "event_id": uuid.uuid4().hex,
        "run_id": bus.run_id,
        "kind": kind,
        "project_id": bus.project_id,
        "agent": agent or ctx.get("agent", ""),
        "node": node or ctx.get("node", ""),
        "task_id": task_id if task_id is not None else ctx.get("task_id"),
        "worker_id": worker_id if worker_id is not None else ctx.get("worker_id"),
        "duration_seconds": duration_seconds,
        "success": success,
        "error": error,
        "metadata": dict(metadata or {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    bus.publish(event)
    return event


# ── telemetry persistence ──────────────────────────────────────────────────

_telemetry_lock = threading.Lock()


def telemetry_subscriber(event: Dict[str, Any]) -> None:
    """Persist every event as an AgentEvent record (kind/worker_id go in metadata)."""
    record = {
        "event_id": event["event_id"],
        "run_id": event["run_id"],
        "project_id": event["project_id"],
        "agent": event["agent"],
        "node": event["node"],
        "task_id": event["task_id"],
        "duration_seconds": event["duration_seconds"],
        "success": event["success"],
        "error": event["error"],
        "metadata": {
            **event["metadata"],
            "kind": event["kind"],
            "worker_id": event["worker_id"],
        },
        "timestamp": event["timestamp"],
    }
    with _telemetry_lock:  # TelemetryStore appends are not thread-locked
        TelemetryStore.log_event(record)
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_forge_events.py -q`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add forge_events.py tests/test_forge_events.py tests/conftest.py
git commit -m "feat: add event bus, context, pricing and telemetry subscriber"
```

---

### Task 2: `track()` client proxy and `@traced_node`

**Files:**
- Modify: `forge_events.py` (append)
- Modify: `tests/test_forge_events.py` (append)

**Interfaces:**
- Consumes: `emit`, `set_context`, `reset_context`, `estimate_cost` from Task 1.
- Produces:
  - `track(client, agent: str)` → proxy with `.messages.create(**kwargs)` that forwards unchanged and emits an `llm_call` event (`metadata`: `model`, `input_tokens`, `output_tokens`, `cost_usd`, `price_known`); other attributes forward to the wrapped client; wrapping an already-tracked client returns it unchanged.
  - `traced_node(agent: str)` → decorator for LangGraph node functions: emits `node_start`/`node_end`, sets context (`agent`, `node`, and `worker_id`/`task_id` read from the state mapping when present), treats `GraphInterrupt` as a pause (`success=True`, `metadata["paused"]=True`), re-raises everything.

- [ ] **Step 1: Write the failing tests (append to `tests/test_forge_events.py`)**

```python
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from langgraph.errors import GraphInterrupt

from forge_events import track, traced_node


def _resp(i=10, o=5):
    return SimpleNamespace(
        content=[SimpleNamespace(text="hi")],
        usage=SimpleNamespace(input_tokens=i, output_tokens=o),
    )


def test_track_forwards_call_and_returns_response(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp()
    result = track(client, "worker").messages.create(
        model="claude-haiku-4-5", max_tokens=5, messages=[]
    )
    assert result is client.messages.create.return_value
    client.messages.create.assert_called_once_with(
        model="claude-haiku-4-5", max_tokens=5, messages=[]
    )


def test_track_emits_tokens_and_cost(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp(10, 5)
    track(client, "worker").messages.create(model="claude-haiku-4-5")
    e = fresh_bus.history()[-1]
    assert e["kind"] == "llm_call" and e["agent"] == "worker"
    md = e["metadata"]
    assert (md["input_tokens"], md["output_tokens"]) == (10, 5)
    assert md["cost_usd"] == pytest.approx((10 * 1.0 + 5 * 5.0) / 1_000_000)
    assert md["price_known"] is True and e["duration_seconds"] >= 0


@pytest.mark.parametrize("response", [
    SimpleNamespace(content=[]),                      # no usage attribute
    SimpleNamespace(content=[], usage=None),          # usage is None
    MagicMock(),                                      # usage is a MagicMock
])
def test_track_tolerates_missing_or_mock_usage(fresh_bus, response):
    client = MagicMock()
    client.messages.create.return_value = response
    out = track(client, "a").messages.create(model="claude-haiku-4-5")
    assert out is response
    md = fresh_bus.history()[-1]["metadata"]
    assert (md["input_tokens"], md["output_tokens"], md["cost_usd"]) == (0, 0, 0.0)


def test_track_records_failure_and_reraises(fresh_bus):
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("api down")
    with pytest.raises(RuntimeError):
        track(client, "a").messages.create(model="claude-haiku-4-5")
    e = fresh_bus.history()[-1]
    assert e["kind"] == "llm_call" and e["success"] is False and "api down" in e["error"]


def test_track_proxies_other_attributes_and_does_not_double_wrap(fresh_bus):
    client = MagicMock()
    tracked = track(client, "a")
    assert tracked.models is client.models
    assert track(tracked, "a") is tracked


def test_traced_node_emits_start_end_and_returns_value(fresh_bus):
    @traced_node("worker")
    def plan_task(state):
        return {"plan": "x"}

    state = {"worker_id": "worker-a", "task": SimpleNamespace(task_id="a")}
    assert plan_task(state) == {"plan": "x"}
    rows = [(e["kind"], e["node"], e["worker_id"], e["task_id"]) for e in fresh_bus.history()]
    assert rows == [
        ("node_start", "plan_task", "worker-a", "a"),
        ("node_end", "plan_task", "worker-a", "a"),
    ]
    assert fresh_bus.history()[-1]["success"] is True


def test_traced_node_attributes_llm_calls_to_node_and_worker(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp()

    @traced_node("worker")
    def execute_task(state):
        track(client, "worker").messages.create(model="claude-haiku-4-5")
        return {}

    execute_task({"worker_id": "worker-a", "task": SimpleNamespace(task_id="a")})
    call = next(e for e in fresh_bus.history() if e["kind"] == "llm_call")
    assert (call["node"], call["worker_id"], call["task_id"]) == ("execute_task", "worker-a", "a")


def test_traced_node_records_failure_and_reraises(fresh_bus):
    @traced_node("teamlead")
    def boom(state):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom({})
    end = fresh_bus.history()[-1]
    assert end["kind"] == "node_end" and end["success"] is False and "nope" in end["error"]


def test_traced_node_treats_graph_interrupt_as_pause(fresh_bus):
    @traced_node("architect")
    def ask_user(state):
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        ask_user({})
    end = fresh_bus.history()[-1]
    assert end["success"] is True and end["metadata"]["paused"] is True


def test_traced_node_preserves_signature_and_resets_context(fresh_bus):
    @traced_node("worker")
    def plan_task(state):
        return {}

    assert plan_task.__name__ == "plan_task"
    assert list(inspect.signature(plan_task).parameters) == ["state"]
    plan_task({"worker_id": "worker-a"})
    assert emit("llm_call")["worker_id"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_forge_events.py -q`
Expected: FAIL — `ImportError: cannot import name 'track'`.

- [ ] **Step 3: Implement (append to `forge_events.py`)**

```python
# ── LLM-call tracking ──────────────────────────────────────────────────────

def _as_int(value: Any) -> int:
    """Token counts must be real ints; anything else (None, MagicMock) is 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class _TrackedMessages:
    def __init__(self, messages: Any, agent: str) -> None:
        self._messages = messages
        self._agent = agent

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "")
        started = time.monotonic()
        try:
            response = self._messages.create(**kwargs)
        except Exception as exc:
            emit(
                "llm_call", agent=self._agent,
                duration_seconds=time.monotonic() - started,
                success=False, error=repr(exc),
                metadata={"model": model, "input_tokens": 0, "output_tokens": 0,
                          "cost_usd": 0.0, "price_known": False},
            )
            raise
        usage = getattr(response, "usage", None)
        in_tok = _as_int(getattr(usage, "input_tokens", None))
        out_tok = _as_int(getattr(usage, "output_tokens", None))
        cost, known = estimate_cost(model, in_tok, out_tok)
        emit(
            "llm_call", agent=self._agent,
            duration_seconds=time.monotonic() - started,
            metadata={"model": model, "input_tokens": in_tok, "output_tokens": out_tok,
                      "cost_usd": cost, "price_known": known},
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class _TrackedClient:
    def __init__(self, client: Any, agent: str) -> None:
        self._client = client
        self.messages = _TrackedMessages(client.messages, agent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def track(client: Any, agent: str) -> Any:
    """Wrap an Anthropic client so every messages.create emits an llm_call event."""
    if isinstance(client, _TrackedClient):
        return client
    return _TrackedClient(client, agent)


# ── node tracing ───────────────────────────────────────────────────────────

def traced_node(agent: str) -> Callable:
    """Decorator for LangGraph node functions (state in, update dict out)."""
    from langgraph.errors import GraphInterrupt

    def decorator(fn: Callable) -> Callable:
        node = fn.__name__

        @functools.wraps(fn)
        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            fields: Dict[str, Any] = {"agent": agent, "node": node}
            if isinstance(state, Mapping):
                if state.get("worker_id"):
                    fields["worker_id"] = state["worker_id"]
                task_id = getattr(state.get("task"), "task_id", None)
                if isinstance(task_id, str):
                    fields["task_id"] = task_id
            token = set_context(**fields)
            started = time.monotonic()
            emit("node_start")
            try:
                result = fn(state, *args, **kwargs)
            except BaseException as exc:
                paused = isinstance(exc, GraphInterrupt)
                emit(
                    "node_end",
                    duration_seconds=time.monotonic() - started,
                    success=paused,
                    error=None if paused else repr(exc),
                    metadata={"paused": paused},
                )
                raise
            else:
                emit("node_end", duration_seconds=time.monotonic() - started)
                return result
            finally:
                reset_context(token)

        return wrapper

    return decorator
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_forge_events.py -q`
Expected: all pass (about 23).

- [ ] **Step 5: Commit**

```bash
git add forge_events.py tests/test_forge_events.py
git commit -m "feat: add Anthropic client tracking and node tracing"
```

---

### Task 3: Instrument ArchitectAgent

**Files:**
- Modify: `ArchitectAgent.py` (import line ~16, `_llm` at ~40, 8 node defs)
- Create: `tests/test_instrumentation.py`

**Interfaces:**
- Consumes: `track`, `traced_node` (Task 2).
- Produces: Architect `llm_call` events with `agent="architect"` attributed to nodes; `node_start`/`node_end` for `analyse_input`, `generate_questions`, `ask_user`, `incorporate_answers`, `check_completeness`, `present_summary`, `handle_user_review`, `produce_architecture_spec`.

- [ ] **Step 1: Write the failing tests (`tests/test_instrumentation.py`)**

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ArchitectAgent as aa


def _resp(text="x", i=10, o=5):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=i, output_tokens=o),
    )


ARCHITECT_NODES = [
    "analyse_input", "generate_questions", "ask_user", "incorporate_answers",
    "check_completeness", "present_summary", "handle_user_review",
    "produce_architecture_spec",
]


def test_architect_llm_emits_llm_call_with_tokens(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp("hello", 7, 3)
    with patch("ArchitectAgent.Anthropic", return_value=client):
        assert aa._llm("sys", "user") == "hello"
    call = next(e for e in fresh_bus.history() if e["kind"] == "llm_call")
    assert call["agent"] == "architect"
    assert (call["metadata"]["input_tokens"], call["metadata"]["output_tokens"]) == (7, 3)


def test_architect_node_is_traced_and_llm_call_attributed(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp("SCORE: 9\nVERDICT: ok")
    state = {"project_id": "p", "understanding": "u", "clarification_round": 0}
    with patch("ArchitectAgent.Anthropic", return_value=client):
        assert aa.check_completeness(state) == {"completeness_score": 9}
    kinds = [(e["kind"], e["node"]) for e in fresh_bus.history()]
    assert kinds == [
        ("node_start", "check_completeness"),
        ("llm_call", "check_completeness"),
        ("node_end", "check_completeness"),
    ]


def test_every_architect_node_is_traced():
    for name in ARCHITECT_NODES:
        assert hasattr(getattr(aa, name), "__wrapped__"), name
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_instrumentation.py -q`
Expected: FAIL (no `llm_call` event, no `__wrapped__`).

- [ ] **Step 3: Implement**

Add after the `from forge_memory import ...` line in `ArchitectAgent.py`:

```python
from forge_events import track, traced_node
```

In `_llm`, change `client = Anthropic()` to:

```python
    client = track(Anthropic(), "architect")
```

Decorate the eight nodes:

```bash
sed -i -E 's/^def (analyse_input|generate_questions|ask_user|incorporate_answers|check_completeness|present_summary|handle_user_review|produce_architecture_spec)\(/@traced_node("architect")\ndef \1(/' ArchitectAgent.py
grep -c '^@traced_node("architect")' ArchitectAgent.py
```

Expected output of the `grep -c`: `8`.

- [ ] **Step 4: Run new tests, then the full suite**

Run: `python -m pytest tests/test_instrumentation.py -q` — Expected: 3 passed.
Run: `python -m pytest tests -q` — Expected: matches the Task 0 baseline (no new failures).

- [ ] **Step 5: Commit** (only after the user has committed/stashed their own edits — see Global Constraints)

```bash
git add ArchitectAgent.py tests/test_instrumentation.py
git commit -m "feat: instrument ArchitectAgent with event tracking"
```

---

### Task 4: Instrument WorkerAgent

**Files:**
- Modify: `WorkerAgent.py` (import line ~21, six `client = Anthropic()` sites, six node defs, `_run_tool_loop` ~248-252)
- Modify: `tests/test_instrumentation.py` (append)

**Interfaces:**
- Consumes: `track`, `traced_node`, `emit`.
- Produces: Worker `llm_call` events (`agent="worker"`), node events with `worker_id`/`task_id` from state, and a `tool_call` event per tool invocation (`metadata`: `tool`, `file_path`).

- [ ] **Step 1: Write the failing tests (append)**

```python
import WorkerAgent as wa

WORKER_NODES = [
    "plan_task", "execute_task", "review_task", "rework_task",
    "generate_tests", "report_to_teamlead",
]


def test_every_worker_node_is_traced():
    for name in WORKER_NODES:
        assert hasattr(getattr(wa, name), "__wrapped__"), name


def test_tool_loop_emits_tool_call_events(fresh_bus):
    tool_block = SimpleNamespace(
        type="tool_use", id="t1", name="write_file",
        input={"file_path": "src/a.py", "project_id": "p"},
    )
    first = SimpleNamespace(content=[tool_block], stop_reason="tool_use")
    done = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")], stop_reason="end_turn",
    )
    client = MagicMock()
    client.messages.create.side_effect = [first, done]
    with patch.object(wa, "_dispatch_tool", return_value="{}"):
        wa._run_tool_loop(client, "sys", [{"role": "user", "content": "go"}])
    tools = [e for e in fresh_bus.history() if e["kind"] == "tool_call"]
    assert len(tools) == 1
    assert tools[0]["metadata"] == {"tool": "write_file", "file_path": "src/a.py"}


def test_tool_loop_tolerates_non_dict_tool_input(fresh_bus):
    tool_block = SimpleNamespace(type="tool_use", id="t1", name="read_file", input=None)
    first = SimpleNamespace(content=[tool_block], stop_reason="tool_use")
    done = SimpleNamespace(content=[], stop_reason="end_turn")
    client = MagicMock()
    client.messages.create.side_effect = [first, done]
    with patch.object(wa, "_dispatch_tool", return_value="{}"):
        wa._run_tool_loop(client, "sys", [])
    tools = [e for e in fresh_bus.history() if e["kind"] == "tool_call"]
    assert tools[0]["metadata"]["file_path"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_instrumentation.py -q`
Expected: FAIL (`__wrapped__` missing, no `tool_call` events).

- [ ] **Step 3: Implement**

Add after the `from forge_memory import ...` line:

```python
from forge_events import emit, track, traced_node
```

Replace the six client constructions and decorate the six nodes:

```bash
sed -i 's/client = Anthropic()/client = track(Anthropic(), "worker")/' WorkerAgent.py
grep -c 'track(Anthropic(), "worker")' WorkerAgent.py
sed -i -E 's/^def (plan_task|execute_task|review_task|rework_task|generate_tests|report_to_teamlead)\(/@traced_node("worker")\ndef \1(/' WorkerAgent.py
grep -c '^@traced_node("worker")' WorkerAgent.py
```

Expected: `6`, then `6`.

In `_run_tool_loop`, immediately before `tool_result_str = _dispatch_tool(block.name, block.input)` add:

```python
            tool_input = block.input if isinstance(block.input, dict) else {}
            emit("tool_call", metadata={
                "tool": block.name,
                "file_path": tool_input.get("file_path"),
            })
```

- [ ] **Step 4: Run new tests, then the full suite**

Run: `python -m pytest tests/test_instrumentation.py -q` — Expected: all pass.
Run: `python -m pytest tests -q` — Expected: matches baseline.

- [ ] **Step 5: Commit**

```bash
git add WorkerAgent.py tests/test_instrumentation.py
git commit -m "feat: instrument WorkerAgent with events and tool-call tracking"
```

---

### Task 5: Instrument TeamLeadAgent (task status events)

**Files:**
- Modify: `TeamLeadAgent.py` (import ~line 24, three `client = Anthropic()` sites, seven node defs, `_update_task_status`, `decompose_tasks`, `schedule_iteration`)
- Modify: `tests/test_instrumentation.py` (append)

**Interfaces:**
- Consumes: `track`, `traced_node`, `emit`.
- Produces: `task_status` events: `task_id` set, `metadata` = `{"status", "dependency_list", "retry_count", "review_score", "desc"}`. Emitted on decompose (all tasks `Open`), on retry promotion, and on every `_update_task_status`.

- [ ] **Step 1: Write the failing tests (append)**

```python
import json

import TeamLeadAgent as tl

TEAMLEAD_NODES = [
    "decompose_tasks", "schedule_iteration", "dispatch_workers",
    "collect_reports", "review_batch", "handle_batch_review", "finalize",
]


def _rec(tid, deps=(), status="Open"):
    return {
        "task_id": tid, "project_id": "p", "desc": f"do {tid}\nmore", "status": status,
        "worker_id": "", "dependency_list": list(deps), "retry_count": 0,
        "report": None, "review_notes": "",
    }


def test_every_teamlead_node_is_traced():
    for name in TEAMLEAD_NODES:
        assert hasattr(getattr(tl, name), "__wrapped__"), name


def test_update_task_status_emits_task_status(fresh_bus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tl._save_registry("p", {"a": _rec("a")})
    tl._update_task_status("p", "a", "Closed", report={"review_score": 9})
    e = next(x for x in fresh_bus.history() if x["kind"] == "task_status")
    assert e["task_id"] == "a"
    assert e["metadata"]["status"] == "Closed"
    assert e["metadata"]["review_score"] == 9
    assert e["metadata"]["desc"] == "do a"


def test_decompose_emits_open_status_for_each_task(fresh_bus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks = [
        {"task_id": "a", "desc": "first", "dependency_list": []},
        {"task_id": "b", "desc": "second", "dependency_list": ["a"]},
    ]
    client = MagicMock()
    client.messages.create.return_value = _resp(json.dumps(tasks))
    state = {"project_id": "p", "architecture_spec": "spec text"}
    with patch("TeamLeadAgent.Anthropic", return_value=client), \
         patch("TeamLeadAgent.get_teamlead_context", return_value=""):
        tl.decompose_tasks(state)
    rows = {
        e["task_id"]: e["metadata"]
        for e in fresh_bus.history() if e["kind"] == "task_status"
    }
    assert set(rows) == {"a", "b"}
    assert rows["b"]["dependency_list"] == ["a"] and rows["b"]["status"] == "Open"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_instrumentation.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add after the `from forge_memory import ...` line:

```python
from forge_events import emit, track, traced_node
```

```bash
sed -i 's/client = Anthropic()/client = track(Anthropic(), "teamlead")/' TeamLeadAgent.py
grep -c 'track(Anthropic(), "teamlead")' TeamLeadAgent.py
sed -i -E 's/^def (decompose_tasks|schedule_iteration|dispatch_workers|collect_reports|review_batch|handle_batch_review|finalize)\(/@traced_node("teamlead")\ndef \1(/' TeamLeadAgent.py
grep -c '^@traced_node("teamlead")' TeamLeadAgent.py
```

Expected: `3`, then `7`.

Add this helper after `_save_registry`:

```python
def _emit_task_status(rec: Dict) -> None:
    report = rec.get("report") or {}
    first_line = (rec.get("desc", "").strip().splitlines() or [""])[0]
    emit("task_status", agent="teamlead", task_id=rec["task_id"], metadata={
        "status": rec["status"],
        "dependency_list": list(rec.get("dependency_list", [])),
        "retry_count": rec.get("retry_count", 0),
        "review_score": report.get("review_score"),
        "desc": first_line[:100],
    })
```

Call sites:
- `_update_task_status`: after `_save_registry(project_id, registry)` add `_emit_task_status(rec)`.
- `decompose_tasks`: after `_save_registry(project_id, registry)` add `for rec in registry.values(): _emit_task_status(rec)`.
- `schedule_iteration`: after the `_save_registry(project_id, registry)` that follows the retry-promotion loop add `for tid in state.get("retry_queue", []): if tid in registry: _emit_task_status(registry[tid])`.

- [ ] **Step 4: Run new tests, then the full suite**

Run: `python -m pytest tests/test_instrumentation.py -q` — Expected: all pass.
Run: `python -m pytest tests -q` — Expected: matches baseline.

- [ ] **Step 5: Commit**

```bash
git add TeamLeadAgent.py tests/test_instrumentation.py
git commit -m "feat: instrument TeamLeadAgent with events and task status updates"
```

---

### Task 6: `run_architect` with injectable human I/O

**Files:**
- Modify: `ArchitectAgent.py` (add `_initial_architect_state`, `run_architect`; make `run_architect_cli` use the helper)
- Create: `tests/test_run_architect.py`

**Interfaces:**
- Consumes: existing graph, `_extract_interrupt`.
- Produces:
  - `run_architect(project_id, raw_input, ask_questions, ask_decision) -> Optional[ArchitectureSpec]`
    - `ask_questions(questions: List[Question]) -> str` returns one answer per line.
    - `ask_decision(state_values: Dict[str, Any]) -> str` returns `"accept"`, `"change: <notes>"` or `"reject"`; `state_values` has `understanding` and `answers`.
  - `run_architect_cli` behavior unchanged.

- [ ] **Step 1: Write the failing tests (`tests/test_run_architect.py`)**

```python
import json
from unittest.mock import patch

import ArchitectAgent as aa


def _fake_llm(system, user, max_tokens=2048):
    if "rough idea" in system:
        return "## What is clear\nA CLI.\n## Initial project name suggestion\nDemo"
    if "clarifying only the most critical" in system:
        return json.dumps([{
            "id": "q1", "question": "Web or CLI?", "type": "multiple_choice",
            "options": ["web", "cli"], "reason": "shape",
        }])
    if "updating your understanding" in system:
        return "## What is clear\nA CLI.\n## Initial project name suggestion\nDemo"
    if "deciding whether requirements" in system:
        return "SCORE: 9\nVERDICT: complete"
    return "# Demo — Architecture Specification"


def _run(ask_decision):
    questions_seen = []

    def ask_questions(questions):
        questions_seen.append(questions)
        return "2"

    with patch.object(aa, "_llm", side_effect=_fake_llm), \
         patch.object(aa, "get_architect_context", return_value=""), \
         patch.object(aa, "record_architect_run"):
        spec = aa.run_architect("proj-1", "a todo cli", ask_questions, ask_decision)
    return spec, questions_seen


def test_run_architect_accept_returns_spec_and_passes_questions():
    decisions = []

    def ask_decision(values):
        decisions.append(values)
        return "accept"

    spec, questions_seen = _run(ask_decision)
    assert spec["spec_text"].startswith("# Demo")
    assert len(questions_seen) == 1 and questions_seen[0][0]["id"] == "q1"
    assert "A CLI" in decisions[0]["understanding"]
    assert decisions[0]["answers"][0]["answer"] == "cli"  # "2" normalised to option


def test_run_architect_reject_returns_none():
    spec, _ = _run(lambda values: "reject")
    assert spec is None


def test_run_architect_change_loops_back_then_accepts():
    replies = iter(["change: use sqlite", "accept"])
    spec, _ = _run(lambda values: next(replies))
    assert spec is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_run_architect.py -q`
Expected: FAIL — `AttributeError: module 'ArchitectAgent' has no attribute 'run_architect'`.

- [ ] **Step 3: Implement**

Add `Callable` to the `typing` import. Add before `run_architect_cli`:

```python
def _initial_architect_state(project_id: str, raw_input: str) -> ArchitectState:
    return {
        "project_id": project_id,
        "raw_input": raw_input,
        "understanding": "",
        "questions": [],
        "answers": [],
        "clarification_round": 0,
        "completeness_score": 0,
        "human_input": "",
        "user_review": None,
        "architecture_spec": None,
    }


def run_architect(
    project_id: str,
    raw_input: str,
    ask_questions: Callable[[List[Question]], str],
    ask_decision: Callable[[Dict[str, Any]], str],
) -> Optional[ArchitectureSpec]:
    """Drive the Architect graph, delegating human input to the callbacks."""
    agent = create_architect_agent()
    config = {"configurable": {"thread_id": project_id}}
    input_val: Any = _initial_architect_state(project_id, raw_input)

    while True:
        interrupted = False
        for event in agent.stream(input_val, config, stream_mode="updates"):
            if _extract_interrupt(event):
                interrupted = True
                break
        if not interrupted:
            break

        state = agent.get_state(config)
        next_node = state.next[0] if state.next else ""
        if next_node == "ask_user":
            reply = ask_questions(state.values.get("questions", []))
        elif next_node == "present_summary":
            reply = ask_decision(state.values)
        else:
            raise RuntimeError(f"Unexpected Architect interrupt at node {next_node!r}")
        input_val = Command(resume=reply)

    return agent.get_state(config).values.get("architecture_spec")
```

In `run_architect_cli`, replace the inline `initial_state: ArchitectState = {...}` dict with `initial_state = _initial_architect_state(project_id, raw_input)`.

- [ ] **Step 4: Run new tests, then the full suite**

Run: `python -m pytest tests/test_run_architect.py -q` — Expected: 3 passed.
Run: `python -m pytest tests -q` — Expected: matches baseline.

- [ ] **Step 5: Commit**

```bash
git add ArchitectAgent.py tests/test_run_architect.py
git commit -m "feat: add run_architect with injectable question and decision callbacks"
```

---

### Task 7: UI state reducer and dashboard rendering

**Files:**
- Create: `forge_ui.py` (state + render functions; runtime added in Task 8)
- Create: `tests/test_forge_ui.py`
- Modify: `requirements.txt` (add `rich`)

**Interfaces:**
- Consumes: event dicts from `forge_events` (`kind`, `agent`, `node`, `task_id`, `worker_id`, `success`, `error`, `metadata`, `timestamp`).
- Produces:
  - `UIState` with `.apply(event) -> Optional[str]` (returns the log line it appended, or `None`), `.set_phase(phase: str)`, `.lock` (`threading.RLock`), fields `phase`, `project_id`, `tasks: Dict[str, TaskView]`, `workers: Dict[str, WorkerView]`, `agents: Dict[str, AgentTotals]`, `input_tokens`, `output_tokens`, `cost_usd`, `log`.
  - `format_event_line(event) -> str`.
  - `render_dashboard(state) -> rich.console.Group` (header, tasks + workers side by side, usage, log).
  - Implementation choice: `Group`/grid rather than a full-height `Layout`, so the dashboard takes only the rows it needs and does not flicker to terminal height.

- [ ] **Step 1: Add the dependency**

Append `rich` as a new line in `requirements.txt`.

- [ ] **Step 2: Write the failing tests (`tests/test_forge_ui.py`)**

```python
import io

from rich.console import Console

from forge_ui import UIState, render_dashboard


def ev(kind, **kw):
    base = {
        "event_id": "e", "run_id": "r", "kind": kind, "project_id": "p",
        "agent": "", "node": "", "task_id": None, "worker_id": None,
        "duration_seconds": 0.0, "success": True, "error": None,
        "metadata": {}, "timestamp": "2026-10-06T12:04:11+00:00",
    }
    base.update(kw)
    return base


def llm(agent, in_t, out_t, cost, **kw):
    return ev("llm_call", agent=agent, metadata={
        "model": "claude-haiku-4-5", "input_tokens": in_t,
        "output_tokens": out_t, "cost_usd": cost, "price_known": True,
    }, **kw)


def test_llm_calls_accumulate_totals_per_agent_and_overall():
    s = UIState()
    s.apply(llm("architect", 100, 20, 0.001))
    s.apply(llm("worker", 300, 80, 0.003, worker_id="worker-a", task_id="a"))
    assert (s.input_tokens, s.output_tokens) == (400, 100)
    assert s.cost_usd == 0.004
    assert s.agents["worker"].input_tokens == 300 and s.agents["worker"].calls == 1
    assert s.workers["worker-a"].input_tokens == 300


def test_worker_lifecycle_node_tool_rework_score_done():
    s = UIState()
    w = dict(agent="worker", worker_id="worker-a", task_id="a")
    s.apply(ev("node_start", node="plan_task", **w))
    assert s.workers["worker-a"].node == "plan_task"
    s.apply(ev("tool_call", node="execute_task", metadata={"tool": "write_file", "file_path": "x.py"}, **w))
    assert s.workers["worker-a"].tool == "write_file"
    s.apply(ev("node_start", node="rework_task", **w))
    assert s.workers["worker-a"].rework == 1 and s.workers["worker-a"].tool == ""
    s.apply(ev("task_status", agent="teamlead", task_id="a",
               metadata={"status": "Closed", "dependency_list": [], "retry_count": 0, "review_score": 8, "desc": "d"}))
    assert s.workers["worker-a"].score == 8
    s.apply(ev("node_end", node="report_to_teamlead", **w))
    assert s.workers["worker-a"].done is True


def test_task_status_builds_task_table():
    s = UIState()
    s.apply(ev("task_status", agent="teamlead", task_id="b",
               metadata={"status": "Open", "dependency_list": ["a"], "retry_count": 1, "review_score": None, "desc": "second"}))
    t = s.tasks["b"]
    assert (t.status, t.deps, t.retries) == ("Open", ["a"], 1)


def test_failed_node_is_logged_and_log_is_bounded():
    s = UIState()
    line = s.apply(ev("node_end", agent="teamlead", node="finalize", success=False, error="ValueError('x')"))
    assert "finalize" in line and "ValueError" in line
    for i in range(30):
        s.apply(ev("node_start", agent="teamlead", node=f"n{i}"))
    assert len(s.log) == UIState.LOG_SIZE


def test_prompt_user_and_run_end_update_phase_and_log():
    s = UIState()
    assert "input" in s.apply(ev("prompt_user", agent="ui", metadata={"kind": "questions"})).lower()
    s.apply(ev("run_end"))
    assert s.phase == "done"


def render_text(state):
    console = Console(file=io.StringIO(), width=120, record=True, force_terminal=False)
    console.print(render_dashboard(state))
    return console.export_text()


def test_render_smoke_shows_tasks_workers_and_usage():
    s = UIState()
    s.project_id = "proj-1"
    s.set_phase("execution")
    s.apply(ev("task_status", agent="teamlead", task_id="auth-model",
               metadata={"status": "InProgress", "dependency_list": [], "retry_count": 0, "review_score": None, "desc": "model"}))
    s.apply(ev("node_start", agent="worker", node="execute_task", worker_id="worker-auth-model", task_id="auth-model"))
    s.apply(llm("worker", 4100, 1200, 0.0101, worker_id="worker-auth-model", task_id="auth-model"))
    text = render_text(s)
    for expected in ("proj-1", "auth-model", "InProgress", "worker-auth-model", "execute_task", "worker", "$0.0101"):
        assert expected in text, expected


def test_render_empty_state_does_not_crash():
    assert "forge" in render_text(UIState()).lower()


def test_render_treats_dynamic_text_literally_not_as_markup():
    s = UIState()
    s.apply(ev("task_status", agent="teamlead", task_id="[red]t[/oops]",
               metadata={"status": "Open", "dependency_list": ["[bold"], "retry_count": 0, "review_score": None, "desc": "[/]x"}))
    s.apply(ev("node_end", agent="teamlead", node="finalize", success=False, error="[/oops] [red"))
    text = render_text(s)  # must not raise MarkupError
    assert "[/oops]" in text and "[red]t" in text
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_forge_ui.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'forge_ui'`.

- [ ] **Step 4: Implement `forge_ui.py`**

```python
"""
forge_ui.py — Rich terminal UI for forge-ops.

Part 1 (this file's state/render half): fold bus events into UIState and render
a dashboard. Part 2 (ForgeUI runtime) is added in the next task.
Imports only plain-data helpers; never imports agents.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STATUS_STYLE = {
    "Open": "dim",
    "InProgress": "yellow",
    "PendingReview": "magenta",
    "ReworkRequired": "red",
    "Closed": "green",
}


@dataclass
class TaskView:
    task_id: str
    status: str = "Open"
    deps: List[str] = field(default_factory=list)
    retries: int = 0
    score: Optional[int] = None


@dataclass
class WorkerView:
    worker_id: str
    task_id: str = ""
    node: str = ""
    tool: str = ""
    rework: int = 0
    score: Optional[int] = None
    input_tokens: int = 0
    output_tokens: int = 0
    done: bool = False


@dataclass
class AgentTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


def format_event_line(event: Dict[str, Any]) -> str:
    ts = (event.get("timestamp") or "")[11:19]
    who = event.get("worker_id") or event.get("agent") or "forge"
    kind = event["kind"]
    md = event.get("metadata") or {}
    if kind == "node_start":
        text = f"▶ {event['node']}"
    elif kind == "node_end":
        text = f"✔ {event['node']}" if event.get("success") else f"✖ {event['node']} failed: {event.get('error')}"
    elif kind == "tool_call":
        text = f"tool {md.get('tool')} {md.get('file_path') or ''}".rstrip()
    elif kind == "llm_call":
        text = f"llm {md.get('input_tokens', 0)}/{md.get('output_tokens', 0)} tok"
    elif kind == "task_status":
        text = f"task {event.get('task_id')} → {md.get('status')}"
    elif kind == "prompt_user":
        text = "waiting for your input"
    else:
        text = kind
    return f"{ts} {who} {text}".strip()


class UIState:
    LOG_SIZE = 8

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.phase = "starting"
        self.project_id = ""
        self.started = time.monotonic()
        self.tasks: Dict[str, TaskView] = {}
        self.workers: Dict[str, WorkerView] = {}
        self.agents: Dict[str, AgentTotals] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.log: Deque[str] = deque(maxlen=self.LOG_SIZE)

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def _worker(self, event: Dict[str, Any]) -> Optional[WorkerView]:
        wid = event.get("worker_id")
        if not wid:
            return None
        w = self.workers.setdefault(wid, WorkerView(worker_id=wid))
        if event.get("task_id"):
            w.task_id = event["task_id"]
        return w

    def apply(self, event: Dict[str, Any]) -> Optional[str]:
        kind = event["kind"]
        md = event.get("metadata") or {}
        with self.lock:
            w = self._worker(event)
            if kind == "llm_call":
                in_t, out_t = md.get("input_tokens", 0), md.get("output_tokens", 0)
                cost = md.get("cost_usd", 0.0)
                totals = self.agents.setdefault(event.get("agent") or "?", AgentTotals())
                totals.input_tokens += in_t
                totals.output_tokens += out_t
                totals.cost_usd += cost
                totals.calls += 1
                self.input_tokens += in_t
                self.output_tokens += out_t
                self.cost_usd += cost
                if w:
                    w.input_tokens += in_t
                    w.output_tokens += out_t
            elif kind == "node_start" and w:
                w.node = event["node"]
                w.tool = ""
                if event["node"] == "rework_task":
                    w.rework += 1
            elif kind == "node_end" and w and event["node"] == "report_to_teamlead":
                w.done = True
                w.node = "done"
            elif kind == "tool_call" and w:
                w.tool = md.get("tool", "")
            elif kind == "task_status":
                tid = event.get("task_id") or ""
                t = self.tasks.setdefault(tid, TaskView(task_id=tid))
                t.status = md.get("status", t.status)
                t.deps = list(md.get("dependency_list", t.deps))
                t.retries = md.get("retry_count", t.retries)
                if md.get("review_score") is not None:
                    t.score = md["review_score"]
                    for worker in self.workers.values():
                        if worker.task_id == tid:
                            worker.score = md["review_score"]
            elif kind == "run_end":
                self.phase = "done"

            line = format_event_line(event)
            self.log.append(line)
            return line


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _render_header(state: UIState) -> Panel:
    elapsed = int(time.monotonic() - state.started)
    text = Text()
    text.append("forge ", style="bold cyan")
    text.append(f"· {state.project_id or '-'} · phase: {state.phase} · {elapsed // 60}:{elapsed % 60:02d}"
                f" · tokens {_fmt_tokens(state.input_tokens)} in / {_fmt_tokens(state.output_tokens)} out"
                f" · est. ${state.cost_usd:.4f}")
    return Panel(text, padding=(0, 1))


def _render_tasks(state: UIState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    for col in ("Task", "Status", "Deps", "Try", "Score"):
        table.add_column(col)
    for t in state.tasks.values():
        table.add_row(
            Text(t.task_id),
            Text(t.status, style=STATUS_STYLE.get(t.status, "")),
            Text(", ".join(t.deps) or "-"),
            Text(str(t.retries)),
            Text("-" if t.score is None else str(t.score)),
        )
    return Panel(table if state.tasks else Text("no tasks yet", style="dim"), title="Tasks", padding=(0, 1))


def _render_workers(state: UIState) -> Panel:
    if not state.workers:
        return Panel(Text("no workers running", style="dim"), title="Workers", padding=(0, 1))
    blocks: List[Text] = []
    for w in state.workers.values():
        t = Text()
        t.append(w.worker_id, style="bold")
        t.append(f"\n  node: {w.node or '-'}")
        if w.tool:
            t.append(f"  tool: {w.tool}")
        t.append(f"\n  rework {w.rework} · tokens {_fmt_tokens(w.input_tokens)}/{_fmt_tokens(w.output_tokens)}"
                 f" · score {'-' if w.score is None else w.score}")
        blocks.append(t)
    return Panel(Group(*blocks), title="Workers", padding=(0, 1))


def _render_usage(state: UIState) -> Panel:
    table = Table(expand=True, box=None, pad_edge=False)
    for col in ("Agent", "Calls", "In", "Out", "Est. cost"):
        table.add_column(col)
    for name, a in state.agents.items():
        table.add_row(Text(name), Text(str(a.calls)), Text(_fmt_tokens(a.input_tokens)),
                      Text(_fmt_tokens(a.output_tokens)), Text(f"${a.cost_usd:.4f}"))
    return Panel(table if state.agents else Text("no LLM calls yet", style="dim"),
                 title="Token usage", padding=(0, 1))


def _render_log(state: UIState) -> Panel:
    body = Text("\n".join(state.log)) if state.log else Text("no events yet", style="dim")
    return Panel(body, title="Events", padding=(0, 1))


def render_dashboard(state: UIState) -> Group:
    with state.lock:
        row = Table.grid(expand=True)
        row.add_column(ratio=1)
        row.add_column(ratio=1)
        row.add_row(_render_tasks(state), _render_workers(state))
        return Group(_render_header(state), row, _render_usage(state), _render_log(state))
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_forge_ui.py -q`
Expected: all pass (9).

- [ ] **Step 6: Commit**

```bash
git add forge_ui.py tests/test_forge_ui.py requirements.txt
git commit -m "feat: add UI state reducer and Rich dashboard rendering"
```

---

### Task 8: ForgeUI runtime (threads, prompts, plain mode)

**Files:**
- Modify: `forge_ui.py` (append `ForgeUI`)
- Modify: `tests/test_forge_ui.py` (append)

**Interfaces:**
- Consumes: `EventBus`, `emit` (`forge_events`), `UIState`, `render_dashboard` (Task 7).
- Produces:
  - `NO_ANSWER = "(no answer)"`.
  - `ForgeUI(bus, console=None)` with `.state: UIState`, `.request_input(kind, payload) -> str` (`kind` is `"questions"` with `{"questions": [...]}` or `"decision"` with `{"understanding": str, "answers": [...]}`; blocks the calling thread until the main thread answers), `.run(pipeline: Callable[[], Any]) -> Any` (main thread; runs `pipeline` in a background thread, renders live, services prompts, re-raises pipeline exceptions after restoring the terminal), `.print_summary(report: Optional[dict])`.
  - Decision replies use the Architect's format: `"accept"`, `"change: <notes>"`, `"reject"`. Question replies are one line per question.

- [ ] **Step 1: Write the failing tests (append to `tests/test_forge_ui.py`)**

```python
import builtins
import threading

import pytest

from forge_events import EventBus
from forge_ui import NO_ANSWER, ForgeUI


def make_ui(terminal=False):
    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=terminal, color_system=None)
    return ForgeUI(EventBus(), console), out


def feed_input(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))


QUESTIONS = [
    {"id": "q1", "question": "Web or CLI?", "type": "multiple_choice", "options": ["web", "cli"], "reason": "shape"},
    {"id": "q2", "question": "Persist data?", "type": "yes_no", "options": None, "reason": "storage"},
    {"id": "q3", "question": "Anything else?", "type": "text", "options": None, "reason": ""},
]


@pytest.mark.parametrize("terminal", [False, True])
def test_run_services_questions_from_pipeline_thread(monkeypatch, terminal):
    ui, out = make_ui(terminal)
    feed_input(monkeypatch, ["2", "y", "use sqlite"])
    result = ui.run(lambda: ui.request_input("questions", {"questions": QUESTIONS}))
    assert result.split("\n") == ["2", "yes", "use sqlite"]
    assert "Web or CLI?" in out.getvalue()


def test_blank_text_answer_becomes_placeholder_so_answers_do_not_shift(monkeypatch):
    ui, _ = make_ui()
    feed_input(monkeypatch, ["", "n"])
    questions = [QUESTIONS[2], QUESTIONS[1]]  # text first, then yes/no
    result = ui.run(lambda: ui.request_input("questions", {"questions": questions}))
    assert result.split("\n") == [NO_ANSWER, "no"]


def test_multiline_text_answer_is_collapsed_to_one_line(monkeypatch):
    ui, _ = make_ui()
    feed_input(monkeypatch, ["  a   b  "])
    result = ui.run(lambda: ui.request_input("questions", {"questions": [QUESTIONS[2]]}))
    assert result == "a b"


def test_decision_accept(monkeypatch):
    ui, out = make_ui()
    feed_input(monkeypatch, ["accept"])
    payload = {"understanding": "## What is clear\nA [bold CLI", "answers": [{"question": "q", "answer": "a"}]}
    assert ui.run(lambda: ui.request_input("decision", payload)) == "accept"
    assert "What is clear" in out.getvalue()


def test_decision_change_reasks_until_notes_given(monkeypatch):
    ui, _ = make_ui()
    feed_input(monkeypatch, ["change", "", "use sqlite"])
    payload = {"understanding": "u", "answers": []}
    assert ui.run(lambda: ui.request_input("decision", payload)) == "change: use sqlite"


def test_pipeline_exception_is_reraised_and_shown(monkeypatch):
    ui, out = make_ui(terminal=True)

    def boom():
        raise RuntimeError("boom [/x]")

    with pytest.raises(RuntimeError):
        ui.run(boom)
    assert "boom [/x]" in out.getvalue()


def test_plain_mode_prints_event_lines_without_live():
    ui, out = make_ui(terminal=False)
    from forge_events import emit, set_bus
    set_bus(ui._bus)
    try:
        ui.run(lambda: emit("node_start", agent="architect", node="analyse_input"))
    finally:
        set_bus(EventBus())
    assert "analyse_input" in out.getvalue()


def test_run_returns_pipeline_value_and_prints_final_dashboard():
    ui, out = make_ui(terminal=True)
    ui.state.project_id = "proj-9"
    assert ui.run(lambda: 42) == 42
    assert "proj-9" in out.getvalue()


def test_print_summary_handles_none_and_report():
    ui, out = make_ui()
    ui.print_summary(None)
    ui.print_summary({
        "project_id": "p", "final_status": "success", "total_tasks": 2, "completed_tasks": 2,
        "failed_tasks": 0, "blocked_tasks": 0, "all_touched_files": ["src/a.py"],
        "batch_review_scores": [8, 9], "summary": "ok [/x]", "completed_at": "t",
    })
    text = out.getvalue()
    assert "src/a.py" in text and "success" in text and "ok [/x]" in text
```

In `test_plain_mode_prints_event_lines_without_live`, the UI must subscribe to the bus given to its constructor; store it as `self._bus`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_forge_ui.py -q`
Expected: FAIL — `ImportError: cannot import name 'ForgeUI'`.

- [ ] **Step 3: Implement (append to `forge_ui.py`)**

Add imports at the top of the file:

```python
import queue
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from rich.rule import Rule

from forge_events import EventBus, emit
```

Then append:

```python
NO_ANSWER = "(no answer)"  # ArchitectAgent.ask_user drops blank lines; never send one


@dataclass
class _InputRequest:
    kind: str
    payload: Dict[str, Any]
    reply: "queue.Queue[str]" = field(default_factory=queue.Queue)


class ForgeUI:
    def __init__(self, bus: EventBus, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self.state = UIState()
        self._bus = bus
        self._plain = not self.console.is_terminal
        self._requests: "queue.Queue[_InputRequest]" = queue.Queue()
        self._done = threading.Event()
        bus.subscribe(self._on_event)

    # -- bus ---------------------------------------------------------------
    def _on_event(self, event: Dict[str, Any]) -> None:
        line = self.state.apply(event)
        if self._plain and line:
            self.console.print(Text(line))

    # -- called from the pipeline thread -----------------------------------
    def request_input(self, kind: str, payload: Dict[str, Any]) -> str:
        emit("prompt_user", agent="ui", metadata={"kind": kind})
        request = _InputRequest(kind, payload)
        self._requests.put(request)
        return request.reply.get()

    # -- main thread -------------------------------------------------------
    def run(self, pipeline: Callable[[], Any]) -> Any:
        outcome: Dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["value"] = pipeline()
            except BaseException as exc:  # surfaced on the main thread below
                outcome["error"] = exc
            finally:
                self._done.set()

        thread = threading.Thread(target=target, name="forge-pipeline", daemon=True)
        live = None if self._plain else Live(
            render_dashboard(self.state), console=self.console,
            refresh_per_second=4, transient=True,
        )
        thread.start()
        if live:
            live.start()
        try:
            while not self._done.is_set():
                try:
                    request = self._requests.get(timeout=0.25)
                except queue.Empty:
                    if live:
                        live.update(render_dashboard(self.state))
                    continue
                if live:
                    live.stop()
                request.reply.put(self._prompt(request))
                if live:
                    live.start()
        except KeyboardInterrupt:
            self.console.print(Text(
                "Interrupted. Waiting for in-flight workers to finish "
                "(Ctrl+C again to force quit).", style="yellow"))
            raise
        finally:
            if live:
                live.stop()

        if "error" in outcome:
            self.console.print(Panel(Text(f"{type(outcome['error']).__name__}: {outcome['error']}"),
                                     title="Pipeline failed", border_style="red"))
            raise outcome["error"]
        if not self._plain:
            self.console.print(render_dashboard(self.state))
        return outcome.get("value")

    # -- prompts -----------------------------------------------------------
    def _prompt(self, request: _InputRequest) -> str:
        if request.kind == "questions":
            return self._ask_questions(request.payload["questions"])
        if request.kind == "decision":
            return self._ask_decision(request.payload)
        raise ValueError(f"unknown input request kind: {request.kind!r}")

    def _ask_questions(self, questions: List[Dict[str, Any]]) -> str:
        self.console.print(Rule("Clarifying questions"))
        answers = [self._ask_one(i, q) for i, q in enumerate(questions, 1)]
        return "\n".join(answers)

    def _ask_one(self, index: int, q: Dict[str, Any]) -> str:
        self.console.print(Text(f"\nQ{index}. {q['question']}", style="bold"))
        if q.get("reason"):
            self.console.print(Text(f"Why: {q['reason']}", style="dim"))
        qtype, options = q.get("type", "text"), q.get("options") or []
        if qtype == "yes_no":
            return "yes" if Confirm.ask("  Answer", console=self.console) else "no"
        if qtype == "multiple_choice" and options:
            for n, opt in enumerate(options, 1):
                self.console.print(Text(f"  {n}. {opt}"))
            return Prompt.ask("  Answer", console=self.console,
                              choices=[str(n) for n in range(1, len(options) + 1)])
        text = " ".join(Prompt.ask("  Answer", console=self.console, default="").split())
        return text or NO_ANSWER

    def _ask_decision(self, payload: Dict[str, Any]) -> str:
        self.console.print(Rule("Architect's understanding"))
        self.console.print(Markdown(payload["understanding"]))
        answers = payload.get("answers") or []
        if answers:
            table = Table(title="Your clarifications", box=None)
            table.add_column("Question")
            table.add_column("Answer")
            for a in answers:
                table.add_row(Text(a["question"]), Text(a["answer"]))
            self.console.print(table)
        choice = Prompt.ask("Do you accept this understanding?", console=self.console,
                            choices=["accept", "change", "reject"], default="accept")
        if choice != "change":
            return choice
        notes = ""
        while not notes:
            notes = " ".join(Prompt.ask("What should change?", console=self.console, default="").split())
        return f"change: {notes}"

    # -- end of run --------------------------------------------------------
    def print_summary(self, report: Optional[Dict[str, Any]]) -> None:
        if report is None:
            self.console.print(Panel(Text("No project was produced (aborted or rejected)."),
                                     title="forge", border_style="yellow"))
            return
        body = Text()
        body.append(f"Status: {report['final_status']}\n", style="bold")
        body.append(f"Tasks: {report['completed_tasks']}/{report['total_tasks']} completed, "
                    f"{report['failed_tasks']} failed, {report['blocked_tasks']} blocked\n")
        if report.get("batch_review_scores"):
            scores = report["batch_review_scores"]
            body.append(f"Avg batch review: {sum(scores) / len(scores):.1f}/10\n")
        body.append(f"Tokens: {_fmt_tokens(self.state.input_tokens)} in / "
                    f"{_fmt_tokens(self.state.output_tokens)} out · est. ${self.state.cost_usd:.4f}\n")
        for fp in sorted(report.get("all_touched_files", [])):
            body.append(f"  - {fp}\n")
        body.append(f"\n{report['summary']}")
        self.console.print(Panel(body, title=f"Project {report['project_id']}", border_style="green"))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_forge_ui.py -q`
Expected: all pass. If a Live/terminal test is flaky on a given platform, report it; do not weaken the assertions.

- [ ] **Step 5: Commit**

```bash
git add forge_ui.py tests/test_forge_ui.py
git commit -m "feat: add ForgeUI runtime with threaded pipeline, prompts and plain mode"
```

---

### Task 9: `forge.py` entry point

**Files:**
- Create: `forge.py`
- Create: `tests/test_forge_cli.py`

**Interfaces:**
- Consumes: `run_architect` (Task 6), `run_teamlead` (existing, `run_teamlead(project_id, architecture_spec: str)`), `ForgeUI`, `get_bus`, `emit`, `telemetry_subscriber`.
- Produces: `main(argv=None) -> int` (exit codes: 0 success/partial, 1 failed/aborted/error, 2 usage/config error, 130 Ctrl+C), `run_pipeline(idea, project_id, ui, project_dir) -> Optional[dict]`, `bootstrap_project(project_id) -> Path`, `make_project_id(idea) -> str`.

- [ ] **Step 1: Write the failing tests (`tests/test_forge_cli.py`)**

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import forge


def test_make_project_id_is_slug_plus_suffix_and_unique():
    a = forge.make_project_id("Build a Todo CLI!!")
    b = forge.make_project_id("Build a Todo CLI!!")
    assert a.startswith("build-a-todo-cli") and a != b
    assert all(c.isalnum() or c == "-" for c in a)


def test_make_project_id_handles_symbol_only_idea():
    assert forge.make_project_id("!!!").startswith("project-")


def test_bootstrap_project_creates_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(forge, "PROJECT_ROOT", tmp_path)
    d = forge.bootstrap_project("p1")
    assert (d / "src").is_dir() and (d / "tests").is_dir()
    assert (d / "locks.json").read_text() == "{}"
    assert (d / "project_info.json").read_text() == "{}"


def test_main_without_api_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(forge, "load_dotenv", lambda *a, **k: None)
    assert forge.main(["an idea"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_run_pipeline_stops_when_architect_is_rejected(tmp_path):
    ui = MagicMock()
    with patch.object(forge, "run_architect", return_value=None), \
         patch.object(forge, "run_teamlead") as teamlead:
        assert forge.run_pipeline("idea", "p", ui, tmp_path) is None
    teamlead.assert_not_called()


def test_run_pipeline_passes_spec_text_to_teamlead_and_saves_spec(tmp_path):
    ui = MagicMock()
    spec = {"project_id": "p", "project_name": "Demo", "spec_text": "# Spec", "created_at": "t"}
    report = {"final_status": "success"}
    with patch.object(forge, "run_architect", return_value=spec), \
         patch.object(forge, "run_teamlead", return_value=report) as teamlead:
        assert forge.run_pipeline("idea", "p", ui, tmp_path) == report
    teamlead.assert_called_once_with("p", "# Spec")
    assert (tmp_path / "architecture_spec.md").read_text() == "# Spec"
    assert ui.state.set_phase.call_args_list[0].args == ("architect",)
    assert ui.state.set_phase.call_args_list[1].args == ("execution",)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_forge_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'forge'`.

- [ ] **Step 3: Implement `forge.py`**

```python
"""
forge.py — single entry point: idea → Architect → TeamLead → Workers, with a Rich UI.

    python forge.py "build a todo CLI"
    python forge.py            # prompts for the idea
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.prompt import Prompt

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    def load_dotenv(*_a: Any, **_k: Any) -> bool:
        return False

from ArchitectAgent import run_architect
from TeamLeadAgent import run_teamlead
from forge_events import emit, get_bus, telemetry_subscriber
from forge_ui import ForgeUI

PROJECT_ROOT = Path(__file__).resolve().parent


def make_project_id(idea: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", idea[:20].lower()).strip("-") or "project"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def bootstrap_project(project_id: str) -> Path:
    project_dir = PROJECT_ROOT / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)
    for name in ("locks.json", "project_info.json"):
        path = project_dir / name
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
    return project_dir


def run_pipeline(idea: str, project_id: str, ui: ForgeUI, project_dir: Path) -> Optional[Dict[str, Any]]:
    ui.state.set_phase("architect")
    spec = run_architect(
        project_id,
        idea,
        ask_questions=lambda questions: ui.request_input("questions", {"questions": questions}),
        ask_decision=lambda values: ui.request_input(
            "decision",
            {"understanding": values.get("understanding", ""), "answers": values.get("answers", [])},
        ),
    )
    if spec is None:
        emit("run_end", agent="forge", success=False, metadata={"reason": "architect rejected"})
        return None

    (project_dir / "architecture_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (project_dir / "architecture_spec.md").write_text(spec["spec_text"], encoding="utf-8")

    ui.state.set_phase("execution")
    report = run_teamlead(project_id, spec["spec_text"])
    emit("run_end", agent="forge", success=report["final_status"] != "failed",
         metadata={"final_status": report["final_status"]})
    return report


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(prog="forge", description="Idea to code with a live agent dashboard.")
    parser.add_argument("idea", nargs="?", help="project idea (prompted if omitted)")
    parser.add_argument("--project-id", help="reuse or choose a project id")
    args = parser.parse_args(argv)

    console = Console()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY is not set.[/] Put it in .env or the environment.")
        return 2

    idea = (args.idea or Prompt.ask("Describe your project idea", console=console)).strip()
    if not idea:
        console.print("[red]No idea given.[/]")
        return 2

    project_id = args.project_id or make_project_id(idea)
    project_dir = bootstrap_project(project_id)

    # Agent logging would corrupt the live display; send it to a file instead.
    log_path = project_dir / "forge.log"
    logging.basicConfig(
        level=logging.INFO, filename=log_path, encoding="utf-8", force=True,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    bus = get_bus()
    bus.project_id = project_id
    bus.subscribe(telemetry_subscriber)
    ui = ForgeUI(bus, console)
    ui.state.project_id = project_id

    try:
        report = ui.run(lambda: run_pipeline(idea, project_id, ui, project_dir))
    except KeyboardInterrupt:
        return 130
    except Exception:
        console.print(f"Details are in {log_path}")
        return 1

    ui.print_summary(report)
    console.print(f"Log: {log_path}")
    return 0 if report and report["final_status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify pass, then the full suite**

Run: `python -m pytest tests/test_forge_cli.py -q` — Expected: 6 passed.
Run: `python -m pytest tests -q` — Expected: matches baseline plus all new tests passing.

- [ ] **Step 5: Manual smoke test (needs a real `ANTHROPIC_API_KEY`; costs a few cents on Haiku)**

Run: `python forge.py "a tiny python CLI that prints a greeting"`
Expected: Rich prompts for any Architect questions; then a live dashboard with a growing task table, per-worker blocks, token/cost header; final project panel; `projects/<id>/forge.log` and `projects/_telemetry/agent_events.jsonl` populated. Report what you observed, including anything that looked wrong.

- [ ] **Step 6: Commit**

```bash
git add forge.py tests/test_forge_cli.py
git commit -m "feat: add forge entry point wiring architect, teamlead and the Rich UI"
```

---

### Task 10: Remove duplicate TypedDicts in `models.py`

`models.py` defines `Question`, `UserAnswer`, `RequirementsReview`, `ArchitectureSpec` twice (identical copies, lines ~85-127).

**Files:**
- Modify: `models.py`
- Test: existing suite

**Interfaces:**
- Consumes/Produces: the same four names remain importable from `models`.

- [ ] **Step 1: Confirm the duplicates**

Run: `grep -n "^class \(Question\|UserAnswer\|RequirementsReview\|ArchitectureSpec\)" models.py`
Expected: each name listed twice.

- [ ] **Step 2: Delete the second copy of the four classes** (the block starting at the second `class Question(TypedDict):` through the end of the second `ArchitectureSpec`, immediately before `class RunRecord`). Keep one copy of each.

- [ ] **Step 3: Verify**

Run: `grep -c "^class \(Question\|UserAnswer\|RequirementsReview\|ArchitectureSpec\)" models.py` — Expected: `4`.
Run: `python -m pytest tests -q` — Expected: matches the post-Task-9 result.

- [ ] **Step 4: Commit**

```bash
git add models.py
git commit -m "refactor: remove duplicate TypedDict definitions in models"
```

---

## Self-Review Notes

- **Spec coverage:** event bus/replay/thread-safety (T1); `track` incl. missing usage (T2); `traced_node` incl. pause handling and worker identity (T2); price table and cost (T1); telemetry persistence (T1, T9 wiring); call-site edits (T3-T5); Architect interactive flow in UI incl. Windows stdin avoidance (T6, T8); dashboard regions (T7); Live stop/start around prompts, error panel, Ctrl+C, non-TTY fallback (T8); entry point and summary (T8, T9); `models.py` dedupe (T10); `rich` in requirements (T7). Spec's "Layout" became `Group`/grid (noted in T7). Spec's "set context at thread start" became state-derived context (noted in Global Constraints). Environment fix (T0) was not in the spec: required for anything to run on Windows.
- **Type consistency:** `emit`, `track`, `traced_node`, `UIState.apply`, `ForgeUI.request_input/run/print_summary`, `run_architect`, `NO_ANSWER` names match across tasks. `ForgeUI._bus` is used by one test and set in the constructor.
- **Known limitation:** `track` wraps `messages.create` only; `messages.stream` is not instrumented (no agent uses it).
