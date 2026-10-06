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


# ── cancellation ───────────────────────────────────────────────────────────

class RunCancelled(Exception):
    """Raised at the next node or LLM-call boundary after the user cancels."""


_cancel = threading.Event()


def cancel_run() -> None:
    _cancel.set()


def clear_cancel() -> None:
    _cancel.clear()


def is_cancelled() -> bool:
    return _cancel.is_set()


def _raise_if_cancelled() -> None:
    if _cancel.is_set():
        raise RunCancelled("run cancelled by user")


# ── LLM-call tracking ──────────────────────────────────────────────────────

def _as_int(value: Any) -> int:
    """Token counts must be real ints; anything else (None, MagicMock) is 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class _TrackedMessages:
    def __init__(self, messages: Any, agent: str) -> None:
        self._messages = messages
        self._agent = agent

    def create(self, **kwargs: Any) -> Any:
        _raise_if_cancelled()
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
            _raise_if_cancelled()
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
