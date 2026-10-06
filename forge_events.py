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
