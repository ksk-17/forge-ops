"""
forge_ui.py — Rich terminal UI for forge-ops.

Part 1 (this half): fold bus events into UIState and render a dashboard.
Part 2 (ForgeUI runtime) follows below. Imports only plain-data helpers;
never imports agents.
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
