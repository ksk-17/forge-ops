from __future__ import annotations

import json
import logging
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from anthropic import Anthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from models import BatchReview, ProjectReport, Task, TaskRecord
from WorkerTools import (
    get_project_schema,
    read_file,
    create_file,
    write_file,
)
from WorkerAgent import run_worker, WorkerReport
 
logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
MAX_WORKER_PARALLELISM = 5
MAX_RETRY_CYCLES = 2
INTEGRATION_PASS_THRESHOLD = 7
TASKS_FILE = "projects/{project_id}/tasks.json"
PROJECT_REPORT_FILE = "projects/{project_id}/project_report.json"

class TeamLeadState(TypedDict):
    project_id: str
    architecture_spec: str
    all_task_ids: List[str]
    current_batch: List[str]
    iteration: int
    worker_reports: Dict[str, Dict]
    all_touched_files: List[str]
    batch_reviews: List[BatchReview]
    latest_review: Optional[BatchReview]
    retry_queue: List[str]
    messages: Annotated[List[Dict], add_messages]
    project_report: Optional[ProjectReport]

def _tasks_path(project_id: str) -> Path:
    return Path(TASKS_FILE.format(project_id=project_id))

def _load_registry(project_id: str) -> Dict[str, TaskRecord]:
    """Load tasks.json → {task_id: TaskRecord}."""
    p = _tasks_path(project_id)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        raw: List[Dict] = json.load(f)
    return {t["task_id"]: t for t in raw}
 
 
def _save_registry(project_id: str, registry: Dict[str, TaskRecord]) -> None:
    """Persist {task_id: TaskRecord} → tasks.json."""
    p = _tasks_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(list(registry.values()), f, indent=2)

def _update_task_status(
    project_id: str,
    task_id: str,
    status: str,
    report: Optional[Dict] = None,
    review_notes: str = "",
    retry_count: Optional[int] = None,
) -> None:
    """Atomically update a single task's fields in the registry."""
    registry = _load_registry(project_id)
    rec = registry[task_id]
    rec["status"] = status
    if report is not None:
        rec["report"] = report
    if review_notes:
        rec["review_notes"] = review_notes
    if retry_count is not None:
        rec["retry_count"] = retry_count
    registry[task_id] = rec
    _save_registry(project_id, registry)

def _ready_tasks(registry: Dict[str, TaskRecord]) -> List[TaskRecord]:
    """
    Return tasks that are Open AND whose every dependency is Closed.
    This is the scheduling predicate — called each iteration.
    """
    closed = {tid for tid, rec in registry.items() if rec["status"] == "Closed"}
    return [
        rec for rec in registry.values()
        if rec["status"] == "Open"
        and all(dep in closed for dep in rec["dependency_list"])
    ]
 
def _counts(registry: Dict[str, TaskRecord]) -> Dict[str, int]:
    """Return status → count breakdown."""
    counts: Dict[str, int] = {}
    for rec in registry.values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    return counts

def _parse_score(text: str) -> int:
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("SCORE:"):
            try:
                return max(0, min(10, int(s.split(":", 1)[1].strip())))
            except ValueError:
                pass
    return 0
 
 
def _parse_list_section(text: str, section: str) -> List[str]:
    """Extract bullet items under a section header like 'ISSUES:' or 'SUGGESTIONS:'."""
    items: List[str] = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith(f"{section}:"):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item.lower() not in ("none", ""):
                    items.append(item)
            elif stripped and not stripped.startswith("-"):
                # Hit the next section header
                break
    return items
 
 
def _parse_summary(text: str) -> str:
    """Extract text after 'SUMMARY:' label."""
    lines = text.splitlines()
    in_summary = False
    parts: List[str] = []
    for line in lines:
        if line.strip().upper().startswith("SUMMARY:"):
            in_summary = True
            rest = line.split(":", 1)[1].strip()
            if rest:
                parts.append(rest)
            continue
        if in_summary:
            parts.append(line.strip())
    return " ".join(parts).strip()

# nodes
def decompose_tasks(state: TeamLeadState) -> Dict[str, Any]:
    project_id = state["project_id"]
    arch_spec = state["architecture_spec"]
    client = Anthropic()
 
    # Read existing schema so we don't duplicate already-existing files
    try:
        existing_schema = get_project_schema(project_id)
        schema_ctx = json.dumps(existing_schema, indent=2)
    except Exception:
        schema_ctx = "{}"
 
    system = textwrap.dedent("""
        You are a technical team lead decomposing an architecture specification
        into concrete, independently-workable coding tasks for junior engineers.
 
        Output a JSON array of task objects. Each object must have EXACTLY
        these fields (no extras):
        {
          "task_id":        "<short slug, e.g. 'auth-model' or 'db-schema'>",
          "desc":           "<precise description: what to implement, which file(s), what interface>",
          "dependency_list": ["<task_id>", ...]   // IDs of tasks that MUST complete first
        }
 
        Rules:
        - Tasks must be small enough for one engineer in one session (~1-3 files).
        - dependency_list must only reference task_ids defined in the same array.
        - If a task has no dependencies, use an empty list [].
        - Order does not matter — the scheduler uses dependency_list to sequence.
        - Do NOT include project_id, status, worker_id — those are added by the system.
        - Output ONLY the raw JSON array. No markdown. No explanation.
    """).strip()
 
    user_message = textwrap.dedent(f"""
        Architecture specification:
        {arch_spec}
 
        Existing project files (do not recreate these):
        {schema_ctx}
 
        Decompose this into tasks now.
    """).strip()
 
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
 
    raw = response.content[0].text if response.content else "[]"
 
    # Strip accidental markdown fences
    if raw.strip().startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.strip().startswith("```")
        )
 
    try:
        task_defs: List[Dict] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("decompose_tasks: failed to parse LLM JSON: %s", exc)
        task_defs = []
 
    # Build TaskRecord list, validate dependency references
    all_ids = {t["task_id"] for t in task_defs}
    registry: Dict[str, TaskRecord] = {}
 
    for td in task_defs:
        tid = td["task_id"]
        # Drop dependency refs that point to unknown tasks
        deps = [d for d in td.get("dependency_list", []) if d in all_ids and d != tid]
        rec: TaskRecord = {
            "task_id": tid,
            "project_id": project_id,
            "desc": td["desc"],
            "status": "Open",
            "worker_id": "",
            "dependency_list": deps,
            "retry_count": 0,
            "report": None,
            "review_notes": "",
        }
        registry[tid] = rec
 
    _save_registry(project_id, registry)
 
    logger.info(
        "[%s] Decomposed into %d tasks: %s",
        project_id, len(registry), list(registry.keys()),
    )
 
    return {
        "all_task_ids": list(registry.keys()),
        "current_batch": [],
        "iteration": 0,
        "worker_reports": {},
        "all_touched_files": [],
        "batch_reviews": [],
        "latest_review": None,
        "retry_queue": [],
        "project_report": None,
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": raw},
        ],
    }

def schedule_iteration(state: TeamLeadState) -> Dict[str, Any]:
    project_id = state["project_id"]
    registry = _load_registry(project_id)
 
    # Promote retry_queue tasks back to Open with updated descriptions
    for tid in state.get("retry_queue", []):
        if tid in registry:
            rec = registry[tid]
            rec["status"] = "Open"
            # Prepend review notes to desc so the worker sees the feedback
            if rec["review_notes"]:
                rec["desc"] = (
                    f"[RETRY — address these issues first]\n{rec['review_notes']}\n\n"
                    f"Original task:\n{rec['desc']}"
                )
            registry[tid] = rec
 
    _save_registry(project_id, registry)
 
    batch = _ready_tasks(registry)
    batch_ids = [t["task_id"] for t in batch]
 
    iteration = state.get("iteration", 0) + 1
    logger.info(
        "[%s] Iteration %d — ready batch: %s",
        project_id, iteration, batch_ids,
    )
 
    return {
        "current_batch": batch_ids,
        "iteration": iteration,
        "retry_queue": [],  # consumed
    }

def dispatch_workers(state: TeamLeadState) -> Dict[str, Any]:
    project_id = state["project_id"]
    batch_ids = state["current_batch"]
 
    if not batch_ids:
        logger.warning("[%s] dispatch_workers called with empty batch", project_id)
        return {"worker_reports": state.get("worker_reports", {})}
 
    registry = _load_registry(project_id)
 
    # Mark all batch tasks as InProgress
    for tid in batch_ids:
        _update_task_status(project_id, tid, "InProgress")
 
    # Build Task objects for the workers
    def _make_task(tid: str) -> Task:
        rec = registry[tid]
        return Task(
            task_id=tid,
            project_id=project_id,
            desc=rec["desc"],
            status="InProgress",
            worker_id=f"worker-{tid}",
            dependency_list=rec["dependency_list"],
        )
 
    def _run_one(tid: str) -> Tuple[str, WorkerReport]:
        worker_id = f"worker-{tid}"
        task = _make_task(tid)
        logger.info("[%s] Dispatching worker for task '%s'", project_id, tid)
        try:
            report = run_worker(task, worker_id)
        except Exception as exc:
            logger.exception("[%s] Worker for '%s' raised: %s", project_id, tid, exc)
            report = WorkerReport(
                task_id=tid,
                worker_id=worker_id,
                status="failed",
                summary=f"Worker crashed: {exc}",
                touched_files=[],
                test_file_path=None,
                review_score=0,
                review_feedback="",
                blockers=[str(exc)],
                suggestions=[],
                execution_notes="",
                completed_at=datetime.utcnow().isoformat() + "Z",
            )
        return tid, report
 
    # Run workers in parallel, capped at MAX_WORKER_PARALLELISM
    new_reports: Dict[str, Dict] = {}
    new_touched: List[str] = []
 
    with ThreadPoolExecutor(max_workers=min(len(batch_ids), MAX_WORKER_PARALLELISM)) as pool:
        futures = {pool.submit(_run_one, tid): tid for tid in batch_ids}
        for future in as_completed(futures):
            tid, report = future.result()
            new_reports[tid] = dict(report)
            new_touched.extend(report.get("touched_files", []))
 
            # Update registry based on report status
            report_status = report["status"]
            if report_status == "completed":
                task_status = "Closed"
            elif report_status == "completed_with_issues":
                task_status = "PendingReview"
            elif report_status == "blocked":
                task_status = "ReworkRequired"
            else:  # failed
                task_status = "ReworkRequired"
 
            _update_task_status(
                project_id, tid, task_status,
                report=dict(report),
            )
            logger.info(
                "[%s] Task '%s' → %s (worker status: %s)",
                project_id, tid, task_status, report_status,
            )
 
    # Merge with existing reports (earlier iterations)
    merged_reports = {**state.get("worker_reports", {}), **new_reports}
 
    # Accumulate touched files without duplicates
    existing_touched = state.get("all_touched_files", [])
    all_touched = list(existing_touched)
    for fp in new_touched:
        if fp not in all_touched:
            all_touched.append(fp)
 
    return {
        "worker_reports": merged_reports,
        "all_touched_files": all_touched,
    }

def collect_reports(state: TeamLeadState) -> Dict[str, Any]:
    project_id = state["project_id"]
    registry = _load_registry(project_id)
 
    retry_queue: List[str] = []
    for tid, rec in registry.items():
        if rec["status"] == "ReworkRequired":
            if rec["retry_count"] < MAX_RETRY_CYCLES:
                retry_queue.append(tid)
                logger.info(
                    "[%s] Task '%s' queued for retry (attempt %d/%d)",
                    project_id, tid, rec["retry_count"] + 1, MAX_RETRY_CYCLES,
                )
            else:
                logger.warning(
                    "[%s] Task '%s' exhausted retries — marking as permanently failed",
                    project_id, tid,
                )
 
    counts = _counts(registry)
    logger.info("[%s] collect_reports — status counts: %s", project_id, counts)
 
    return {"retry_queue": retry_queue}
 
 
def review_batch(state: TeamLeadState) -> Dict[str, Any]:
    project_id = state["project_id"]
    client = Anthropic()
 
    # Read all touched files across the current batch
    batch_ids = state.get("current_batch", [])
    batch_reports = {
        tid: state["worker_reports"][tid]
        for tid in batch_ids
        if tid in state.get("worker_reports", {})
    }
 
    files_for_review: Dict[str, str] = {}
    for report in batch_reports.values():
        for fp in report.get("touched_files", []):
            if fp in files_for_review:
                continue
            try:
                raw = read_file(project_id, [fp])
                files_for_review[fp] = "\n".join(raw[fp].values())
            except Exception as exc:
                files_for_review[fp] = f"(Could not read: {exc})"
 
    files_block = "\n\n".join(
        f"=== {fp} ===\n{content}"
        for fp, content in files_for_review.items()
    ) or "(no files in this batch)"
 
    # Summarise worker reports for context
    reports_summary = "\n".join(
        f"- {tid}: {r['status']} (score {r['review_score']}/10) — {r['summary']}"
        for tid, r in batch_reports.items()
    )
 
    system = textwrap.dedent("""
        You are a senior engineer doing an integration review of a batch of tasks
        that were implemented in parallel.
 
        Focus ONLY on cross-cutting concerns — not per-task correctness (workers
        already reviewed that). Look for:
        - Interface mismatches (function signatures that callers don't match)
        - Missing or incorrect imports between modules
        - Duplicate implementations of the same logic
        - Inconsistent naming conventions across files
        - Missing glue code (e.g. a module is imported but never registered)
 
        Respond in this EXACT format:
 
        SCORE: <0-10>
        VERDICT: <pass|fail>
        ISSUES:
        - <issue>
        (or "- None")
        SUGGESTIONS:
        - <suggestion>
        (or "- None")
        SUMMARY:
        <2-3 sentences>
 
        Scoring:
          9-10  no integration issues
          7-8   minor issues, acceptable  (pass)
          5-6   integration gaps that will cause runtime errors  (fail)
          0-4   fundamental integration problems  (fail)
    """).strip()
 
    user_message = textwrap.dedent(f"""
        Batch iteration: {state.get('iteration', '?')}
 
        Worker report summaries:
        {reports_summary}
 
        Files implemented:
        {files_block}
    """).strip()
 
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
 
    review_text = response.content[0].text if response.content else ""
 
    # Parse structured fields
    score = _parse_score(review_text)
    verdict: Literal["pass", "fail"] = "pass" if score >= INTEGRATION_PASS_THRESHOLD else "fail"
    issues = _parse_list_section(review_text, "ISSUES")
    suggestions = _parse_list_section(review_text, "SUGGESTIONS")
    summary = _parse_summary(review_text)
 
    batch_review: BatchReview = {
        "score": score,
        "verdict": verdict,
        "issues": issues,
        "suggestions": suggestions,
        "summary": summary,
    }
 
    logger.info(
        "[%s] Batch review — score %d/10 verdict=%s",
        project_id, score, verdict,
    )
 
    existing_reviews = list(state.get("batch_reviews", []))
    existing_reviews.append(batch_review)
 
    return {
        "latest_review": batch_review,
        "batch_reviews": existing_reviews,
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": review_text},
        ],
    }
 
 
def handle_batch_review(state: TeamLeadState) -> Dict[str, Any]:
    project_id = state["project_id"]
    review = state.get("latest_review")
    registry = _load_registry(project_id)
 
    if review and review["verdict"] == "fail" and review["issues"]:
        # Write integration issues back onto ReworkRequired tasks as review_notes
        issues_text = "\n".join(f"- {i}" for i in review["issues"])
        for tid in state.get("retry_queue", []):
            if tid in registry:
                _update_task_status(
                    project_id, tid,
                    "ReworkRequired",
                    review_notes=f"Integration review issues:\n{issues_text}",
                    retry_count=registry[tid]["retry_count"] + 1,
                )
 
    counts = _counts(_load_registry(project_id))
    logger.info("[%s] handle_batch_review — counts: %s", project_id, counts)
    return {}
 
 
def finalize(state: TeamLeadState) -> Dict[str, Any]:
    project_id = state["project_id"]
    registry = _load_registry(project_id)
    client = Anthropic()
 
    counts = _counts(registry)
    all_touched = state.get("all_touched_files", [])
    batch_reviews = state.get("batch_reviews", [])
    scores = [r["score"] for r in batch_reviews]
 
    # Ask LLM for an executive summary
    worker_summaries = "\n".join(
        f"- [{rec['task_id']}] {rec['status']}: "
        + (rec["report"]["summary"] if rec.get("report") else "no report")
        for rec in registry.values()
    )
 
    system = "You are a technical lead writing a concise project completion summary for senior management. 3-5 sentences max. Plain text, no markdown."
    user_message = f"Project: {project_id}\nTask outcomes:\n{worker_summaries}\nBatch review scores: {scores}"
 
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    summary = response.content[0].text if response.content else "Project completed."
 
    total = len(registry)
    completed = counts.get("Closed", 0)
    failed = sum(
        1 for rec in registry.values()
        if rec["status"] == "ReworkRequired" and rec["retry_count"] >= MAX_RETRY_CYCLES
    )
    blocked = counts.get("ReworkRequired", 0) - failed
 
    if completed == total:
        final_status: Literal["success", "partial", "failed"] = "success"
    elif completed > 0:
        final_status = "partial"
    else:
        final_status = "failed"
 
    report: ProjectReport = {
        "project_id": project_id,
        "total_tasks": total,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "blocked_tasks": blocked,
        "all_touched_files": all_touched,
        "batch_review_scores": scores,
        "final_status": final_status,
        "summary": summary,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
 
    # Persist the project report to disk
    report_path = Path(PROJECT_REPORT_FILE.format(project_id=project_id))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
 
    logger.info(
        "[%s] Finalized — status=%s completed=%d/%d",
        project_id, final_status, completed, total,
    )
 
    return {"project_report": report}
 
# conditional rendering
def route_after_review(
    state: TeamLeadState,
) -> Literal["schedule_iteration", "finalize"]:
    """
    Conditional edge after handle_batch_review.
 
    Routes to:
      schedule_iteration  — if there are still Open tasks OR retryable tasks
      finalize            — if all tasks are terminal (Closed / exhausted)
    """
    project_id = state["project_id"]
    registry = _load_registry(project_id)
    counts = _counts(registry)
 
    open_tasks = counts.get("Open", 0)
    in_progress = counts.get("InProgress", 0)
    pending_review = counts.get("PendingReview", 0)
    rework_required = counts.get("ReworkRequired", 0)
 
    # Any retryable tasks?
    retryable = [
        rec for rec in registry.values()
        if rec["status"] == "ReworkRequired" and rec["retry_count"] < MAX_RETRY_CYCLES
    ]
 
    if open_tasks > 0 or in_progress > 0 or pending_review > 0 or retryable:
        logger.info(
            "[%s] route_after_review → schedule_iteration "
            "(open=%d in_progress=%d pending=%d retryable=%d)",
            project_id, open_tasks, in_progress, pending_review, len(retryable),
        )
        return "schedule_iteration"
 
    logger.info(
        "[%s] route_after_review → finalize "
        "(closed=%d rework_exhausted=%d)",
        project_id,
        counts.get("Closed", 0),
        rework_required,
    )
    return "finalize"
 
 
def route_after_schedule(
    state: TeamLeadState,
) -> Literal["dispatch_workers", "finalize"]:
    """
    Conditional edge after schedule_iteration.
 
    If schedule_iteration found no ready tasks (could happen if all remaining
    tasks are blocked by failed dependencies), skip to finalize instead of
    hanging in an empty dispatch loop.
    """
    if state.get("current_batch"):
        return "dispatch_workers"
    logger.warning(
        "[%s] schedule_iteration produced empty batch — going straight to finalize",
        state["project_id"],
    )
    return "finalize"
 
 
# build graph

def build_teamlead_graph() -> StateGraph:
    graph = StateGraph(TeamLeadState)
 
    # Register nodes
    graph.add_node("decompose_tasks", decompose_tasks)
    graph.add_node("schedule_iteration", schedule_iteration)
    graph.add_node("dispatch_workers", dispatch_workers)
    graph.add_node("collect_reports", collect_reports)
    graph.add_node("review_batch", review_batch)
    graph.add_node("handle_batch_review", handle_batch_review)
    graph.add_node("finalize", finalize)
 
    # Linear entry
    graph.add_edge(START, "decompose_tasks")
 
    # decompose → schedule (conditional: skip if no tasks were produced)
    graph.add_conditional_edges(
        "decompose_tasks",
        lambda s: "schedule_iteration" if s.get("all_task_ids") else "finalize",
        {"schedule_iteration": "schedule_iteration", "finalize": "finalize"},
    )
 
    # schedule → dispatch (conditional: skip if nothing ready)
    graph.add_conditional_edges(
        "schedule_iteration",
        route_after_schedule,
        {"dispatch_workers": "dispatch_workers", "finalize": "finalize"},
    )
 
    # Linear execution pipeline
    graph.add_edge("dispatch_workers", "collect_reports")
    graph.add_edge("collect_reports", "review_batch")
    graph.add_edge("review_batch", "handle_batch_review")
 
    # Loop or terminate
    graph.add_conditional_edges(
        "handle_batch_review",
        route_after_review,
        {"schedule_iteration": "schedule_iteration", "finalize": "finalize"},
    )
 
    graph.add_edge("finalize", END)
 
    return graph
 
 
def create_teamlead_agent():
    """Compile and return the runnable TeamLead agent."""
    return build_teamlead_graph().compile()
 
 
def run_teamlead(
    project_id: str,
    architecture_spec: str,
) -> ProjectReport:
    """
    Orchestrate an entire project from architecture spec to completion.
 
    Args:
        project_id:         Unique identifier for the project (used for all
                            file paths under projects/{project_id}/).
        architecture_spec:  Free-text architecture document from ArchitectAgent.
 
    Returns:
        ProjectReport with final status, touched files, and review scores.
    """
    agent = create_teamlead_agent()
 
    initial_state: TeamLeadState = {
        "project_id": project_id,
        "architecture_spec": architecture_spec,
        "all_task_ids": [],
        "current_batch": [],
        "iteration": 0,
        "worker_reports": {},
        "all_touched_files": [],
        "batch_reviews": [],
        "latest_review": None,
        "retry_queue": [],
        "messages": [],
        "project_report": None,
    }
 
    final_state = agent.invoke(initial_state)
    report = final_state.get("project_report")
 
    if report is None:
        report = ProjectReport(
            project_id=project_id,
            total_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            blocked_tasks=0,
            all_touched_files=[],
            batch_review_scores=[],
            final_status="failed",
            summary="TeamLeadAgent completed without producing a report.",
            completed_at=datetime.utcnow().isoformat() + "Z",
        )
 
    return report