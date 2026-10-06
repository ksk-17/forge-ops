"""
forge_memory.py — Telemetry store + mem0-powered memory layer for forge-ops.

Two systems in one module, both sharing the same project directory layout:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. TELEMETRY STORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Append-only JSONL files that record everything that happens during a run.
Zero external dependencies — pure stdlib JSON + pathlib.

Directory layout:
  projects/_telemetry/
    runs.jsonl              ← one record per complete project run
    agent_events.jsonl      ← fine-grained per-node events
    prompt_versions.jsonl   ← versioned snapshots of every system prompt
    worker_scores.jsonl     ← per-task review scores over time

What gets recorded:
  • Every project run: inputs, outputs, timing, final status
  • Every agent node execution: node name, agent, duration, token counts
  • Every LLM call: model, prompt hash, response score (where applicable)
  • Per-task worker scores: used by Ralph to identify weak prompt versions

Public API:
  TelemetryStore.log_run(run: RunRecord)
  TelemetryStore.log_event(event: AgentEvent)
  TelemetryStore.log_prompt_version(pv: PromptVersion)
  TelemetryStore.log_worker_score(ws: WorkerScore)
  TelemetryStore.query_runs(project_id, limit) → List[RunRecord]
  TelemetryStore.query_worker_scores(task_pattern, limit) → List[WorkerScore]
  TelemetryStore.prompt_score_history(agent, node) → List[dict]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2. MEMORY LAYER  (mem0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mem0 gives every agent a long-term semantic memory that persists across runs.
Agents can store facts, recall relevant past context, and update their
understanding without re-reading all historical telemetry every call.

Three memory scopes:
  user_id   = "global"          → system-wide facts (tech stack preferences,
                                   project conventions, recurring patterns)
  user_id   = f"project:{pid}"  → project-scoped facts (decisions made, files
                                   created, constraints discovered)
  user_id   = f"agent:{name}"   → per-agent memories (what worked/failed,
                                   patterns per agent role)

Memory is injected into agent system prompts as a "Relevant context from
memory" section so agents automatically benefit without code changes to
the core graph nodes.

Public API:
  AgentMemory.add(text, scope, metadata)    → store a new memory
  AgentMemory.search(query, scope, limit)   → semantic search
  AgentMemory.get_context_block(query, scopes) → formatted str for system prompts
  AgentMemory.record_run_outcome(project_id, report)
  AgentMemory.record_task_outcome(task_id, project_id, report)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3. INTEGRATION HELPERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  record_architect_run(spec, answers)
  record_teamlead_run(project_id, report, task_records)
  record_worker_run(task_id, project_id, report)
  get_worker_context(task_desc, project_id) → str  (inject into plan_task)
  get_architect_context(raw_input) → str            (inject into analyse_input)
  get_teamlead_context(project_id) → str            (inject into decompose_tasks)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TELEMETRY_DIR = Path("projects/_telemetry")
_RUNS_FILE = _TELEMETRY_DIR / "runs.jsonl"
_EVENTS_FILE = _TELEMETRY_DIR / "agent_events.jsonl"
_PROMPTS_FILE = _TELEMETRY_DIR / "prompt_versions.jsonl"
_SCORES_FILE = _TELEMETRY_DIR / "worker_scores.jsonl"


def _ensure_telemetry_dir() -> None:
    _TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Telemetry data models
# ---------------------------------------------------------------------------

class RunRecord(TypedDict):
    """One complete project run from raw_input → ProjectReport."""
    run_id: str                     # uuid or timestamp-based
    project_id: str
    project_name: str
    raw_input: str                  # user's original idea
    architecture_spec: str          # produced by ArchitectAgent
    final_status: str               # success / partial / failed
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    batch_review_scores: List[int]
    avg_review_score: float
    total_duration_seconds: float
    started_at: str                 # ISO-8601
    completed_at: str


class AgentEvent(TypedDict):
    """One node execution within an agent graph."""
    event_id: str
    run_id: str
    project_id: str
    agent: str                      # "WorkerAgent" / "TeamLeadAgent" / "ArchitectAgent"
    node: str                       # "plan_task" / "execute_task" / etc.
    task_id: Optional[str]
    duration_seconds: float
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]        # node-specific data (scores, file counts, etc.)
    timestamp: str


class PromptVersion(TypedDict):
    """Snapshot of a system prompt at a point in time."""
    prompt_id: str                  # sha256 of content
    agent: str
    node: str
    version: int
    content: str
    avg_score: float                # updated as runs complete
    run_count: int
    recorded_at: str


class WorkerScore(TypedDict):
    """Per-task quality score for Ralph loop training data."""
    score_id: str
    run_id: str
    project_id: str
    task_id: str
    task_desc: str                  # first 200 chars
    review_score: int               # 0-10
    rework_count: int
    status: str
    prompt_version_id: str          # which plan_task prompt was used
    completed_at: str


# ---------------------------------------------------------------------------
# TelemetryStore
# ---------------------------------------------------------------------------

class TelemetryStore:
    """
    Append-only JSONL telemetry store. Thread-safe for single-process use.
    Each method opens, appends, and closes immediately — no state held.
    """

    @staticmethod
    def _append(path: Path, record: Dict) -> None:
        _ensure_telemetry_dir()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_all(path: Path) -> List[Dict]:
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    # ── write ────────────────────────────────────────────────────────────────

    @staticmethod
    def log_run(run: RunRecord) -> None:
        TelemetryStore._append(_RUNS_FILE, dict(run))
        logger.debug("Telemetry: logged run %s", run["run_id"])

    @staticmethod
    def log_event(event: AgentEvent) -> None:
        TelemetryStore._append(_EVENTS_FILE, dict(event))

    @staticmethod
    def log_prompt_version(pv: PromptVersion) -> None:
        TelemetryStore._append(_PROMPTS_FILE, dict(pv))
        logger.debug("Telemetry: logged prompt version %s", pv["prompt_id"])

    @staticmethod
    def log_worker_score(ws: WorkerScore) -> None:
        TelemetryStore._append(_SCORES_FILE, dict(ws))

    # ── read ─────────────────────────────────────────────────────────────────

    @staticmethod
    def query_runs(
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[RunRecord]:
        records = TelemetryStore._read_all(_RUNS_FILE)
        if project_id:
            records = [r for r in records if r.get("project_id") == project_id]
        if status:
            records = [r for r in records if r.get("final_status") == status]
        return records[-limit:]

    @staticmethod
    def query_events(
        run_id: Optional[str] = None,
        agent: Optional[str] = None,
        node: Optional[str] = None,
        limit: int = 200,
    ) -> List[AgentEvent]:
        records = TelemetryStore._read_all(_EVENTS_FILE)
        if run_id:
            records = [r for r in records if r.get("run_id") == run_id]
        if agent:
            records = [r for r in records if r.get("agent") == agent]
        if node:
            records = [r for r in records if r.get("node") == node]
        return records[-limit:]

    @staticmethod
    def query_worker_scores(
        task_pattern: Optional[str] = None,
        min_score: Optional[int] = None,
        max_score: Optional[int] = None,
        limit: int = 100,
    ) -> List[WorkerScore]:
        records = TelemetryStore._read_all(_SCORES_FILE)
        if task_pattern:
            records = [r for r in records if task_pattern.lower() in r.get("task_desc", "").lower()]
        if min_score is not None:
            records = [r for r in records if r.get("review_score", 0) >= min_score]
        if max_score is not None:
            records = [r for r in records if r.get("review_score", 10) <= max_score]
        return records[-limit:]

    @staticmethod
    def prompt_score_history(agent: str, node: str) -> List[Dict]:
        """Return avg scores per prompt version for a given agent+node."""
        versions = TelemetryStore._read_all(_PROMPTS_FILE)
        versions = [v for v in versions if v.get("agent") == agent and v.get("node") == node]

        scores = TelemetryStore._read_all(_SCORES_FILE)
        # Group scores by prompt_version_id
        score_map: Dict[str, List[int]] = {}
        for ws in scores:
            pid = ws.get("prompt_version_id", "unknown")
            score_map.setdefault(pid, []).append(ws.get("review_score", 0))

        result = []
        for v in versions:
            vid = v["prompt_id"]
            sc = score_map.get(vid, [])
            result.append({
                "prompt_id": vid,
                "version": v.get("version", 0),
                "recorded_at": v.get("recorded_at", ""),
                "run_count": len(sc),
                "avg_score": round(sum(sc) / len(sc), 2) if sc else None,
            })
        return result

    @staticmethod
    def summary_stats() -> Dict[str, Any]:
        """High-level dashboard stats."""
        runs = TelemetryStore._read_all(_RUNS_FILE)
        scores = TelemetryStore._read_all(_SCORES_FILE)

        total_runs = len(runs)
        if runs:
            status_counts = {}
            for r in runs:
                s = r.get("final_status", "unknown")
                status_counts[s] = status_counts.get(s, 0) + 1
            avg_batch_score = (
                sum(r.get("avg_review_score", 0) for r in runs) / total_runs
            )
        else:
            status_counts = {}
            avg_batch_score = 0.0

        score_vals = [s.get("review_score", 0) for s in scores]
        avg_worker_score = round(sum(score_vals) / len(score_vals), 2) if score_vals else 0.0

        return {
            "total_runs": total_runs,
            "status_breakdown": status_counts,
            "avg_batch_review_score": round(avg_batch_score, 2),
            "total_task_executions": len(scores),
            "avg_worker_review_score": avg_worker_score,
            "low_score_tasks": len([s for s in scores if s.get("review_score", 10) < 6]),
        }


# ---------------------------------------------------------------------------
# Prompt version tracking helper
# ---------------------------------------------------------------------------

def _prompt_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def record_prompt_version(agent: str, node: str, content: str) -> str:
    """
    Record a system prompt if it hasn't been seen before (content-addressed).
    Returns the prompt_id (hash) for use in WorkerScore records.
    """
    pid = _prompt_hash(content)

    # Check if this exact hash already exists
    existing = TelemetryStore._read_all(_PROMPTS_FILE)
    if any(v.get("prompt_id") == pid for v in existing):
        return pid   # already recorded

    # Find highest version for this agent+node
    same = [v for v in existing if v.get("agent") == agent and v.get("node") == node]
    version = max((v.get("version", 0) for v in same), default=0) + 1

    pv: PromptVersion = {
        "prompt_id": pid,
        "agent": agent,
        "node": node,
        "version": version,
        "content": content,
        "avg_score": 0.0,
        "run_count": 0,
        "recorded_at": datetime.utcnow().isoformat() + "Z",
    }
    TelemetryStore.log_prompt_version(pv)
    return pid


# ---------------------------------------------------------------------------
# mem0 Memory Layer
# ---------------------------------------------------------------------------

class AgentMemory:
    """
    Semantic memory for all agents, backed by mem0.

    mem0 handles:
      - Vector embedding of memories
      - Semantic similarity search
      - Automatic deduplication and update
      - Persistence across runs

    Scopes (mem0 user_id):
      "global"            — system-wide patterns and preferences
      "project:{pid}"     — per-project decisions and constraints
      "agent:architect"   — ArchitectAgent patterns
      "agent:teamlead"    — TeamLeadAgent decomposition patterns
      "agent:worker"      — WorkerAgent implementation patterns

    Falls back gracefully if mem0 is not installed.
    """

    _client = None
    _available = False

    @classmethod
    def _get_client(cls):
        if cls._client is not None:
            return cls._client
        try:
            from mem0 import Memory
            cls._client = Memory()
            cls._available = True
            logger.info("mem0 memory layer initialised")
        except ImportError:
            logger.warning(
                "mem0 not installed — memory layer disabled. "
                "Install with: pip install mem0ai"
            )
            cls._available = False
        except Exception as exc:
            logger.warning("mem0 initialisation failed: %s — memory disabled", exc)
            cls._available = False
        return cls._client

    @classmethod
    def is_available(cls) -> bool:
        cls._get_client()
        return cls._available

    # ── write ────────────────────────────────────────────────────────────────

    @classmethod
    def add(
        cls,
        text: str,
        scope: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Store a memory in the given scope.

        Args:
            text:     The fact or observation to remember.
            scope:    mem0 user_id — use scope constants above.
            metadata: Optional key-value tags (agent, project_id, task_id, etc.)

        Returns:
            Memory ID string, or None if mem0 is unavailable.
        """
        client = cls._get_client()
        if not cls._available or client is None:
            return None
        try:
            result = client.add(
                messages=[{"role": "user", "content": text}],
                user_id=scope,
                metadata=metadata or {},
            )
            # mem0 returns {"results": [{"id": ..., "memory": ..., "event": ...}]}
            results = result.get("results", [])
            mem_id = results[0].get("id") if results else None
            logger.debug("Memory added to scope '%s': %s", scope, text[:80])
            return mem_id
        except Exception as exc:
            logger.warning("Memory add failed: %s", exc)
            return None

    @classmethod
    def search(
        cls,
        query: str,
        scope: str,
        limit: int = 5,
    ) -> List[Dict]:
        """
        Semantic search for relevant memories in a scope.

        Returns list of dicts with keys: id, memory, score, metadata.
        Returns [] if mem0 is unavailable.
        """
        client = cls._get_client()
        if not cls._available or client is None:
            return []
        try:
            results = client.search(query, user_id=scope, limit=limit)
            return results.get("results", []) if isinstance(results, dict) else results
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)
            return []

    @classmethod
    def get_context_block(
        cls,
        query: str,
        scopes: List[str],
        limit_per_scope: int = 3,
    ) -> str:
        """
        Search multiple scopes and format results as a context block
        ready to inject into an agent system prompt.

        Returns "" if no memories found or mem0 unavailable.

        Example output:
            --- Relevant context from memory ---
            • Previously used PostgreSQL for this project's storage layer.
            • REST endpoints should follow /api/v1/{resource} convention.
            • Worker implementing auth modules tends to need JWT library.
            ------------------------------------
        """
        if not cls.is_available():
            return ""

        all_memories: List[str] = []
        seen: set = set()

        for scope in scopes:
            results = cls.search(query, scope, limit=limit_per_scope)
            for r in results:
                text = r.get("memory", "")
                if text and text not in seen:
                    seen.add(text)
                    all_memories.append(text)

        if not all_memories:
            return ""

        lines = ["--- Relevant context from memory ---"]
        for mem in all_memories:
            lines.append(f"• {mem}")
        lines.append("------------------------------------")
        return "\n".join(lines)

    # ── high-level outcome recording ─────────────────────────────────────────

    @classmethod
    def record_run_outcome(cls, project_id: str, report: Dict) -> None:
        """
        After a full project run, store high-level outcomes as memories
        so future runs can learn from them.
        """
        if not cls.is_available():
            return

        status = report.get("final_status", "unknown")
        scores = report.get("batch_review_scores", [])
        avg = round(sum(scores) / len(scores), 1) if scores else 0

        # Global memory: patterns that apply to all future projects
        cls.add(
            f"Project '{project_id}' completed with status '{status}'. "
            f"Average integration review score: {avg}/10. "
            f"Total tasks: {report.get('total_tasks', 0)}, "
            f"completed: {report.get('completed_tasks', 0)}.",
            scope="global",
            metadata={"project_id": project_id, "type": "run_outcome"},
        )

        # Project-scoped memory: what files were produced
        touched = report.get("all_touched_files", [])
        if touched:
            cls.add(
                f"Files produced in project '{project_id}': {', '.join(touched[:10])}.",
                scope=f"project:{project_id}",
                metadata={"project_id": project_id, "type": "files_produced"},
            )

        # Agent memory: teamlead patterns
        if status == "success":
            cls.add(
                f"Successful project run with {report.get('total_tasks', 0)} tasks "
                f"and avg review score {avg}/10.",
                scope="agent:teamlead",
                metadata={"type": "success_pattern"},
            )
        elif status == "failed":
            cls.add(
                f"Failed project run for '{project_id}'. "
                f"Failed tasks: {report.get('failed_tasks', 0)}. "
                "Consider smaller task decomposition.",
                scope="agent:teamlead",
                metadata={"type": "failure_pattern"},
            )

    @classmethod
    def record_task_outcome(
        cls,
        task_id: str,
        project_id: str,
        report: Dict,
    ) -> None:
        """
        After a WorkerAgent completes a task, store what worked and what didn't.
        """
        if not cls.is_available():
            return

        status = report.get("status", "unknown")
        score = report.get("review_score", 0)
        desc = report.get("summary", "")
        touched = report.get("touched_files", [])
        blockers = report.get("blockers", [])

        # Project memory: what was implemented
        if touched:
            cls.add(
                f"Task '{task_id}' implemented files: {', '.join(touched)}. "
                f"Review score: {score}/10. Status: {status}.",
                scope=f"project:{project_id}",
                metadata={"task_id": task_id, "type": "task_outcome"},
            )

        # Worker agent memory: patterns from high/low scoring tasks
        if score >= 9:
            cls.add(
                f"High-quality implementation (score {score}/10): {desc[:150]}",
                scope="agent:worker",
                metadata={"type": "good_pattern", "score": score},
            )
        elif score <= 5 or blockers:
            blockers_text = "; ".join(blockers[:3]) if blockers else "low review score"
            cls.add(
                f"Implementation issue (score {score}/10): {blockers_text}. "
                f"Task: {desc[:100]}",
                scope="agent:worker",
                metadata={"type": "failure_pattern", "score": score},
            )

    @classmethod
    def record_architecture(cls, project_id: str, spec: Dict, answers: List[Dict]) -> None:
        """
        After ArchitectAgent produces a spec, store key decisions as memories.
        """
        if not cls.is_available():
            return

        project_name = spec.get("project_name", project_id)

        # Store the Q&A decisions for this project
        if answers:
            decisions = "\n".join(
                f"  - {a['question']}: {a['answer']}"
                for a in answers[:10]
            )
            cls.add(
                f"Architecture decisions for '{project_name}':\n{decisions}",
                scope=f"project:{project_id}",
                metadata={"type": "architecture_decisions", "project_id": project_id},
            )

        # Global: remember what kinds of projects have been built
        cls.add(
            f"Built architecture spec for project '{project_name}' ({project_id}).",
            scope="global",
            metadata={"project_id": project_id, "type": "project_created"},
        )

        # Architect agent: remember spec patterns
        spec_text = spec.get("spec_text", "")
        if spec_text:
            cls.add(
                f"Architecture pattern for '{project_name}': "
                f"{spec_text[:300]}",
                scope="agent:architect",
                metadata={"project_id": project_id, "type": "spec_pattern"},
            )


# ---------------------------------------------------------------------------
# Integration helpers — called from agent nodes
# ---------------------------------------------------------------------------

def get_worker_context(task_desc: str, project_id: str) -> str:
    """
    Retrieve relevant memory for a WorkerAgent about to plan a task.
    Call this inside plan_task and prepend to the system prompt.

    Searches:
      - agent:worker  (past implementation patterns)
      - project:{pid} (what's already been built in this project)
    """
    return AgentMemory.get_context_block(
        query=task_desc,
        scopes=["agent:worker", f"project:{project_id}"],
        limit_per_scope=3,
    )


def get_architect_context(raw_input: str) -> str:
    """
    Retrieve relevant memory for the ArchitectAgent starting requirements gathering.
    Call this inside analyse_input and prepend to the system prompt.

    Searches:
      - agent:architect (past spec patterns)
      - global          (system-wide conventions and preferences)
    """
    return AgentMemory.get_context_block(
        query=raw_input,
        scopes=["agent:architect", "global"],
        limit_per_scope=3,
    )


def get_teamlead_context(project_id: str, architecture_spec: str = "") -> str:
    """
    Retrieve relevant memory for the TeamLeadAgent about to decompose tasks.
    Call this inside decompose_tasks and prepend to the system prompt.

    Searches:
      - agent:teamlead    (past decomposition patterns)
      - project:{pid}     (constraints and decisions already made)
      - global            (system-wide conventions)
    """
    query = architecture_spec[:300] if architecture_spec else f"project {project_id}"
    return AgentMemory.get_context_block(
        query=query,
        scopes=["agent:teamlead", f"project:{project_id}", "global"],
        limit_per_scope=2,
    )


# ---------------------------------------------------------------------------
# High-level recording helpers — called from agent report_to_teamlead /
# finalize / produce_architecture_spec nodes
# ---------------------------------------------------------------------------

def record_worker_run(
    task_id: str,
    project_id: str,
    report: Dict,
    plan_task_prompt: str = "",
) -> None:
    """
    Record a WorkerAgent task completion in both telemetry and memory.
    Call from report_to_teamlead node (or after run_worker returns).
    """
    import uuid

    run_id = f"worker-{task_id}-{int(time.time())}"

    # Telemetry: agent event
    event: AgentEvent = {
        "event_id": str(uuid.uuid4())[:8],
        "run_id": run_id,
        "project_id": project_id,
        "agent": "WorkerAgent",
        "node": "report_to_teamlead",
        "task_id": task_id,
        "duration_seconds": 0.0,  # caller can pass if tracked
        "success": report.get("status") in ("completed", "completed_with_issues"),
        "error": "; ".join(report.get("blockers", [])) or None,
        "metadata": {
            "review_score": report.get("review_score", 0),
            "rework_count": report.get("rework_count", 0),
            "touched_files": report.get("touched_files", []),
            "status": report.get("status"),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    TelemetryStore.log_event(event)

    # Telemetry: worker score (training data for Ralph)
    prompt_id = record_prompt_version("WorkerAgent", "plan_task", plan_task_prompt) if plan_task_prompt else "unknown"
    ws: WorkerScore = {
        "score_id": str(uuid.uuid4())[:8],
        "run_id": run_id,
        "project_id": project_id,
        "task_id": task_id,
        "task_desc": report.get("summary", "")[:200],
        "review_score": report.get("review_score", 0),
        "rework_count": report.get("rework_count", 0),
        "status": report.get("status", "unknown"),
        "prompt_version_id": prompt_id,
        "completed_at": report.get("completed_at", datetime.utcnow().isoformat() + "Z"),
    }
    TelemetryStore.log_worker_score(ws)

    # Memory: store outcome for future agents
    AgentMemory.record_task_outcome(task_id, project_id, report)
    logger.info("Telemetry+memory recorded for task '%s'", task_id)


def record_teamlead_run(
    project_id: str,
    report: Dict,
    architecture_spec: str = "",
    project_name: str = "",
) -> None:
    """
    Record a full TeamLeadAgent project run.
    Call from finalize node or after run_teamlead returns.
    """
    import uuid

    scores = report.get("batch_review_scores", [])
    run_id = f"teamlead-{project_id}-{int(time.time())}"

    run_record: RunRecord = {
        "run_id": run_id,
        "project_id": project_id,
        "project_name": project_name or project_id,
        "raw_input": "",  # caller can populate
        "architecture_spec": architecture_spec[:500],
        "final_status": report.get("final_status", "unknown"),
        "total_tasks": report.get("total_tasks", 0),
        "completed_tasks": report.get("completed_tasks", 0),
        "failed_tasks": report.get("failed_tasks", 0),
        "batch_review_scores": scores,
        "avg_review_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "total_duration_seconds": 0.0,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "completed_at": report.get("completed_at", datetime.utcnow().isoformat() + "Z"),
    }
    TelemetryStore.log_run(run_record)

    event: AgentEvent = {
        "event_id": str(uuid.uuid4())[:8],
        "run_id": run_id,
        "project_id": project_id,
        "agent": "TeamLeadAgent",
        "node": "finalize",
        "task_id": None,
        "duration_seconds": 0.0,
        "success": report.get("final_status") == "success",
        "error": None,
        "metadata": {
            "final_status": report.get("final_status"),
            "total_tasks": report.get("total_tasks", 0),
            "completed_tasks": report.get("completed_tasks", 0),
            "avg_review_score": run_record["avg_review_score"],
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    TelemetryStore.log_event(event)

    # Memory
    AgentMemory.record_run_outcome(project_id, report)
    logger.info(
        "Telemetry+memory recorded for project '%s' — status: %s",
        project_id, report.get("final_status"),
    )


def record_architect_run(
    project_id: str,
    spec: Dict,
    answers: List[Dict],
    raw_input: str = "",
) -> None:
    """
    Record an ArchitectAgent spec generation.
    Call after run_architect_cli returns successfully.
    """
    import uuid

    event: AgentEvent = {
        "event_id": str(uuid.uuid4())[:8],
        "run_id": f"architect-{project_id}-{int(time.time())}",
        "project_id": project_id,
        "agent": "ArchitectAgent",
        "node": "produce_architecture_spec",
        "task_id": None,
        "duration_seconds": 0.0,
        "success": True,
        "error": None,
        "metadata": {
            "project_name": spec.get("project_name", project_id),
            "spec_length": len(spec.get("spec_text", "")),
            "clarification_rounds": len(set(a.get("question_id", "") for a in answers)),
            "answers_collected": len(answers),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    TelemetryStore.log_event(event)
    AgentMemory.record_architecture(project_id, spec, answers)
    logger.info("Telemetry+memory recorded for architect run on '%s'", project_id)