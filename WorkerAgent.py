from __future__ import annotations

import json
import logging
import textwrap
from typing import Annotated, Any, Dict, List, Literal, Optional
from datetime import datetime

from anthropic import Anthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from models import Task, Edit, WorkerReport
from WorkerTools import (
    get_project_schema,
    read_file,
    create_file,
    write_file,
)

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-5"
MAX_REWORK_CYCLES = 2
MAX_TOOL_ITERATIONS = 20
REVIEW_PASS_THRESHOLD = 7

class WorkerState(TypedDict):
    task: Task
    worker_id: str
    plan: str
    touched_files: List[str]
    execution_notes: str
    review_score: int
    review_feedback: str
    rework_count: int
    test_file_path: Optional[str]
    test_code: str
    messages: List[Dict]
    report: Optional["WorkerReport"]

WORKER_TOOL_SPECS = [
    {
        "name": "get_project_schema",
        "description": (
            "Return the full project schema (all files, their descriptions, "
            "and tracked method signatures). Call this first to understand the "
            "existing codebase before writing anything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project identifier"}
            },
            "required": ["project_id"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read one or more project files.  Returns a dict of "
            "{file_path: {line_number: line_content}}.  Always read a file "
            "before writing to it so you know the current line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relative file paths to read",
                },
            },
            "required": ["project_id", "file_paths"],
        },
    },
    {
        "name": "create_file",
        "description": (
            "Create a new empty file in the project and register it in the "
            "project schema.  Returns None on success or an error string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "file_path": {"type": "string", "description": "Relative path for the new file"},
                "file_description": {"type": "string", "description": "What this file contains"},
                "worker_id": {"type": "string"},
            },
            "required": ["project_id", "file_path", "file_description", "worker_id"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Apply a list of edits to an existing project file.  Each edit is "
            "one of: insert (before start_line), update (replace lines), or "
            "delete (remove start_line..end_line inclusive).  Edits are applied "
            "in order.  Always read the file first to get current line numbers.  "
            "Returns None on success or an error string on failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "worker_id": {"type": "string"},
                "file_path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["insert", "update", "delete"],
                            },
                            "start_line": {"type": "integer"},
                            "end_line": {
                                "type": "integer",
                                "description": "Required for delete; optional for update",
                            },
                            "new_value": {
                                "type": "string",
                                "description": "New content (for insert/update)",
                            },
                            "method_description_changes": {
                                "type": "object",
                                "description": (
                                    "Map of {method_name: description} for any "
                                    "methods added/changed in this edit"
                                ),
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["operation", "start_line"],
                    },
                },
            },
            "required": ["project_id", "worker_id", "file_path", "edits"],
        },
    },
]

def _dispatch_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    try:
        if tool_name == "get_project_schema":
            result = get_project_schema(tool_input["project_id"])
            return json.dumps(result, indent=2)

        elif tool_name == "read_file":
            result = read_file(tool_input["project_id"], tool_input["file_paths"])
            # Convert int keys to strings for JSON serialisation
            serialisable = {
                fp: {str(ln): content for ln, content in lines.items()}
                for fp, lines in result.items()
            }
            return json.dumps(serialisable, indent=2)

        elif tool_name == "create_file":
            error = create_file(
                tool_input["project_id"],
                tool_input["file_path"],
                tool_input["file_description"],
                tool_input["worker_id"],
            )
            return json.dumps({"success": error is None, "error": error})

        elif tool_name == "write_file":
            raw_edits = tool_input["edits"]
            edits = []
            for e in raw_edits:
                edits.append(
                    Edit(
                        operation=e["operation"],
                        start_line=e["start_line"],
                        end_line=e.get("end_line"),
                        new_value=e.get("new_value", ""),
                        method_description_changes=e.get("method_description_changes", {}),
                    )
                )
            error = write_file(
                tool_input["project_id"],
                tool_input["worker_id"],
                tool_input["file_path"],
                edits,
            )
            return json.dumps({"success": error is None, "error": error})

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as exc:
        logger.exception("Tool '%s' raised an exception", tool_name)
        return json.dumps({"error": str(exc)})
    
def _run_tool_loop(
    client: Anthropic,
    system_prompt: str,
    initial_messages: List[Dict],
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> tuple[List[Dict], List[str], str]:
    messages = list(initial_messages)
    touched_files: List[str] = []
    notes_parts: List[str] = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            system=system_prompt,
            tools=WORKER_TOOL_SPECS,
            messages=messages,
        )

        # Collect any text the assistant produced this turn
        for block in response.content:
            if hasattr(block, "text") and block.text:
                notes_parts.append(block.text)

        # Build the assistant turn for history.
        # Serialize SDK content blocks to plain dicts — the Anthropic API requires
        # plain dicts in the messages list, not SDK objects.
        serialized_content = []
        for block in response.content:
            if block.type == "text":
                serialized_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                serialized_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": serialized_content})

        # If no tool calls → LLM is done
        if response.stop_reason != "tool_use":
            break

        # Process tool calls and build the tool_result turn
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_result_str = _dispatch_tool(block.name, block.input)

            # Track written/created files
            if block.name in ("write_file", "create_file"):
                fp = block.input.get("file_path")
                if fp and fp not in touched_files:
                    touched_files.append(fp)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    if iterations >= max_iterations:
        notes_parts.append(
            f"[WorkerAgent] WARNING: hit tool iteration cap ({max_iterations}). "
            "Execution may be incomplete."
        )

    return messages, touched_files, "\n\n".join(notes_parts)

def _parse_review_score(review_text: str) -> int:
    """Extract the integer from a line like 'SCORE: 8'."""
    for line in review_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SCORE:"):
            try:
                return max(0, min(10, int(stripped.split(":", 1)[1].strip())))
            except ValueError:
                pass
    return 0

# Nodes
def plan_task(state: WorkerState) -> Dict[str, Any]:
    task: Task = state["task"]
    client = Anthropic()

    # Read project schema for context
    try:
        schema = get_project_schema(task.project_id)
        schema_summary = json.dumps(schema, indent=2)
    except Exception as exc:
        schema_summary = f"(Could not load project schema: {exc})"

    system = textwrap.dedent(f"""
        You are a senior software engineer planning implementation of a coding task.
        You have full context of the project schema below.

        Your job: produce a clear, numbered implementation plan.
        - List every file you will create or modify (with the relative path).
        - For each file, describe exactly what functions/classes/changes are needed.
        - Note any dependencies on other tasks or files that might be missing.
        - Keep the plan concise but precise — this is what you will execute next.

        Project schema:
        {schema_summary}
    """).strip()

    user_message = textwrap.dedent(f"""
        Task ID   : {task.task_id}
        Task desc : {task.desc}
        Worker ID : {state['worker_id']}
        Dependencies already completed: {task.dependency_list}

        Produce your implementation plan now.
    """).strip()

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    plan = response.content[0].text if response.content else "(empty plan)"
    logger.info("[%s] Plan produced (%d chars)", task.task_id, len(plan))

    return {
        "plan": plan,
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": plan},
        ],
        "rework_count": 0,
        "touched_files": [],
        "execution_notes": "",
        "review_score": 0,
        "review_feedback": "",
        "test_file_path": None,
        "test_code": "",
        "report": None,
    }


def execute_task(state: WorkerState) -> Dict[str, Any]:
    task: Task = state["task"]
    client = Anthropic()

    system = textwrap.dedent(f"""
        You are a senior software engineer implementing a coding task.
        Use the provided tools to read, create, and write files in the project.

        Rules:
        1. Always call get_project_schema first if you haven't seen it this session.
        2. Always call read_file before writing to any existing file so you have
           accurate line numbers.
        3. Write clean, well-commented, production-quality code.
        4. When you are fully done (all files written), respond with a plain-text
           summary of everything you implemented — do NOT call any more tools.

        Your implementation plan:
        {state['plan']}

        Project ID : {task.project_id}
        Worker ID  : {state['worker_id']}
        Task ID    : {task.task_id}
    """).strip()

    initial_messages = [
        {
            "role": "user",
            "content": (
                f"Execute the implementation plan now for task: {task.desc}\n\n"
                f"Start by calling get_project_schema to confirm the current state "
                f"of the project, then proceed file by file."
            ),
        }
    ]

    messages, touched_files, notes = _run_tool_loop(client, system, initial_messages)
    logger.info("[%s] Execution done. Touched: %s", task.task_id, touched_files)

    return {
        "messages": messages,
        "touched_files": touched_files,
        "execution_notes": notes,
    }


def review_task(state: WorkerState) -> Dict[str, Any]:
    task: Task = state["task"]
    client = Anthropic()

    # Read all touched files for the reviewer
    files_content = {}
    for fp in state.get("touched_files", []):
        try:
            raw = read_file(task.project_id, [fp])
            lines = raw[fp]
            files_content[fp] = "\n".join(lines.values())
        except Exception as exc:
            files_content[fp] = f"(Could not read: {exc})"

    files_block = "\n\n".join(
        f"=== {fp} ===\n{content}"
        for fp, content in files_content.items()
    ) or "(no files were written)"

    rework_context = ""
    if state["rework_count"] > 0:
        rework_context = (
            f"\nThis is rework cycle {state['rework_count']}. "
            f"Previous feedback was:\n{state['review_feedback']}\n"
        )

    system = textwrap.dedent("""
        You are a strict but fair senior code reviewer.
        Review the implementation below against the task specification.

        Respond in this EXACT format (fill in the blanks):

        SCORE: <integer 0-10>
        VERDICT: <PASS|FAIL>
        ISSUES:
        - <issue 1>
        - <issue 2>
        (or "- None" if no issues)
        SUGGESTIONS:
        - <non-blocking suggestion 1>
        (or "- None")
        SUMMARY:
        <2-3 sentence summary of the review>

        Scoring guide:
          10 — perfect, production-ready
           8-9 — good, minor style issues only  (PASS)
           7 — acceptable, some improvements needed (PASS)
           5-6 — significant gaps, needs rework (FAIL)
           0-4 — major problems or task not completed (FAIL)
    """).strip()

    user_message = textwrap.dedent(f"""
        Task specification:
        {task.desc}

        Implementation plan followed:
        {state['plan']}
        {rework_context}
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

    # Parse structured fields out of the review text
    score = _parse_review_score(review_text)
    feedback = review_text

    logger.info(
        "[%s] Review score: %d/10 (rework cycle %d)",
        task.task_id, score, state["rework_count"],
    )

    return {
        "review_score": score,
        "review_feedback": feedback,
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": review_text},
        ],
    }

def rework_task(state: WorkerState) -> Dict[str, Any]:
    task: Task = state["task"]
    client = Anthropic()

    system = textwrap.dedent(f"""
        You are a senior software engineer fixing issues found in a code review.
        Use the provided tools to correct the identified problems.

        Original task:
        {task.desc}

        Review feedback to address:
        {state['review_feedback']}

        Rules:
        1. Read the relevant files first (read_file) to get current line numbers.
        2. Make targeted, minimal changes — do not rewrite things that were correct.
        3. When all issues are resolved, respond with a plain-text summary of the
           changes you made.

        Project ID : {task.project_id}
        Worker ID  : {state['worker_id']}
    """).strip()

    initial_messages = [
        {
            "role": "user",
            "content": (
                "Please fix the issues identified in the review now. "
                "Start by reading the relevant files to confirm current state."
            ),
        }
    ]

    messages, new_touched, notes = _run_tool_loop(client, system, initial_messages)

    # Merge touched files (don't duplicate)
    existing = state.get("touched_files", [])
    merged = list(existing)
    for fp in new_touched:
        if fp not in merged:
            merged.append(fp)

    existing_notes = state.get("execution_notes", "")
    combined_notes = (
        existing_notes
        + f"\n\n--- Rework cycle {state['rework_count'] + 1} ---\n"
        + notes
    )

    logger.info("[%s] Rework cycle %d done.", task.task_id, state["rework_count"] + 1)

    return {
        "messages": messages,
        "touched_files": merged,
        "execution_notes": combined_notes,
        "rework_count": state["rework_count"] + 1,
    }

def generate_tests(state: WorkerState) -> Dict[str, Any]:
    task: Task = state["task"]
    client = Anthropic()

    # Gather final file contents for the LLM to test against
    files_content = {}
    for fp in state.get("touched_files", []):
        try:
            raw = read_file(task.project_id, [fp])
            files_content[fp] = "\n".join(raw[fp].values())
        except Exception as exc:
            files_content[fp] = f"(Could not read: {exc})"

    files_block = "\n\n".join(
        f"=== {fp} ===\n{content}"
        for fp, content in files_content.items()
    ) or "(no files to test)"

    system = textwrap.dedent("""
        You are a senior test engineer writing pytest test cases.

        Write a COMPLETE, runnable pytest file.
        Requirements:
        - Use pytest fixtures for any shared setup.
        - Cover happy paths, edge cases, and error paths.
        - Use monkeypatch / unittest.mock where external I/O is involved.
        - Each test function must have a clear docstring explaining what it tests.
        - Do NOT include any imports that aren't in the standard library or pytest
          unless they are modules from the project itself.
        - Output ONLY the raw Python source of the test file — no markdown fences,
          no explanation text, just the code.
    """).strip()

    user_message = textwrap.dedent(f"""
        Task that was implemented:
        {task.desc}

        Source files to test:
        {files_block}

        Write the complete pytest test file now.
    """).strip()

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    test_code = response.content[0].text if response.content else ""

    # Strip accidental markdown fences
    if test_code.startswith("```"):
        lines = test_code.splitlines()
        test_code = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    # Save the test file into the project
    test_file_path = f"tests/test_{task.task_id}.py"
    worker_id = state["worker_id"]

    create_result = create_file(
        task.project_id, test_file_path,
        f"Auto-generated tests for task {task.task_id}",
        worker_id,
    )

    if create_result is None:
        # File created — now write the content
        edits = [Edit(operation="insert", start_line=1, end_line=None, new_value=test_code)]
        write_result = write_file(task.project_id, worker_id, test_file_path, edits)
        if write_result:
            logger.warning("[%s] Could not write test file: %s", task.task_id, write_result)
            test_file_path = None
    else:
        logger.warning("[%s] Could not create test file: %s", task.task_id, create_result)
        test_file_path = None

    logger.info("[%s] Tests generated → %s", task.task_id, test_file_path)

    return {
        "test_file_path": test_file_path,
        "test_code": test_code,
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": test_code},
        ],
    }


def report_to_teamlead(state: WorkerState) -> Dict[str, Any]:
    task: Task = state["task"]
    client = Anthropic()

    # Ask the LLM to distil a clean summary and extract structured fields
    system = textwrap.dedent("""
        You are a software engineer writing a status report for your team lead.
        Given the task description, review feedback, and execution notes, produce
        a JSON object with EXACTLY these fields:

        {
          "summary": "<2-3 sentence plain-English summary of what was done>",
          "blockers": ["<blocker 1>", ...],
          "suggestions": ["<suggestion 1>", ...]
        }

        blockers — things that MUST be resolved before this task can be merged
                   (e.g. missing dependency file, compilation error, failing test).
                   Empty list if none.
        suggestions — non-blocking improvements the team lead may want to address
                   in a follow-up task.  Empty list if none.

        Output ONLY the JSON object, no markdown, no explanation.
    """).strip()

    user_message = textwrap.dedent(f"""
        Task: {task.desc}
        Review score: {state['review_score']}/10
        Review feedback:
        {state['review_feedback']}
        Execution notes:
        {state['execution_notes']}
    """).strip()

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text if response.content else "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort fallback
        parsed = {
            "summary": state["execution_notes"][:300],
            "blockers": [],
            "suggestions": [],
        }

    # Determine status
    score = state["review_score"]
    blockers = parsed.get("blockers", [])
    rework_exhausted = state["rework_count"] >= MAX_REWORK_CYCLES and score < REVIEW_PASS_THRESHOLD

    if blockers:
        status: Literal["completed", "completed_with_issues", "blocked", "failed"] = "blocked"
    elif rework_exhausted:
        status = "failed"
    elif score >= REVIEW_PASS_THRESHOLD:
        status = "completed" if score >= 9 else "completed_with_issues"
    else:
        status = "failed"

    report: WorkerReport = {
        "task_id": task.task_id,
        "worker_id": state["worker_id"],
        "status": status,
        "summary": parsed.get("summary", ""),
        "touched_files": state.get("touched_files", []),
        "test_file_path": state.get("test_file_path"),
        "review_score": score,
        "review_feedback": state["review_feedback"],
        "blockers": blockers,
        "suggestions": parsed.get("suggestions", []),
        "execution_notes": state.get("execution_notes", ""),
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }

    logger.info(
        "[%s] Report ready — status=%s score=%d/10 blockers=%d",
        task.task_id, status, score, len(blockers),
    )

    return {"report": report}

# Conditional edge
def should_rework(state: WorkerState) -> Literal["rework_task", "generate_tests"]:
    """
    After review: rework if score is below threshold AND we haven't hit the cap.
    Otherwise proceed to test generation.
    """
    score = state.get("review_score", 0)
    rework_count = state.get("rework_count", 0)

    if score < REVIEW_PASS_THRESHOLD and rework_count < MAX_REWORK_CYCLES:
        logger.info(
            "Review score %d < threshold %d, rework %d/%d — routing to rework",
            score, REVIEW_PASS_THRESHOLD, rework_count, MAX_REWORK_CYCLES,
        )
        return "rework_task"

    logger.info(
        "Review score %d — routing to generate_tests (rework cycles used: %d)",
        score, rework_count,
    )
    return "generate_tests"

# build graph
def build_worker_graph() -> StateGraph:
    graph = StateGraph(WorkerState)

    # Nodes
    graph.add_node("plan_task", plan_task)
    graph.add_node("execute_task", execute_task)
    graph.add_node("review_task", review_task)
    graph.add_node("rework_task", rework_task)
    graph.add_node("generate_tests", generate_tests)
    graph.add_node("report_to_teamlead", report_to_teamlead)

    # Linear edges
    graph.add_edge(START, "plan_task")
    graph.add_edge("plan_task", "execute_task")
    graph.add_edge("execute_task", "review_task")

    # Conditional edge: review → rework OR generate_tests
    graph.add_conditional_edges(
        "review_task",
        should_rework,
        {
            "rework_task": "rework_task",
            "generate_tests": "generate_tests",
        },
    )

    # Rework loops back to review
    graph.add_edge("rework_task", "review_task")

    # After tests, always report
    graph.add_edge("generate_tests", "report_to_teamlead")
    graph.add_edge("report_to_teamlead", END)

    return graph


def create_worker_agent():
    """Compile and return the runnable worker agent."""
    graph = build_worker_graph()
    return graph.compile()

def run_worker(task: Task, worker_id: str) -> WorkerReport:
    """
    Execute a task end-to-end and return the WorkerReport.

    Args:
        task:       The Task dataclass assigned by TeamLeadAgent.
        worker_id:  Stable identifier for this worker instance.

    Returns:
        WorkerReport TypedDict ready to be sent to TeamLeadAgent.
    """
    agent = create_worker_agent()

    initial_state: WorkerState = {
        "task": task,
        "worker_id": worker_id,
        "plan": "",
        "touched_files": [],
        "execution_notes": "",
        "review_score": 0,
        "review_feedback": "",
        "rework_count": 0,
        "test_file_path": None,
        "test_code": "",
        "messages": [],
        "report": None,
    }

    final_state = agent.invoke(initial_state)
    report = final_state.get("report")

    if report is None:
        # Should never happen — defensive fallback
        report = WorkerReport(
            task_id=task.task_id,
            worker_id=worker_id,
            status="failed",
            summary="Agent completed without producing a report.",
            touched_files=final_state.get("touched_files", []),
            test_file_path=None,
            review_score=0,
            review_feedback="",
            blockers=["No report was generated — check agent logs."],
            suggestions=[],
            execution_notes=final_state.get("execution_notes", ""),
            completed_at=datetime.utcnow().isoformat() + "Z",
        )

    return report