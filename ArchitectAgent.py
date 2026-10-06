from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import sys, select, os

from anthropic import Anthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict
from models import ArchitectureSpec, Question, RequirementsReview, UserAnswer
from forge_memory import get_architect_context, record_architect_run

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
MAX_CLARIFICATION_ROUNDS = 4
COMPLETENESS_THRESHOLD = 8
MAX_QUESTIONS_PER_ROUND = 3

class ArchitectState(TypedDict):
    project_id: str
    raw_input: str
    understanding: str
    questions: List[Question]
    answers: List[UserAnswer]
    clarification_round: int
    completeness_score: int
    human_input: str
    user_review: Optional[RequirementsReview]
    architecture_spec: Optional[ArchitectureSpec]

# Helpers

def _llm(system: str, user: str, max_tokens: int = 2048) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text if response.content else ""


def _parse_questions(raw: str) -> List[Question]:
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        )
    try:
        items = json.loads(text)
        questions: List[Question] = []
        for i, item in enumerate(items[:MAX_QUESTIONS_PER_ROUND]):
            q: Question = {
                "id": item.get("id", f"q{i+1}"),
                "question": item["question"],
                "type": item.get("type", "text"),
                "options": item.get("options"),
                "reason": item.get("reason", ""),
            }
            questions.append(q)
        return questions
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("_parse_questions: failed to parse: %s", exc)
        return []


def _format_question_for_display(q: Question, index: int) -> str:
    lines = [f"\n  Q{index}: {q['question']}"]
    if q["reason"]:
        lines.append(f"        (Why: {q['reason']})")
    if q["type"] == "yes_no":
        lines.append("        Answer: [y/n]")
    elif q["type"] == "multiple_choice" and q["options"]:
        for i, opt in enumerate(q["options"], 1):
            lines.append(f"        {i}. {opt}")
        lines.append(f"        Answer: [1-{len(q['options'])}]")
    else:
        lines.append("        Answer: (type your response)")
    return "\n".join(lines)


def _normalise_answer(answer: str, q: Question) -> str:
    answer = answer.strip()
    if q["type"] == "yes_no":
        lower = answer.lower()
        if lower in ("y", "yes", "1", "true"): return "yes"
        if lower in ("n", "no", "0", "false"): return "no"
        return answer
    if q["type"] == "multiple_choice" and q["options"]:
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(q["options"]):
                return q["options"][idx]
        except ValueError:
            pass
        return answer
    return answer


def _render_understanding(understanding: str, answers: List[UserAnswer]) -> str:
    lines = [
        "=" * 64,
        "  ARCHITECT'S UNDERSTANDING OF YOUR REQUIREMENTS",
        "=" * 64,
        "",
        understanding,
        "",
    ]
    if answers:
        lines += ["─" * 64, "  CLARIFICATIONS YOU PROVIDED", "─" * 64]
        for ans in answers:
            lines.append(f"  Q: {ans['question']}")
            lines.append(f"  A: {ans['answer']}")
            lines.append("")
    lines += [
        "─" * 64,
        "  Do you accept this understanding?",
        "  Options:",
        "    accept  — proceed to architecture generation",
        "    change  — describe what needs to change",
        "    reject  — abort",
        "─" * 64,
    ]
    return "\n".join(lines)


# Graph nodes

def analyse_input(state: ArchitectState) -> Dict[str, Any]:
    memory_ctx = get_architect_context(state['raw_input'])

    memory_section = f"\n{memory_ctx}\n" if memory_ctx else ""

    system = textwrap.dedent(f"""
        You are a senior software architect turning a client's rough idea into a
        structured understanding.
        {memory_section}
        Read their description and produce a document with these sections:

        ## What is clear
        (Things explicitly stated by the client)

        ## Assumed by default
        (Industry-standard assumptions you will make WITHOUT asking — e.g. REST
        for APIs, PostgreSQL for relational storage, pytest for testing, error
        handling via exceptions, etc. Be generous here: assume the obvious.)

        ## Critical gaps — must ask
        (ONLY list things where a wrong assumption would cause a fundamentally
        different architecture. Examples: "Is this a web API or a CLI?" or
        "Does this need real-time sync?". Do NOT list style/naming/format
        preferences — those can always be decided by the engineer.)

        ## Initial project name suggestion

        Rule: if a reasonable default exists for something, assume it and list it
        under "Assumed by default". Only move something to "Critical gaps" if
        getting it wrong would waste significant engineering effort.
    """).strip()

    understanding = _llm(system, f"Project description:\n{state['raw_input']}")
    logger.info("[%s] analyse_input complete (%d chars)", state["project_id"], len(understanding))

    return {
        "understanding": understanding,
        "clarification_round": 0,
        "questions": [],
        "answers": [],
        "completeness_score": 0,
        "user_review": None,
        "architecture_spec": None,
    }


def generate_questions(state: ArchitectState) -> Dict[str, Any]:
    answered_ids = {a["question_id"] for a in state.get("answers", [])}
    prev_answers_block = ""
    if state.get("answers"):
        prev_answers_block = "Answers already collected:\n" + "\n".join(
            f"- {a['question']}: {a['answer']}"
            for a in state["answers"]
        )

    system = textwrap.dedent(f"""
        You are a senior software architect clarifying only the most critical
        unknowns before writing an architecture specification.

        STRICT RULES — violating these wastes the user's time:
        1. Ask at most {MAX_QUESTIONS_PER_ROUND} questions per round.
        2. Only ask about things where a WRONG assumption would cause a
           fundamentally different system design (e.g. wrong database engine,
           wrong deployment target, missing core feature category).
        3. DO NOT ask about: naming conventions, output formatting, error message
           verbosity, priority scales, date formats, confirmation dialogs,
           installation method, pip packaging, CLI library choice, or any other
           detail that a competent engineer would decide without client input.
        4. If there are fewer than {MAX_QUESTIONS_PER_ROUND} truly critical
           unknowns, output fewer questions. An empty array [] is valid and
           preferred over asking unnecessary questions.
        5. Assume standard industry defaults for everything not asked about.

        Question types:
          yes_no          — for binary architectural decisions
          multiple_choice — when options lead to genuinely different designs
          text            — only for open-ended unknowns with no obvious default

        Output a JSON array ONLY. No markdown, no explanation. Each item:
        {{
          "id": "q<N>",
          "question": "<question text — plain, no jargon>",
          "type": "yes_no" | "multiple_choice" | "text",
          "options": ["<opt1>", "<opt2>"] or null,
          "reason": "<one sentence: what architectural decision this unlocks>"
        }}
    """).strip()

    user_msg = (
        f"Current understanding:\n{state['understanding']}\n\n"
        f"{prev_answers_block}\n\n"
        f"Round: {state.get('clarification_round', 0) + 1} / {MAX_CLARIFICATION_ROUNDS}\n"
        "Generate questions now."
    )

    raw = _llm(system, user_msg)
    questions = _parse_questions(raw)

    logger.info(
        "[%s] generate_questions: %d questions in round %d",
        state["project_id"], len(questions), state.get("clarification_round", 0) + 1,
    )

    return {"questions": questions}


def ask_user(state: ArchitectState) -> Dict[str, Any]:
    questions = state.get("questions", [])
    if not questions:
        return {"answers": state.get("answers", [])}

    # Build the display prompt
    display_lines = [
        "\n" + "─" * 64,
        f"  CLARIFYING QUESTIONS  (Round {state.get('clarification_round', 0) + 1})",
        "─" * 64,
    ]
    for i, q in enumerate(questions, 1):
        display_lines.append(_format_question_for_display(q, i))

    display_lines += [
        "\n" + "─" * 64,
        "  Please answer each question in order.",
        "  Type one answer per line, then press Enter twice (blank line) to submit.",
        "─" * 64,
    ]

    prompt = "\n".join(display_lines)

    # Interrupt — hand control back to the host with the question prompt
    raw_input: str = interrupt(prompt)

    # Parse multi-line answers (one per question)
    raw_lines = [line for line in raw_input.strip().splitlines() if line.strip()]

    new_answers: List[UserAnswer] = []
    for i, q in enumerate(questions):
        raw_ans = raw_lines[i] if i < len(raw_lines) else ""
        new_answers.append({
            "question_id": q["id"],
            "question": q["question"],
            "answer": _normalise_answer(raw_ans, q),
        })

    all_answers = list(state.get("answers", [])) + new_answers

    return {"answers": all_answers, "human_input": ""}


def incorporate_answers(state: ArchitectState) -> Dict[str, Any]:
    answers_block = "\n".join(
        f"Q: {a['question']}\nA: {a['answer']}"
        for a in state.get("answers", [])
    )

    system = textwrap.dedent("""
        You are a senior software architect updating your understanding based
        on the client's answers to your clarifying questions.

        Rewrite the understanding document incorporating all answers.
        Keep the same sections:
        ## What is clear
        ## What is implied (reasonable assumptions)
        ## What is missing or ambiguous
        ## Initial project name suggestion

        If a previous gap is now resolved, remove it from "missing/ambiguous".
        If an answer opens a NEW gap, add it to "missing/ambiguous".
        Be concise and precise — this document will become the architecture spec.
    """).strip()

    user_msg = (
        f"Previous understanding:\n{state['understanding']}\n\n"
        f"Client answers:\n{answers_block}\n\n"
        "Update the understanding now."
    )

    updated = _llm(system, user_msg)
    logger.info("[%s] incorporate_answers: understanding updated", state["project_id"])

    return {
        "understanding": updated,
        "clarification_round": state.get("clarification_round", 0) + 1,
    }


def check_completeness(state: ArchitectState) -> Dict[str, Any]:
    system = textwrap.dedent("""
        You are a senior architect deciding whether requirements are complete
        enough to write a solid architecture spec.

        Score from 0-10 using this guide:
          10 — complete: all architectural decisions are known
           8 — sufficient: core shape is clear, minor details can be assumed
           6 — one more round of questions would significantly help
           4 — fundamental unknowns remain (e.g. don't know if web or CLI)
           0 — no useful information

        Key principle: if the core system shape is clear (what it does, how
        it stores data, who uses it), score >= 8 even if minor details are
        missing. Engineers fill in minor details — clients define the shape.

        Respond with EXACTLY this format (nothing else):
        SCORE: <integer>
        VERDICT: <one sentence>
    """).strip()

    raw = _llm(
        system,
        f"Understanding:\n{state['understanding']}\n\nScore completeness:",
        max_tokens=100,
    )

    score = 0
    for line in raw.splitlines():
        if line.strip().upper().startswith("SCORE:"):
            try:
                score = max(0, min(10, int(line.split(":", 1)[1].strip())))
            except ValueError:
                pass

    logger.info(
        "[%s] check_completeness: score=%d/10 round=%d",
        state["project_id"], score, state.get("clarification_round", 0),
    )

    return {"completeness_score": score}


def present_summary(state: ArchitectState) -> Dict[str, Any]:
    display = _render_understanding(
        state["understanding"],
        state.get("answers", []),
    )

    raw_input: str = interrupt(display)
    raw_lower = raw_input.strip().lower()

    if raw_lower.startswith("accept"):
        review: RequirementsReview = {"decision": "accepted", "change_notes": ""}
    elif raw_lower.startswith("change"):
        notes = raw_input.strip()
        if ":" in notes:
            notes = notes.split(":", 1)[1].strip()
        review = {"decision": "change", "change_notes": notes}
    else:
        review = {"decision": "rejected", "change_notes": ""}

    logger.info(
        "[%s] present_summary: user decision = %s",
        state["project_id"], review["decision"],
    )

    return {"user_review": review, "human_input": ""}


def handle_user_review(state: ArchitectState) -> Dict[str, Any]:
    review = state.get("user_review") or {}
    if review.get("decision") == "change" and review.get("change_notes"):
        # Treat the change request as a new "answer" so incorporate_answers
        # can update the understanding accordingly.
        change_answer: UserAnswer = {
            "question_id": "user-review-change",
            "question": "User review: requested change",
            "answer": review["change_notes"],
        }
        updated_answers = list(state.get("answers", [])) + [change_answer]
        return {"answers": updated_answers}

    # accepted or rejected — nothing to modify
    return {}


def produce_architecture_spec(state: ArchitectState) -> Dict[str, Any]:
    # Extract project name from understanding
    project_name = state["project_id"]
    for line in state["understanding"].splitlines():
        if "project name" in line.lower() and ":" in line:
            candidate = line.split(":", 1)[1].strip().strip("#").strip()
            if candidate:
                project_name = candidate
            break

    system = textwrap.dedent("""
        You are a senior software architect writing a formal architecture
        specification for a development team.

        Convert the confirmed requirements into a structured document following
        this EXACT template (fill in every section):

        # <Project Name> — Architecture Specification

        ## Overview
        <2-4 sentences: what the system does and its primary goal>

        ## Tech stack
        <list of languages, frameworks, databases, tools — be specific>

        ## Modules / Components
        For each module/component, use this sub-template:

        ### <Module Name>
        **Responsibility:** <what this module does>
        **Public interface:** <functions, classes, endpoints, or exports this module exposes>
        **Dependencies:** <other modules this one imports or calls — "none" if standalone>
        **Files:** <exact relative file paths, e.g. src/auth.py, src/models/user.py>

        ## Data models
        <describe key data structures / database schemas / TypedDicts>

        ## Constraints & non-functional requirements
        <performance, security, scalability, testing requirements, etc.>

        ## Out of scope
        <what will NOT be built in this iteration>

        Rules:
        - Be extremely precise about file paths, function signatures, and interfaces.
        - The team lead will decompose this into 1-3 file tasks — write at that level.
        - Every module must be independently testable.
        - No vague language ("handle", "manage", "process") — use concrete verbs.
        - Output ONLY the specification document. No preamble, no explanation.
    """).strip()

    answers_block = "\n".join(
        f"- {a['question']}: {a['answer']}"
        for a in state.get("answers", [])
    )

    user_msg = textwrap.dedent(f"""
        Confirmed requirements understanding:
        {state['understanding']}

        All client answers:
        {answers_block}

        Project ID: {state['project_id']}

        Produce the architecture specification now.
    """).strip()

    spec_text = _llm(system, user_msg, max_tokens=4096)

    spec: ArchitectureSpec = {
        "project_id": state["project_id"],
        "project_name": project_name,
        "spec_text": spec_text,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    logger.info(
        "[%s] produce_architecture_spec: spec ready (%d chars)",
        state["project_id"], len(spec_text),
    )

    # Record to telemetry store + mem0 memory layer
    record_architect_run(
        project_id=state["project_id"],
        spec=dict(spec),
        answers=state.get("answers", []),
        raw_input=state.get("raw_input", ""),
    )

    return {"architecture_spec": spec}

# Conditional edges

def route_after_completeness(
    state: ArchitectState,
) -> Literal["generate_questions", "present_summary"]:
    score = state.get("completeness_score", 0)
    rounds = state.get("clarification_round", 0)

    if score < COMPLETENESS_THRESHOLD and rounds < MAX_CLARIFICATION_ROUNDS:
        logger.info(
            "route_after_completeness → generate_questions "
            "(score=%d < %d, round %d/%d)",
            score, COMPLETENESS_THRESHOLD, rounds, MAX_CLARIFICATION_ROUNDS,
        )
        return "generate_questions"

    logger.info(
        "route_after_completeness → present_summary (score=%d, rounds=%d)",
        score, rounds,
    )
    return "present_summary"


def route_after_review(
    state: ArchitectState,
) -> Literal["incorporate_answers", "produce_architecture_spec", "__end__"]:
    decision = (state.get("user_review") or {}).get("decision", "rejected")

    if decision == "accepted":
        logger.info("route_after_review → produce_architecture_spec")
        return "produce_architecture_spec"
    elif decision == "change":
        logger.info("route_after_review → incorporate_answers (user requested change)")
        return "incorporate_answers"
    else:
        logger.info("route_after_review → __end__ (user rejected)")
        return "__end__"


# Graph assembly

def build_architect_graph() -> StateGraph:
    graph = StateGraph(ArchitectState)

    # Register nodes
    graph.add_node("analyse_input", analyse_input)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("ask_user", ask_user)
    graph.add_node("incorporate_answers", incorporate_answers)
    graph.add_node("check_completeness", check_completeness)
    graph.add_node("present_summary", present_summary)
    graph.add_node("handle_user_review", handle_user_review)
    graph.add_node("produce_architecture_spec", produce_architecture_spec)

    # Entry
    graph.add_edge(START, "analyse_input")
    graph.add_edge("analyse_input", "generate_questions")
    graph.add_edge("generate_questions", "ask_user")
    graph.add_edge("ask_user", "incorporate_answers")
    graph.add_edge("incorporate_answers", "check_completeness")

    # Conditional: loop or proceed
    graph.add_conditional_edges(
        "check_completeness",
        route_after_completeness,
        {
            "generate_questions": "generate_questions",
            "present_summary": "present_summary",
        },
    )

    graph.add_edge("present_summary", "handle_user_review")

    # Conditional: accept / change / reject
    graph.add_conditional_edges(
        "handle_user_review",
        route_after_review,
        {
            "incorporate_answers": "incorporate_answers",
            "produce_architecture_spec": "produce_architecture_spec",
            "__end__": END,
        },
    )

    graph.add_edge("produce_architecture_spec", END)

    return graph


def create_architect_agent(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    return build_architect_graph().compile(checkpointer=checkpointer)


def _drain_stdin() -> None:
    try:
        # Only meaningful on Unix ttys; select on Windows stdin is limited
        if sys.stdin.isatty() and hasattr(select, "select"):
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = os.read(sys.stdin.fileno(), 1024)
                if not ch:
                    break
    except Exception:
        pass   # Non-fatal — worst case the next input() gets a stale newline


def _read_line(prompt: str = "") -> str:
    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    try:
        return sys.stdin.readline().rstrip("\n").rstrip("\r")
    except EOFError:
        return ""


def _collect_question_answers(n_questions: int) -> str:
    _drain_stdin()
    lines = []
    consecutive_empty = 0
    while True:
        line = _read_line()

        if line == "":
            consecutive_empty += 1
            if lines:
                # Blank line after at least one answer → user submitted
                break
            if consecutive_empty >= 3:
                # Three consecutive empty reads = EOF (StringIO exhausted
                # or piped input ended) — stop to avoid infinite loop
                break
            continue   # leading blank or single stray blank → skip

        consecutive_empty = 0
        lines.append(line)
        if len(lines) >= n_questions:
            _drain_stdin()
            break

    return "\n".join(lines)


def _extract_interrupt(event: dict) -> str:
    raw = event.get("__interrupt__", ())
    if not raw:
        return ""
    first = raw[0]
    val = getattr(first, "value", None)
    if val is not None:
        return str(val)
    return str(first)


def run_architect_cli(project_id: str, raw_input: str) -> Optional[ArchitectureSpec]:
    agent = create_architect_agent()
    config = {"configurable": {"thread_id": project_id}}

    initial_state: ArchitectState = {
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

    input_val: Any = initial_state

    while True:
        # ── Run / resume the graph until it interrupts or finishes ────────
        interrupt_prompt = ""
        for event in agent.stream(input_val, config, stream_mode="updates"):
            prompt = _extract_interrupt(event)
            if prompt:
                interrupt_prompt = prompt
                break          # stop consuming — graph is paused, collect input now

        # ── Graph finished (no interrupt) ─────────────────────────────────
        if not interrupt_prompt:
            break

        # ── Identify which human node interrupted ─────────────────────────
        # state.next is the most reliable signal; the prompt text is backup.
        state = agent.get_state(config)
        next_node = state.next[0] if state.next else ""

        # ── ask_user ──────────────────────────────────────────────────────
        if next_node == "ask_user" or "CLARIFYING QUESTIONS" in interrupt_prompt:
            print(interrupt_prompt)
            import sys; sys.stdout.flush()
            n_questions = len(state.values.get("questions", [])) or 1
            human_input = _collect_question_answers(n_questions)
            input_val = Command(resume=human_input)

        # ── present_summary ───────────────────────────────────────────────
        elif next_node == "present_summary" or "UNDERSTANDING OF YOUR REQUIREMENTS" in interrupt_prompt:
            print(interrupt_prompt)
            print()
            # Drain any stale bytes (e.g. the blank line from the last answer
            # block) before reading the decision — without this, input()
            # returns "" immediately and the decision is silently "rejected".
            _drain_stdin()
            decision = _read_line("  Your decision (accept / change: <notes> / reject): ").strip()
            if not decision:
                decision = _read_line("  (Type accept, change: <notes>, or reject): ").strip()
            if not decision:
                decision = "reject"
            input_val = Command(resume=decision)

        # ── unknown interrupt ─────────────────────────────────────────────
        else:
            logger.warning("Unexpected interrupt at node=%s", next_node)
            print(interrupt_prompt)
            print()
            try:
                answer = input("  Your response: ").strip()
            except EOFError:
                answer = ""
            input_val = Command(resume=answer)

    # ── Retrieve final spec ───────────────────────────────────────────────
    final_state = agent.get_state(config).values
    spec = final_state.get("architecture_spec")

    if spec:
        logger.info(
            "[%s] Architecture spec produced: %s (%d chars)",
            project_id, spec["project_name"], len(spec["spec_text"]),
        )
    else:
        logger.info(
            "[%s] No architecture spec produced (user rejected or aborted).",
            project_id,
        )

    return spec