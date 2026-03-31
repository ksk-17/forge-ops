from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from anthropic import Anthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict

from models import ArchitectureSpec, Question, RequirementsReview, UserAnswer

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
MAX_CLARIFICATION_ROUNDS = 4
COMPLETENESS_THRESHOLD = 8
MAX_QUESTIONS_PER_ROUND = 4

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

def analyse_input(state: ArchitectState) -> Dict[str, Any]:
    system = textwrap.dedent("""
        You are a senior software architect gathering requirements from a client.
        Read their initial project description carefully.

        Produce a structured understanding in plain text with these sections:
        ## What is clear
        ## What is implied (reasonable assumptions)
        ## What is missing or ambiguous
        ## Initial project name suggestion

        Be concise but thorough. Do not ask questions yet — just analyse.
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
        You are a senior software architect clarifying requirements.
        Generate up to {MAX_QUESTIONS_PER_ROUND} targeted questions to fill
        the gaps identified in the current understanding.

        Each question must have a clear purpose — only ask what is genuinely
        needed to write a complete architecture specification.

        Question types available:
          yes_no          — for binary decisions
          multiple_choice — when there are 2-4 distinct options
          text            — for open-ended clarification

        Output a JSON array ONLY. No markdown, no explanation. Each item:
        {{
          "id": "q<N>",
          "question": "<question text>",
          "type": "yes_no" | "multiple_choice" | "text",
          "options": ["<opt1>", "<opt2>"] or null,
          "reason": "<why this affects the architecture>"
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
        You are a senior architect assessing requirements completeness.

        Score the understanding from 0 to 10:
          10 — everything needed to write a full architecture spec is known
           8 — minor gaps but sufficient to proceed
           6 — important gaps remain, another round would help significantly
           4 — major gaps, architecture would be guesswork
           0 — essentially no requirements

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
    return build_architect_graph().compile(
        checkpointer=checkpointer,
        interrupt_before=["ask_user", "present_summary"],
    )

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

    # First invocation — runs until the first interrupt
    result = agent.invoke(initial_state, config)

    # Drive the interrupt/resume loop
    while True:
        state = agent.get_state(config)

        # Check if the graph has finished
        if not state.next:
            break

        next_node = state.next[0]

        # ── ask_user interrupt ──────────────────────────────────────────
        if next_node == "ask_user":
            # The interrupt value is the question prompt string
            interrupt_value = _get_interrupt_value(state)
            print(interrupt_value)

            # Collect answers line by line until blank line
            print()
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                except EOFError:
                    break

            human_input = "\n".join(lines)
            agent.update_state(config, {"human_input": human_input}, as_node="ask_user")
            agent.invoke(None, config)

        # ── present_summary interrupt ───────────────────────────────────
        elif next_node == "present_summary":
            interrupt_value = _get_interrupt_value(state)
            print(interrupt_value)
            print()

            try:
                decision = input("  Your decision: ").strip()
            except EOFError:
                decision = "reject"

            agent.update_state(config, {"human_input": decision}, as_node="present_summary")
            agent.invoke(None, config)

        else:
            # Non-interrupt node — shouldn't happen, but break to avoid infinite loop
            logger.warning("Unexpected next node: %s", next_node)
            break

    # Retrieve final state
    final_state = agent.get_state(config).values
    spec = final_state.get("architecture_spec")

    if spec:
        logger.info(
            "[%s] Architecture spec produced: %s (%d chars)",
            project_id, spec["project_name"], len(spec["spec_text"]),
        )
    else:
        logger.info("[%s] No architecture spec produced (user rejected or aborted).", project_id)

    return spec


def _get_interrupt_value(state) -> str:
    # LangGraph stores interrupt values in state.tasks[*].interrupts[*].value
    for task in state.tasks:
        for interrupt_obj in getattr(task, "interrupts", []):
            return str(getattr(interrupt_obj, "value", ""))
    return ""