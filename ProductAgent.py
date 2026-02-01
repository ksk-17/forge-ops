# Product agent
# Converts user ask → PRD + acceptance criteria + constraints
# Produces “Definition of Done” checklist

from langgraph.graph import StateGraph, START, END
from langchain.messages import AIMessage, SystemMessage, HumanMessage, AnyMessage
from langchain.chat_models import init_chat_model
from typing import TypedDict, List, Annotated, Optional
from pydantic import BaseModel
import operator
import os

from logging_config import get_logger

logger = get_logger("product_agent")

# initialization
MAX_CALLS = int(os.getenv("PRODUCT_AGENT_MAX_CALLS", "3"))

# prompts
PRODUCT_PROMPT = """
You are the Product Agent (Product Owner) in a multi-agent software delivery pipeline.

IMPORTANT CONTEXT:
Your output will be handed to a downstream Solution Architect agent.
- The Solution Architect is responsible for technical design, data models, API design, security controls, privacy implementation details, performance targets, observability/logging, storage choices, and test strategy.
- Therefore, you MUST focus on product behavior and business rules, not implementation.

Goal:
Convert the user's request into a clear, minimal, implementation-ready PRODUCT PRD: user-visible features, workflows, and business rules.
Do not over-engineer or add enterprise-grade NFRs unless the user explicitly asks.

OUTPUT (STRICT):
Return output parseable into an object with EXACTLY these 3 fields:
- requirements (string)
- acceptance_criteria (string)
- constraints (string)

Each field MUST be a single STRING containing ONLY bullet lines starting with "- ".
No paragraphs.

TRACEABILITY (MANDATORY):
- Every requirement bullet MUST start with "R1:", "R2:", ...
- Every acceptance criteria bullet MUST start with "AC R#.x:" and reference exactly one requirement.
- Multiple AC bullets per requirement are allowed and expected.
- Coverage rule: each requirement must have at least one acceptance criteria bullet.

AC ATOMICITY (MANDATORY):
- Each acceptance criteria bullet MUST be exactly ONE Given/When/Then scenario.
- Do NOT chain scenarios (no semicolons, no multiple When/Then in one bullet). Split into separate bullets.

SCOPE RULES (CRITICAL):
- Only include requirements that are directly implied by the user ask.
- Do NOT introduce new major features (accounts, sync, analytics, encryption, advanced accessibility, rate-limiting, compliance regimes) unless explicitly requested.
- Keep quality constraints lightweight and user-centric (e.g., “app should not lose data after restart” only if implied).

AMBIGUITY RULE:
- Do NOT ask questions.
- If something is missing but required to define behavior, add a minimal assumption under constraints as "Assumption: ...".
- Capture non-goals and exclusions as "Out of scope: ...".

CONSTRAINTS CONTENT (PRODUCT-LEVEL ONLY):
- Assumptions, out-of-scope, priorities (MVP vs later), UX decisions, policy/business rules.
- Do NOT specify implementation details (databases, encryption mechanisms, OS classes, logging rate limits, exact latency p95, etc.). Leave these to the Solution Architect.

Now produce the PRD from the latest user ask.
""".strip()

PRODUCT_PROMPT_REVIEW_EXTRACTION = """
You are the Product Agent (Product Owner) revising a PRD based on reviewer feedback.

IMPORTANT CONTEXT:
Your output will be handed to a downstream Solution Architect agent who owns technical design (data models, APIs, security/privacy implementation details, performance targets, observability, storage choices).
Do NOT add technical implementation requirements unless the user explicitly asked.

You will be given:
- The user's original ask
- The current PRD (requirements, acceptance_criteria, constraints as strings)
- Reviewer comments

TASK:
Revise the PRD to satisfy valid reviewer comments while staying faithful to the user's ask AND keeping product-level scope.

OUTPUT (STRICT):
Return output parseable into an object with EXACTLY these 3 fields:
- requirements (string)
- acceptance_criteria (string)
- constraints (string)

Each field is a single STRING containing ONLY bullet lines starting with "- ".

TRACEABILITY (MANDATORY):
- Requirements: "R1:", "R2:", ...
- Acceptance Criteria: "AC R#.x:" references exactly one requirement.
- Ensure each requirement has at least one acceptance criteria bullet.

AC ATOMICITY (MANDATORY):
- Each AC bullet is exactly one Given/When/Then scenario.
- No chained scenarios; split into multiple bullets.

REVISION RULES:
- Fix missing coverage and make acceptance criteria atomic.
- Remove accidental technical over-specification (encryption/logging rate limits/device lock classes/perf p95 targets) unless the user asked for them.
- If reviewer requests technical details, record in constraints as:
  "Open issue: Technical design detail to be handled by Solution Architect: ..."

CONSTRAINTS CONTENT (PRODUCT-LEVEL ONLY):
- Assumption / Out of scope / Open issue / priority decisions.
- No deep technical constraints.

Return the corrected PRD.
""".strip()

REVIEW_PROMPT = """
You are the PRD Review Agent.

You will evaluate the candidate PRD and return:
- satisfied: true if the PRD is implementation-ready at the PRODUCT level (for handoff to Solution Architect)
- comments: bullet-point fix instructions

Context:
This PRD will be handed to a downstream Solution Architect agent who owns technical design decisions
(data models, APIs, security/privacy implementation details, performance targets, observability/logging, storage choices).
Therefore, do NOT require deep technical constraints unless explicitly requested by the user.

EXPECTED SCHEMA:
- requirements: STRING containing only bullet lines (each line starts with "- ")
- acceptance_criteria: STRING containing only bullet lines (each line starts with "- ")
- constraints: STRING containing only bullet lines (each line starts with "- ")

Rubric (MUST-FIX if any item fails):
1) Structure/format:
   - Requirements/AC/Constraints each contain only bullet lines starting with "- "
   - No extra sections or prose paragraphs
2) Testability:
   - Each acceptance_criteria bullet is exactly ONE atomic Given/When/Then scenario
   - No chained scenarios (no semicolons, no multiple When/Then in one bullet)
   - Avoid vague terms; if user says "fast/secure/smooth", require measurable product-level targets OR mark as assumption/open issue
3) Coverage:
   - Each requirement has at least one acceptance criteria bullet that verifies it
4) Product-level completeness (NOT technical design):
   - Requirements cover the user-requested workflows and behaviors
   - NFRs are required ONLY if explicitly mentioned by the user; otherwise they may appear as:
     "Open issue: NFR targets to be defined by Solution Architect"
5) Ambiguity handling:
   - Assumptions appear under constraints as "Assumption: ..."
6) Scope control:
   - Out of scope / dependencies / open issues captured under constraints when relevant
7) Consistency:
   - No contradictions across sections

Decision rule:
- satisfied=true ONLY if there are no must-fix issues.

Return comments as bullet points starting with:
- Must fix:
- Nice to have:

Now review the candidate PRD.
""".strip()

# result classes
class AnswerClass(BaseModel):
    requirements: str
    acceptance_criteria: str
    constraints: str

class ReviewClass(BaseModel):
    satisfied: bool
    comments: str

# model
model = init_chat_model('gpt-5.2')

# state
class ProductAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add] = []
    num_calls: int = 0
    last_answer: Optional[AnswerClass]
    last_review: Optional[ReviewClass]

# util functions
def get_human_message(messages: List[AnyMessage]) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            return m.content
    return ""

def render_answer(answer: AnswerClass) -> str:
    return (
        f"Requirements:\n{answer.requirements}\n\n"
        f"Acceptance Criteria:\n{answer.acceptance_criteria}\n\n"
        f"Constraints:\n{answer.constraints}"
    )

def render_review(review: ReviewClass) -> str:
    return (
        f"Review Satisfied:\n{review.satisfied}\n\n"
        f"Comments:\n{review.comments}"
    )

# nodes
def answer_node(state: ProductAgentState) -> ProductAgentState:
    """LLM genrates the answer"""

    logger.debug("answer_node started")
    human_text = get_human_message(state['messages'])

    if (state["num_calls"] == 0):
        system_prompt = PRODUCT_PROMPT + "\n\nUser ask:\n" + human_text
        logger.debug("answer_node using product_prompt")
    else:
        review_text = state['last_review'].comments if state['last_review'] else ""
        prev_answer = state['last_answer']
        prev_answer_text = (
            f"Requirements:\n{prev_answer.requirements}\n\n"
            f"Acceptance Criteria:\n{prev_answer.acceptance_criteria}\n\n"
            f"Constraints:\n{prev_answer.constraints}"
        )

        system_prompt = (
            PRODUCT_PROMPT_REVIEW_EXTRACTION
            + "\n\nUser ask:\n" + human_text
            + "\n\nCurrent PRD:\n" + prev_answer_text
            + "\n\nReviewer comments:\n" + review_text
        )
        logger.debug("answer_node using product_prompt_review_extraction")

    llm = model.with_structured_output(AnswerClass)
    answer: AnswerClass = llm.invoke([SystemMessage(content = system_prompt)])
    logger.debug("answer_node has recieved the answer")

    rendered = render_answer(answer)

    logger.debug("answer_node ended")
    return {
        "messages": state['messages'] + [AIMessage(content=rendered)],
        'num_calls': state['num_calls'] + 1,
        'last_answer': answer,
        'last_review': state.get('last_review')
    }

def review_node(state: ProductAgentState) -> ProductAgentState:
    """LLM reviews the answer and states if the answer is satisfied"""

    logger.debug("review_node started")

    last_answer = state['last_answer'] if state['last_answer'] else ""
    last_answer_text = render_answer(last_answer)

    review_prompt = f"{REVIEW_PROMPT}\n\nCandidate Answer:\n{last_answer_text}"

    llm = model.with_structured_output(ReviewClass)
    review: ReviewClass = llm.invoke([SystemMessage(content = review_prompt)])
    logger.debug("review_node has recieved the review")

    rendered = render_review(review)

    logger.debug("review_node ended")
    return {
        "messages": state['messages'] + [AIMessage(content=rendered)],
        'num_calls': state['num_calls'],
        'last_answer': state.get("last_answer"),
        'last_review': review
    }

def should_continue(state: ProductAgentState) -> str:
    """Decide if we should continue or not"""

    logger.debug("should_continue started")

    num_calls = state.get("num_calls", 0)

    review = state['last_review']
    if review and review.satisfied:
        logger.debug("should_continue hit end")
        return "end"
    else: 
        if num_calls >= MAX_CALLS:
            logger.warning("MAX_CALLS reached. Forcing end")
            return "end"
        logger.debug("should_continue hit continue")
        return "continue"

# build the agent
agent_builder = StateGraph(ProductAgentState)

# add nodes
agent_builder.add_node("answer_node", answer_node)
agent_builder.add_node("review_node", review_node)

# add edges
agent_builder.add_edge(START, "answer_node")
agent_builder.add_edge("answer_node", "review_node")
agent_builder.add_conditional_edges("review_node", should_continue, {"continue": "answer_node", "end": END})

# compile
agent = agent_builder.compile()