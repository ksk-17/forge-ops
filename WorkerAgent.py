from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import operator
from dataclasses import dataclass
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel
from models import ReviewClass, Task, Edit, render_review, render_task
from SharedState import SharedState
import os

from logging_config import get_logger

logger = get_logger("worker_agent")

MAX_CALLS = int(os.getenv("WORKER_AGENT_MAX_CALLS", "3"))

# prompts
WORKER_PROMPT = """You are a code editor agent tasked with completing work on files. 
Analyze the task description carefully and provide the necessary file edits to complete it.

Your response MUST be a JSON object with the following structure:
{
    "file_name": "path/to/file.ext",
    "changes": [
        {
            "op": "replace" | "insert_before" | "insert_after" | "delete",
            "identifier": "text to find in the file",
            "value": "replacement text or text to insert",
            "occurence": null or integer (use if identifier matches multiple times)
        }
    ]
}

Operations:
- "replace": Replace the identifier text with the value
- "insert_before": Insert value before the identifier
- "insert_after": Insert value after the identifier  
- "delete": Remove the identifier (value can be empty string)

Be precise with identifiers to avoid ambiguity. If an identifier could match multiple locations, specify the occurence (0-indexed).
Include necessary context around the identifier to make it unique."""

WORKER_PROMPT_WITH_REVIEW = """You are a code editor agent refining your previous work based on review feedback.

The review pointed out issues with your previous answer. Carefully address each concern and provide improved edits.

Your response MUST be a JSON object with the following structure:
{
    "file_name": "path/to/file.ext",
    "changes": [
        {
            "op": "replace" | "insert_before" | "insert_after" | "delete",
            "identifier": "text to find in the file",
            "value": "replacement text or text to insert",
            "occurence": null or integer
        }
    ]
}

Operations:
- "replace": Replace the identifier text with the value
- "insert_before": Insert value before the identifier
- "insert_after": Insert value after the identifier
- "delete": Remove the identifier (value can be empty string)

Make targeted corrections to address the review feedback. Ensure identifiers are specific and unambiguous."""

REVIEW_PROMPT = """You are a code review agent. Evaluate the provided answer and determine if it correctly addresses the task.

Review the candidate answer carefully and provide your assessment:

1. Does the answer correctly address the task requirements?
2. Are the file edits syntactically correct and well-formed?
3. Will the changes successfully implement the desired functionality?
4. Are there any potential issues, bugs, or incomplete aspects?

Your response MUST be a JSON object with the following structure:
{
    "satisfied": true | false,
    "comments": "detailed feedback on the answer quality and any issues found"
}

If satisfied=true, the task is considered complete. If satisfied=false, provide specific guidance on what needs to be fixed."""

# Prompt for generating test cases for the generated code
TEST_CASE_PROMPT = """You are a test-case generation agent.
Given a task and the generated source code, produce a complete, runnable pytest test file that verifies the behavior described in the task and exercised by the generated code.

Requirements:
- Output MUST be a JSON object matching the `AnswerClass` structure:
    {
        "file_name": "tests/test_<module_or_feature>.py",
        "changes": [
            {"op": "replace" | "insert_before" | "insert_after" | "delete", "identifier": "", "value": "<full file content>", "occurence": null}
        ]
    }

- When creating a new test file, write the entire file content as the `value` of a single change. It is acceptable to use `op: "replace"` with an empty `identifier` to indicate writing the file from scratch.
- Tests must be runnable with `pytest` and should not call external network services or LLMs. Use fixtures like `tmp_path` for filesystem work and avoid global state.
- The test file should import the module under test using the workspace-relative import (e.g., `from mymodule import MyClass`) and use clear, deterministic assertions.
- Provide fixtures and small helper functions inside the test file as needed. Keep tests focused, isolated, and fast.
- If the generated code requires specific inputs or setup, include those setup steps in the test file (creating files, populating data, etc.).
- Include at least one positive test (expected behavior) and one negative or edge-case test if applicable.

Output format and examples:
- Example `changes` entry to create a new file:
    {
        "op": "replace",
        "identifier": "",
        "value": "<entire pytest test file content here>",
        "occurence": null
    }

Be concise but complete: produce tests that another developer can run immediately with `pytest` to validate the generated code."""

# data classes
class AnswerClass(BaseModel):
    file_name: str
    changes: List[Edit]

model = init_chat_model('gpt-5-mini')

# util functions
def render_answer(answer: AnswerClass) -> str:
    return (
        f"File Name:\n{answer.file_name}\n\n"
        f"Changes:\n{answer.changes}\n\n"
    )

# agent state
class WorkerAgentState(TypedDict):
    task_id: str
    worker_id: str
    messages: Annotated[List[AnyMessage], operator.add]
    last_answer: Optional[AnswerClass]
    last_review: Optional[ReviewClass]
    num_calls: int
    shared_state: SharedState

# nodes
def answer_node(state: WorkerAgentState) -> WorkerAgentState:
    """LLM generates the answer"""

    logger.debug("answer_node started")

    shared_state = state['shared_state']
    task = shared_state.get_task(state['task_id'])
    
    if state.get('num_calls') == 0:
        system_prompt = WORKER_PROMPT + '\n\nTask Description:\n' + render_task(task)
    else:
        review_text = state['last_review'].comments if state['last_review'] else ''
        last_answer = state['last_answer']
        system_prompt = WORKER_PROMPT_WITH_REVIEW + '\n\nLast Answer:\n' + render_answer(last_answer) + '\n\nReview Comments:\n' + review_text
    
    llm = model.with_structured_output(AnswerClass)
    answer: AnswerClass = llm.invoke([SystemMessage(content=system_prompt)])
    logger.debug("answer_node has recieved the answer")

    rendered = render_answer(answer)

    shared_state.file_manager.create_file(shared_state.project_id, answer.file_name)
    lock_acquired = shared_state.file_manager.acquire_lock(shared_state.project_id, answer.file_name, state['worker_id'])
    if lock_acquired:
        file_path = shared_state.file_manager.abs_path(shared_state.project_id, answer.file_name)
        shared_state.file_editor.apply_patches(str(file_path), edits=answer.changes)
        logger.debug("applied changes successfully")
        shared_state.file_manager.release_lock(shared_state.project_id, answer.file_name, state['worker_id'])
    else:
        logger.debug("lock cannot be acquired")
        logger.debug("failed to apply changes")

    logger.debug("answer_node_ended")
    return {
        "task_id": state['task_id'],
        "worker_id": state['worker_id'],
        "messages": state['messages'] + [AIMessage(content=rendered)],
        "last_answer": answer,
        "last_review": state.get('last_review'),
        "num_calls": state['num_calls'] + 1,
        "shared_state": state['shared_state']
    }

def review_node(state: WorkerAgentState) -> WorkerAgentState:

    logger.debug("review_node started")

    last_answer = state['last_answer'] if state['last_answer'] else ''
    last_answer_text = render_answer(last_answer)

    system_prompt = f"{REVIEW_PROMPT}\n\nCandidate Answer:\n{last_answer_text}"

    llm = model.with_structured_output(ReviewClass)
    review: ReviewClass = llm.invoke([SystemMessage(content = system_prompt)])
    logger.debug("review_node has recieved the review")
    
    rendered = render_review(review)
    
    logger.debug("review_node ended")
    return {
        "task_id": state['task_id'],
        "worker_id": state['worker_id'],
        "messages": state['messages'] + [AIMessage(content=rendered)],
        "num_calls": state['num_calls'],
        "last_answer": state.get('last_answer'),
        "last_review": review,
        "shared_state": state['shared_state']
    }

def should_continue(state: WorkerAgentState) -> str:
    """Decide if we should continue or not"""

    logger.debug("should_continue started")

    num_calls = state.get("num_calls", 0)
    shared_state = state['shared_state']

    review = state['last_review']
    if review and review.satisfied:
        logger.debug("should_continue hit end")
        shared_state.update_task_status(state['task_id'], "Pending Review")
        return "end"
    else: 
        if num_calls >= MAX_CALLS:
            logger.warning("MAX_CALLS reached. Forcing end")
            return "end"
        logger.debug("should_continue hit continue")
        return "continue"

def test_case_node(state: WorkerAgentState) -> WorkerAgentState:
    """Node to generate test cases for the answer. Not implemented in this example."""

    logger.debug("test_case_node started")

    last_answer = state['last_answer'] if state['last_answer'] else ''
    last_answer_text = render_answer(last_answer)

    system_prompt = f"{TEST_CASE_PROMPT}\n\nCandidate Answer:\n{last_answer_text}"

    llm = model.with_structured_output(AnswerClass)
    test_cases = llm.invoke([SystemMessage(content = system_prompt)])
    logger.debug("test_case_node has recieved the test cases")

    rendered = render_answer(test_cases)

    shared_state = state['shared_state']
    shared_state.file_manager.create_file(shared_state.project_id, test_cases.file_name)
    lock_acquired = shared_state.file_manager.acquire_lock(shared_state.project_id, test_cases.file_name, state['worker_id'])
    if lock_acquired:
        file_path = shared_state.file_manager.abs_path(shared_state.project_id, test_cases.file_name)
        shared_state.file_editor.apply_patches(str(file_path), edits=test_cases.changes)
        logger.debug("applied changes successfully")
        shared_state.file_manager.release_lock(shared_state.project_id, test_cases.file_name, state['worker_id'])
    else:
        logger.debug("lock cannot be acquired")
        logger.debug("failed to apply changes")

    logger.debug("test_case_node_ended")
    return {
        "task_id": state['task_id'],
        "worker_id": state['worker_id'],
        "messages": state['messages'] + [AIMessage(content=rendered)],
        "last_answer": state.get('last_answer'),
        "last_review": state.get('last_review'),
        "num_calls": state['num_calls'] + 1,
        "shared_state": state['shared_state']
    }

# build the agent
agent_builder = StateGraph(WorkerAgentState)

# add nodes
agent_builder.add_node("answer_node", answer_node)
agent_builder.add_node("review_node", review_node)
agent_builder.add_node("test_case_node", test_case_node)

# add edges
agent_builder.add_edge(START, "answer_node")
agent_builder.add_edge("answer_node", "review_node")
agent_builder.add_conditional_edges("review_node", should_continue, {"continue": "answer_node", "end": test_case_node})
agent_builder.add_edge("test_case_node", END)

# compile
agent = agent_builder.compile()



