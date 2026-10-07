import json
from unittest.mock import patch

import ArchitectAgent as aa


def _fake_llm(system, user, max_tokens=2048):
    if "rough idea" in system:
        return "## What is clear\nA CLI.\n## Initial project name suggestion\nDemo"
    if "clarifying only the most critical" in system:
        return json.dumps([{
            "id": "q1", "question": "Web or CLI?", "type": "multiple_choice",
            "options": ["web", "cli"], "reason": "shape",
        }])
    if "updating your understanding" in system:
        return "## What is clear\nA CLI.\n## Initial project name suggestion\nDemo"
    if "deciding whether requirements" in system:
        return "SCORE: 9\nVERDICT: complete"
    return "# Demo — Architecture Specification"


def _run(ask_decision):
    questions_seen = []

    def ask_questions(questions):
        questions_seen.append(questions)
        return "2"

    with patch.object(aa, "_llm", side_effect=_fake_llm), \
         patch.object(aa, "get_architect_context", return_value=""), \
         patch.object(aa, "record_architect_run"):
        spec = aa.run_architect("proj-1", "a todo cli", ask_questions, ask_decision)
    return spec, questions_seen


def test_run_architect_accept_returns_spec_and_passes_questions():
    decisions = []

    def ask_decision(values):
        decisions.append(values)
        return "accept"

    spec, questions_seen = _run(ask_decision)
    assert spec["spec_text"].startswith("# Demo")
    assert len(questions_seen) == 1 and questions_seen[0][0]["id"] == "q1"
    assert "A CLI" in decisions[0]["understanding"]
    assert decisions[0]["answers"][0]["answer"] == "cli"  # "2" normalised to option


def test_run_architect_reject_returns_none():
    spec, _ = _run(lambda values: "reject")
    assert spec is None


def test_run_architect_change_loops_back_then_accepts():
    replies = iter(["change: use sqlite", "accept"])
    spec, _ = _run(lambda values: next(replies))
    assert spec is not None
