from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ArchitectAgent as aa


def _resp(text="x", i=10, o=5):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=i, output_tokens=o),
    )


ARCHITECT_NODES = [
    "analyse_input", "generate_questions", "ask_user", "incorporate_answers",
    "check_completeness", "present_summary", "handle_user_review",
    "produce_architecture_spec",
]


def test_architect_llm_emits_llm_call_with_tokens(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp("hello", 7, 3)
    with patch("ArchitectAgent.Anthropic", return_value=client):
        assert aa._llm("sys", "user") == "hello"
    call = next(e for e in fresh_bus.history() if e["kind"] == "llm_call")
    assert call["agent"] == "architect"
    assert (call["metadata"]["input_tokens"], call["metadata"]["output_tokens"]) == (7, 3)


def test_architect_node_is_traced_and_llm_call_attributed(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp("SCORE: 9\nVERDICT: ok")
    state = {"project_id": "p", "understanding": "u", "clarification_round": 0}
    with patch("ArchitectAgent.Anthropic", return_value=client):
        assert aa.check_completeness(state) == {"completeness_score": 9}
    kinds = [(e["kind"], e["node"]) for e in fresh_bus.history()]
    assert kinds == [
        ("node_start", "check_completeness"),
        ("llm_call", "check_completeness"),
        ("node_end", "check_completeness"),
    ]


def test_every_architect_node_is_traced():
    for name in ARCHITECT_NODES:
        assert hasattr(getattr(aa, name), "__wrapped__"), name
