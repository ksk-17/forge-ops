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


import WorkerAgent as wa

WORKER_NODES = [
    "plan_task", "execute_task", "review_task", "rework_task",
    "generate_tests", "report_to_teamlead",
]


def test_every_worker_node_is_traced():
    for name in WORKER_NODES:
        assert hasattr(getattr(wa, name), "__wrapped__"), name


def test_tool_loop_emits_tool_call_events(fresh_bus):
    tool_block = SimpleNamespace(
        type="tool_use", id="t1", name="write_file",
        input={"file_path": "src/a.py", "project_id": "p"},
    )
    first = SimpleNamespace(content=[tool_block], stop_reason="tool_use")
    done = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done")], stop_reason="end_turn",
    )
    client = MagicMock()
    client.messages.create.side_effect = [first, done]
    with patch.object(wa, "_dispatch_tool", return_value="{}"):
        wa._run_tool_loop(client, "sys", [{"role": "user", "content": "go"}])
    tools = [e for e in fresh_bus.history() if e["kind"] == "tool_call"]
    assert len(tools) == 1
    assert tools[0]["metadata"] == {"tool": "write_file", "file_path": "src/a.py"}


def test_tool_loop_tolerates_non_dict_tool_input(fresh_bus):
    tool_block = SimpleNamespace(type="tool_use", id="t1", name="read_file", input=None)
    first = SimpleNamespace(content=[tool_block], stop_reason="tool_use")
    done = SimpleNamespace(content=[], stop_reason="end_turn")
    client = MagicMock()
    client.messages.create.side_effect = [first, done]
    with patch.object(wa, "_dispatch_tool", return_value="{}"):
        wa._run_tool_loop(client, "sys", [])
    tools = [e for e in fresh_bus.history() if e["kind"] == "tool_call"]
    assert tools[0]["metadata"]["file_path"] is None
