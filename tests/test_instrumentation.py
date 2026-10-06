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


import json

import TeamLeadAgent as tl

TEAMLEAD_NODES = [
    "decompose_tasks", "schedule_iteration", "dispatch_workers",
    "collect_reports", "review_batch", "handle_batch_review", "finalize",
]


def _rec(tid, deps=(), status="Open"):
    return {
        "task_id": tid, "project_id": "p", "desc": f"do {tid}\nmore", "status": status,
        "worker_id": "", "dependency_list": list(deps), "retry_count": 0,
        "report": None, "review_notes": "",
    }


def test_every_teamlead_node_is_traced():
    for name in TEAMLEAD_NODES:
        assert hasattr(getattr(tl, name), "__wrapped__"), name


def test_update_task_status_emits_task_status(fresh_bus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tl._save_registry("p", {"a": _rec("a")})
    tl._update_task_status("p", "a", "Closed", report={"review_score": 9})
    e = next(x for x in fresh_bus.history() if x["kind"] == "task_status")
    assert e["task_id"] == "a"
    assert e["metadata"]["status"] == "Closed"
    assert e["metadata"]["review_score"] == 9
    assert e["metadata"]["desc"] == "do a"


def test_decompose_emits_open_status_for_each_task(fresh_bus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tasks = [
        {"task_id": "a", "desc": "first", "dependency_list": []},
        {"task_id": "b", "desc": "second", "dependency_list": ["a"]},
    ]
    client = MagicMock()
    client.messages.create.return_value = _resp(json.dumps(tasks))
    state = {"project_id": "p", "architecture_spec": "spec text"}
    with patch("TeamLeadAgent.Anthropic", return_value=client), \
         patch("TeamLeadAgent.get_teamlead_context", return_value=""):
        tl.decompose_tasks(state)
    rows = {
        e["task_id"]: e["metadata"]
        for e in fresh_bus.history() if e["kind"] == "task_status"
    }
    assert set(rows) == {"a", "b"}
    assert rows["b"]["dependency_list"] == ["a"] and rows["b"]["status"] == "Open"
