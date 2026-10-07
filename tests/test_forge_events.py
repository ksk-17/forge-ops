import json
import threading

import pytest

import forge_events as fe
from forge_events import EventBus, emit, estimate_cost, reset_context, set_context


def test_publish_delivers_in_order(fresh_bus):
    seen = []
    fresh_bus.subscribe(seen.append)
    for i in range(3):
        emit("node_start", node=f"n{i}")
    assert [e["node"] for e in seen] == ["n0", "n1", "n2"]


def test_history_is_bounded():
    bus = EventBus(history_size=3)
    for i in range(5):
        bus.publish({"i": i})
    assert [e["i"] for e in bus.history()] == [2, 3, 4]


def test_replay_gives_late_subscriber_past_events(fresh_bus):
    emit("node_start", node="a")
    seen = []
    fresh_bus.subscribe(seen.append, replay=True)
    assert [e["node"] for e in seen] == ["a"]


def test_failing_subscriber_does_not_break_publish_or_others(fresh_bus):
    good = []

    def bad(_event):
        raise RuntimeError("boom")

    fresh_bus.subscribe(bad)
    fresh_bus.subscribe(good.append)
    emit("node_start", node="a")  # must not raise
    assert len(good) == 1


def test_publish_is_thread_safe(fresh_bus):
    seen = []
    fresh_bus.subscribe(seen.append)

    def work():
        for _ in range(100):
            emit("llm_call")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 800
    assert len({e["event_id"] for e in seen}) == 800


def test_emit_inherits_context_and_context_resets(fresh_bus):
    token = set_context(agent="worker", node="plan_task", worker_id="worker-a", task_id="a")
    try:
        e = emit("llm_call")
    finally:
        reset_context(token)
    assert (e["agent"], e["node"], e["worker_id"], e["task_id"]) == (
        "worker", "plan_task", "worker-a", "a",
    )
    assert emit("llm_call")["worker_id"] is None


def test_emit_stamps_run_and_project(fresh_bus):
    fresh_bus.project_id = "proj-1"
    e = emit("run_end")
    assert e["run_id"] == fresh_bus.run_id
    assert e["project_id"] == "proj-1"
    assert e["timestamp"].endswith("+00:00") or e["timestamp"].endswith("Z")


def test_estimate_cost_matches_model_prefix():
    assert estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000) == (6.0, True)


def test_estimate_cost_unknown_model_is_zero_and_flagged():
    assert estimate_cost("some-future-model", 1000, 1000) == (0.0, False)


def test_telemetry_subscriber_writes_agent_event_shape(tmp_path, monkeypatch, fresh_bus):
    import forge_memory

    monkeypatch.setattr(forge_memory, "_TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(forge_memory, "_EVENTS_FILE", tmp_path / "agent_events.jsonl")
    fresh_bus.subscribe(fe.telemetry_subscriber)

    emit("llm_call", agent="worker", node="plan_task", metadata={"input_tokens": 5})

    rec = json.loads((tmp_path / "agent_events.jsonl").read_text().splitlines()[0])
    assert rec["agent"] == "worker"
    assert rec["metadata"]["kind"] == "llm_call"
    assert rec["metadata"]["input_tokens"] == 5
    assert {
        "event_id", "run_id", "project_id", "agent", "node", "task_id",
        "duration_seconds", "success", "error", "metadata", "timestamp",
    } <= set(rec)


import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from langgraph.errors import GraphInterrupt

from forge_events import track, traced_node


def _resp(i=10, o=5):
    return SimpleNamespace(
        content=[SimpleNamespace(text="hi")],
        usage=SimpleNamespace(input_tokens=i, output_tokens=o),
    )


def test_track_forwards_call_and_returns_response(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp()
    result = track(client, "worker").messages.create(
        model="claude-haiku-4-5", max_tokens=5, messages=[]
    )
    assert result is client.messages.create.return_value
    client.messages.create.assert_called_once_with(
        model="claude-haiku-4-5", max_tokens=5, messages=[]
    )


def test_track_emits_tokens_and_cost(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp(10, 5)
    track(client, "worker").messages.create(model="claude-haiku-4-5")
    e = fresh_bus.history()[-1]
    assert e["kind"] == "llm_call" and e["agent"] == "worker"
    md = e["metadata"]
    assert (md["input_tokens"], md["output_tokens"]) == (10, 5)
    assert md["cost_usd"] == pytest.approx((10 * 1.0 + 5 * 5.0) / 1_000_000)
    assert md["price_known"] is True and e["duration_seconds"] >= 0


@pytest.mark.parametrize("response", [
    SimpleNamespace(content=[]),                      # no usage attribute
    SimpleNamespace(content=[], usage=None),          # usage is None
    MagicMock(),                                      # usage is a MagicMock
])
def test_track_tolerates_missing_or_mock_usage(fresh_bus, response):
    client = MagicMock()
    client.messages.create.return_value = response
    out = track(client, "a").messages.create(model="claude-haiku-4-5")
    assert out is response
    md = fresh_bus.history()[-1]["metadata"]
    assert (md["input_tokens"], md["output_tokens"], md["cost_usd"]) == (0, 0, 0.0)


def test_track_records_failure_and_reraises(fresh_bus):
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("api down")
    with pytest.raises(RuntimeError):
        track(client, "a").messages.create(model="claude-haiku-4-5")
    e = fresh_bus.history()[-1]
    assert e["kind"] == "llm_call" and e["success"] is False and "api down" in e["error"]


def test_track_proxies_other_attributes_and_does_not_double_wrap(fresh_bus):
    client = MagicMock()
    tracked = track(client, "a")
    assert tracked.models is client.models
    assert track(tracked, "a") is tracked


def test_traced_node_emits_start_end_and_returns_value(fresh_bus):
    @traced_node("worker")
    def plan_task(state):
        return {"plan": "x"}

    state = {"worker_id": "worker-a", "task": SimpleNamespace(task_id="a")}
    assert plan_task(state) == {"plan": "x"}
    rows = [(e["kind"], e["node"], e["worker_id"], e["task_id"]) for e in fresh_bus.history()]
    assert rows == [
        ("node_start", "plan_task", "worker-a", "a"),
        ("node_end", "plan_task", "worker-a", "a"),
    ]
    assert fresh_bus.history()[-1]["success"] is True


def test_traced_node_attributes_llm_calls_to_node_and_worker(fresh_bus):
    client = MagicMock()
    client.messages.create.return_value = _resp()

    @traced_node("worker")
    def execute_task(state):
        track(client, "worker").messages.create(model="claude-haiku-4-5")
        return {}

    execute_task({"worker_id": "worker-a", "task": SimpleNamespace(task_id="a")})
    call = next(e for e in fresh_bus.history() if e["kind"] == "llm_call")
    assert (call["node"], call["worker_id"], call["task_id"]) == ("execute_task", "worker-a", "a")


def test_traced_node_records_failure_and_reraises(fresh_bus):
    @traced_node("teamlead")
    def boom(state):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom({})
    end = fresh_bus.history()[-1]
    assert end["kind"] == "node_end" and end["success"] is False and "nope" in end["error"]


def test_traced_node_treats_graph_interrupt_as_pause(fresh_bus):
    @traced_node("architect")
    def ask_user(state):
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        ask_user({})
    end = fresh_bus.history()[-1]
    assert end["success"] is True and end["metadata"]["paused"] is True


def test_traced_node_preserves_signature_and_resets_context(fresh_bus):
    @traced_node("worker")
    def plan_task(state):
        return {}

    assert plan_task.__name__ == "plan_task"
    assert list(inspect.signature(plan_task).parameters) == ["state"]
    plan_task({"worker_id": "worker-a"})
    assert emit("llm_call")["worker_id"] is None


from forge_events import RunCancelled, cancel_run, clear_cancel, is_cancelled


@pytest.fixture
def clean_cancel():
    clear_cancel()
    yield
    clear_cancel()


def test_traced_node_refuses_to_start_after_cancel(fresh_bus, clean_cancel):
    called = []

    @traced_node("worker")
    def plan_task(state):
        called.append(1)

    cancel_run()
    with pytest.raises(RunCancelled):
        plan_task({})
    assert called == [] and is_cancelled()


def test_track_refuses_llm_call_after_cancel(fresh_bus, clean_cancel):
    client = MagicMock()
    cancel_run()
    with pytest.raises(RunCancelled):
        track(client, "a").messages.create(model="claude-haiku-4-5")
    client.messages.create.assert_not_called()
