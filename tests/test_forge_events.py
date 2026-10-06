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
