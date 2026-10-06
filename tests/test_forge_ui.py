import io

from rich.console import Console

from forge_ui import UIState, render_dashboard


def ev(kind, **kw):
    base = {
        "event_id": "e", "run_id": "r", "kind": kind, "project_id": "p",
        "agent": "", "node": "", "task_id": None, "worker_id": None,
        "duration_seconds": 0.0, "success": True, "error": None,
        "metadata": {}, "timestamp": "2026-10-06T12:04:11+00:00",
    }
    base.update(kw)
    return base


def llm(agent, in_t, out_t, cost, **kw):
    return ev("llm_call", agent=agent, metadata={
        "model": "claude-haiku-4-5", "input_tokens": in_t,
        "output_tokens": out_t, "cost_usd": cost, "price_known": True,
    }, **kw)


def test_llm_calls_accumulate_totals_per_agent_and_overall():
    s = UIState()
    s.apply(llm("architect", 100, 20, 0.001))
    s.apply(llm("worker", 300, 80, 0.003, worker_id="worker-a", task_id="a"))
    assert (s.input_tokens, s.output_tokens) == (400, 100)
    assert s.cost_usd == 0.004
    assert s.agents["worker"].input_tokens == 300 and s.agents["worker"].calls == 1
    assert s.workers["worker-a"].input_tokens == 300


def test_worker_lifecycle_node_tool_rework_score_done():
    s = UIState()
    w = dict(agent="worker", worker_id="worker-a", task_id="a")
    s.apply(ev("node_start", node="plan_task", **w))
    assert s.workers["worker-a"].node == "plan_task"
    s.apply(ev("tool_call", node="execute_task", metadata={"tool": "write_file", "file_path": "x.py"}, **w))
    assert s.workers["worker-a"].tool == "write_file"
    s.apply(ev("node_start", node="rework_task", **w))
    assert s.workers["worker-a"].rework == 1 and s.workers["worker-a"].tool == ""
    s.apply(ev("task_status", agent="teamlead", task_id="a",
               metadata={"status": "Closed", "dependency_list": [], "retry_count": 0, "review_score": 8, "desc": "d"}))
    assert s.workers["worker-a"].score == 8
    s.apply(ev("node_end", node="report_to_teamlead", **w))
    assert s.workers["worker-a"].done is True


def test_task_status_builds_task_table():
    s = UIState()
    s.apply(ev("task_status", agent="teamlead", task_id="b",
               metadata={"status": "Open", "dependency_list": ["a"], "retry_count": 1, "review_score": None, "desc": "second"}))
    t = s.tasks["b"]
    assert (t.status, t.deps, t.retries) == ("Open", ["a"], 1)


def test_failed_node_is_logged_and_log_is_bounded():
    s = UIState()
    line = s.apply(ev("node_end", agent="teamlead", node="finalize", success=False, error="ValueError('x')"))
    assert "finalize" in line and "ValueError" in line
    for i in range(30):
        s.apply(ev("node_start", agent="teamlead", node=f"n{i}"))
    assert len(s.log) == UIState.LOG_SIZE


def test_prompt_user_and_run_end_update_phase_and_log():
    s = UIState()
    assert "input" in s.apply(ev("prompt_user", agent="ui", metadata={"kind": "questions"})).lower()
    s.apply(ev("run_end"))
    assert s.phase == "done"


def render_text(state):
    console = Console(file=io.StringIO(), width=120, record=True, force_terminal=False)
    console.print(render_dashboard(state))
    return console.export_text()


def test_render_smoke_shows_tasks_workers_and_usage():
    s = UIState()
    s.project_id = "proj-1"
    s.set_phase("execution")
    s.apply(ev("task_status", agent="teamlead", task_id="auth-model",
               metadata={"status": "InProgress", "dependency_list": [], "retry_count": 0, "review_score": None, "desc": "model"}))
    s.apply(ev("node_start", agent="worker", node="execute_task", worker_id="worker-auth-model", task_id="auth-model"))
    s.apply(llm("worker", 4100, 1200, 0.0101, worker_id="worker-auth-model", task_id="auth-model"))
    text = render_text(s)
    for expected in ("proj-1", "auth-model", "InProgress", "worker-auth-model", "execute_task", "worker", "$0.0101"):
        assert expected in text, expected


def test_render_empty_state_does_not_crash():
    assert "forge" in render_text(UIState()).lower()


def test_render_treats_dynamic_text_literally_not_as_markup():
    s = UIState()
    s.apply(ev("task_status", agent="teamlead", task_id="[red]t[/oops]",
               metadata={"status": "Open", "dependency_list": ["[bold"], "retry_count": 0, "review_score": None, "desc": "[/]x"}))
    s.apply(ev("node_end", agent="teamlead", node="finalize", success=False, error="[/oops] [red"))
    text = render_text(s)  # must not raise MarkupError
    assert "[/oops]" in text and "[red]t" in text


import builtins

import pytest

from forge_events import EventBus, emit, get_bus, set_bus
from forge_ui import NO_ANSWER, ForgeUI


def make_ui(terminal=False):
    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=terminal, color_system=None)
    return ForgeUI(EventBus(), console), out


def feed_input(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))


QUESTIONS = [
    {"id": "q1", "question": "Web or CLI?", "type": "multiple_choice", "options": ["web", "cli"], "reason": "shape"},
    {"id": "q2", "question": "Persist data?", "type": "yes_no", "options": None, "reason": "storage"},
    {"id": "q3", "question": "Anything else?", "type": "text", "options": None, "reason": ""},
]


@pytest.mark.parametrize("terminal", [False, True])
def test_run_services_questions_from_pipeline_thread(monkeypatch, terminal):
    ui, out = make_ui(terminal)
    feed_input(monkeypatch, ["2", "y", "use sqlite"])
    result = ui.run(lambda: ui.request_input("questions", {"questions": QUESTIONS}))
    assert result.split("\n") == ["2", "yes", "use sqlite"]
    assert "Web or CLI?" in out.getvalue()


def test_blank_text_answer_becomes_placeholder_so_answers_do_not_shift(monkeypatch):
    ui, _ = make_ui()
    feed_input(monkeypatch, ["", "n"])
    questions = [QUESTIONS[2], QUESTIONS[1]]  # text first, then yes/no
    result = ui.run(lambda: ui.request_input("questions", {"questions": questions}))
    assert result.split("\n") == [NO_ANSWER, "no"]


def test_multiline_text_answer_is_collapsed_to_one_line(monkeypatch):
    ui, _ = make_ui()
    feed_input(monkeypatch, ["  a   b  "])
    result = ui.run(lambda: ui.request_input("questions", {"questions": [QUESTIONS[2]]}))
    assert result == "a b"


def test_decision_accept(monkeypatch):
    ui, out = make_ui()
    feed_input(monkeypatch, ["accept"])
    payload = {"understanding": "## What is clear\nA [bold CLI", "answers": [{"question": "q", "answer": "a"}]}
    assert ui.run(lambda: ui.request_input("decision", payload)) == "accept"
    assert "What is clear" in out.getvalue()


def test_decision_change_reasks_until_notes_given(monkeypatch):
    ui, _ = make_ui()
    feed_input(monkeypatch, ["change", "", "use sqlite"])
    payload = {"understanding": "u", "answers": []}
    assert ui.run(lambda: ui.request_input("decision", payload)) == "change: use sqlite"


def test_pipeline_exception_is_reraised_and_shown():
    ui, out = make_ui(terminal=True)

    def boom():
        raise RuntimeError("boom [/x]")

    with pytest.raises(RuntimeError):
        ui.run(boom)
    assert "boom [/x]" in out.getvalue()


def test_plain_mode_prints_event_lines_without_live():
    ui, out = make_ui(terminal=False)
    previous = get_bus()
    set_bus(ui._bus)
    try:
        ui.run(lambda: emit("node_start", agent="architect", node="analyse_input"))
    finally:
        set_bus(previous)
    assert "analyse_input" in out.getvalue()


def test_run_returns_pipeline_value_and_prints_final_dashboard():
    ui, out = make_ui(terminal=True)
    ui.state.project_id = "proj-9"
    assert ui.run(lambda: 42) == 42
    assert "proj-9" in out.getvalue()


def test_print_summary_handles_none_and_report():
    ui, out = make_ui()
    ui.print_summary(None)
    ui.print_summary({
        "project_id": "p", "final_status": "success", "total_tasks": 2, "completed_tasks": 2,
        "failed_tasks": 0, "blocked_tasks": 0, "all_touched_files": ["src/a.py"],
        "batch_review_scores": [8, 9], "summary": "ok [/x]", "completed_at": "t",
    })
    text = out.getvalue()
    assert "src/a.py" in text and "success" in text and "ok [/x]" in text


from forge_events import clear_cancel, is_cancelled
from forge_ui import _render_workers


def test_keyboard_interrupt_during_prompt_cancels_run_and_propagates(monkeypatch):
    ui, _ = make_ui(terminal=False)

    def raise_ki(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", raise_ki)
    clear_cancel()
    try:
        with pytest.raises(KeyboardInterrupt):
            ui.run(lambda: ui.request_input("questions", {"questions": [QUESTIONS[2]]}))
        assert is_cancelled()
    finally:
        clear_cancel()


def test_run_clears_a_stale_cancel_flag():
    from forge_events import cancel_run
    ui, _ = make_ui()
    cancel_run()
    assert ui.run(lambda: 1) == 1
    assert not is_cancelled()


def _workers_text(state):
    console = Console(file=io.StringIO(), width=100, record=True, force_terminal=False)
    console.print(_render_workers(state))
    return console.export_text()


def test_finished_workers_are_hidden_and_return_when_retried():
    s = UIState()
    w = dict(agent="worker", worker_id="worker-a", task_id="a")
    s.apply(ev("node_start", node="plan_task", **w))
    s.apply(ev("node_end", node="report_to_teamlead", **w))
    text = _workers_text(s)
    assert "worker-a" not in text and "1 finished" in text
    s.apply(ev("node_start", node="plan_task", **w))
    assert "worker-a" in _workers_text(s)
