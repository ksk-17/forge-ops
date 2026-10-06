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
