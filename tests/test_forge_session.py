import io
import json
from pathlib import Path
from unittest.mock import MagicMock

from rich.console import Console

from forge import RunResult
from forge_session import Session


def make_session(tmp_path, run_idea=None):
    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=False, color_system=None)
    run_idea = run_idea or MagicMock(return_value=RunResult(
        report={"final_status": "success"}, input_tokens=1000, output_tokens=200,
        cost_usd=0.002, log_path=Path("x.log"),
    ))
    return Session(console, run_idea, tmp_path / "projects"), out, run_idea


def test_blank_line_does_nothing(tmp_path):
    s, out, run = make_session(tmp_path)
    assert s.handle("   ") is True
    run.assert_not_called()


def test_plain_text_is_run_as_a_project_idea(tmp_path):
    s, _, run = make_session(tmp_path)
    assert s.handle("  build a todo cli  ") is True
    run.assert_called_once_with("build a todo cli")


def test_exit_and_quit_end_the_session(tmp_path):
    s, _, run = make_session(tmp_path)
    assert s.handle("/exit") is False
    assert s.handle("/QUIT") is False
    run.assert_not_called()


def test_help_lists_commands(tmp_path):
    s, out, _ = make_session(tmp_path)
    assert s.handle("/help") is True
    text = out.getvalue()
    for cmd in ("/help", "/projects", "/usage", "/exit"):
        assert cmd in text


def test_unknown_command_is_reported_and_not_run_as_idea(tmp_path):
    s, out, run = make_session(tmp_path)
    assert s.handle("/frobnicate now") is True
    assert "Unknown command" in out.getvalue() and "/frobnicate" in out.getvalue()
    run.assert_not_called()


def test_usage_accumulates_across_runs(tmp_path):
    s, out, _ = make_session(tmp_path)
    s.handle("one")
    s.handle("two")
    s.handle("/usage")
    text = out.getvalue()
    assert "2 run" in text and "2.0k" in text and "$0.0040" in text


def test_projects_lists_dirs_with_status_and_skips_telemetry(tmp_path):
    projects = tmp_path / "projects"
    (projects / "_telemetry").mkdir(parents=True)
    (projects / "todo-1").mkdir()
    (projects / "todo-1" / "project_report.json").write_text(json.dumps({"final_status": "success"}))
    (projects / "blog-2").mkdir()
    s, out, _ = make_session(tmp_path)
    s.handle("/projects")
    text = out.getvalue()
    assert "todo-1" in text and "success" in text and "blog-2" in text
    assert "_telemetry" not in text


def test_projects_when_none_exist(tmp_path):
    s, out, _ = make_session(tmp_path)
    s.handle("/projects")
    assert "No projects yet" in out.getvalue()


def test_projects_tolerates_corrupt_report(tmp_path):
    d = tmp_path / "projects" / "bad-1"
    d.mkdir(parents=True)
    (d / "project_report.json").write_text("{not json")
    s, out, _ = make_session(tmp_path)
    s.handle("/projects")
    assert "bad-1" in out.getvalue()


def test_failed_run_does_not_end_session_and_shows_error(tmp_path):
    s, out, _ = make_session(tmp_path, run_idea=MagicMock(side_effect=RuntimeError("kaboom [/x]")))
    assert s.handle("idea") is True
    assert "kaboom [/x]" in out.getvalue()


def test_ctrl_c_during_run_cancels_but_keeps_session(tmp_path):
    s, out, _ = make_session(tmp_path, run_idea=MagicMock(side_effect=KeyboardInterrupt))
    assert s.handle("idea") is True
    assert "Cancelled" in out.getvalue()


def test_loop_runs_until_exit_and_prints_banner(tmp_path):
    s, out, run = make_session(tmp_path)
    lines = iter(["/help", "build x", "/exit", "never reached"])
    assert s.loop(read=lambda: next(lines)) == 0
    run.assert_called_once_with("build x")
    assert "forge" in out.getvalue().lower() and "/exit" in out.getvalue()


def test_loop_ends_on_eof(tmp_path):
    s, _, run = make_session(tmp_path)

    def read():
        raise EOFError

    assert s.loop(read=read) == 0
    run.assert_not_called()


def test_ctrl_c_at_prompt_shows_hint_and_continues(tmp_path):
    s, out, _ = make_session(tmp_path)
    steps = iter([KeyboardInterrupt, "/exit"])

    def read():
        step = next(steps)
        if step is KeyboardInterrupt:
            raise KeyboardInterrupt
        return step

    assert s.loop(read=read) == 0
    assert "/exit" in out.getvalue()
