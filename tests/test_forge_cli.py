from pathlib import Path
from unittest.mock import MagicMock, patch

import forge


def test_make_project_id_is_slug_plus_suffix_and_unique():
    a = forge.make_project_id("Build a Todo CLI!!")
    b = forge.make_project_id("Build a Todo CLI!!")
    assert a.startswith("build-a-todo-cli") and a != b
    assert all(c.isalnum() or c == "-" for c in a)


def test_make_project_id_handles_symbol_only_idea():
    assert forge.make_project_id("!!!").startswith("project-")


def test_bootstrap_project_creates_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(forge, "PROJECT_ROOT", tmp_path)
    d = forge.bootstrap_project("p1")
    assert (d / "src").is_dir() and (d / "tests").is_dir()
    assert (d / "locks.json").read_text() == "{}"
    assert (d / "project_info.json").read_text() == "{}"


def test_main_without_api_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(forge, "load_dotenv", lambda *a, **k: None)
    assert forge.main(["an idea"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_run_pipeline_stops_when_architect_is_rejected(tmp_path):
    ui = MagicMock()
    with patch.object(forge, "run_architect", return_value=None), \
         patch.object(forge, "run_teamlead") as teamlead:
        assert forge.run_pipeline("idea", "p", ui, tmp_path) is None
    teamlead.assert_not_called()


def test_run_pipeline_passes_spec_text_to_teamlead_and_saves_spec(tmp_path):
    ui = MagicMock()
    spec = {"project_id": "p", "project_name": "Demo", "spec_text": "# Spec", "created_at": "t"}
    report = {"final_status": "success"}
    with patch.object(forge, "run_architect", return_value=spec), \
         patch.object(forge, "run_teamlead", return_value=report) as teamlead:
        assert forge.run_pipeline("idea", "p", ui, tmp_path) == report
    teamlead.assert_called_once_with("p", "# Spec")
    assert (tmp_path / "architecture_spec.md").read_text() == "# Spec"
    assert ui.state.set_phase.call_args_list[0].args == ("architect",)
    assert ui.state.set_phase.call_args_list[1].args == ("execution",)


import io
import logging


def test_make_output_safe_replaces_unencodable_instead_of_raising():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    forge.make_output_safe(stream)
    stream.write("▶ ok")
    stream.flush()
    assert raw.getvalue().startswith(b"? ok")


def test_main_enters_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(forge, "load_dotenv", lambda *a, **k: None)
    forge.main(["x"])
    assert Path.cwd() == forge.PROJECT_ROOT


def test_main_logs_traceback_when_pipeline_fails(monkeypatch, tmp_path, fresh_bus):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(forge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(forge, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(forge, "run_pipeline", MagicMock(side_effect=RuntimeError("kaboom")))
    try:
        assert forge.main(["idea", "--project-id", "p1"]) == 1
        for h in logging.getLogger().handlers:
            h.flush()
        log = (tmp_path / "projects" / "p1" / "forge.log").read_text(encoding="utf-8")
        assert "kaboom" in log and "Traceback" in log
    finally:
        for h in list(logging.getLogger().handlers):
            h.close()
            logging.getLogger().removeHandler(h)


def _result(report):
    return forge.RunResult(report=report, input_tokens=0, output_tokens=0,
                           cost_usd=0.0, log_path=Path("x.log"))


def _ready(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(forge, "load_dotenv", lambda *a, **k: None)


def test_main_without_idea_starts_interactive_session(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    session = MagicMock()
    session.loop.return_value = 0
    with patch.object(forge, "Session", return_value=session):
        assert forge.main([]) == 0
    session.loop.assert_called_once()


def test_main_with_idea_runs_once_and_maps_exit_code(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    for report, expected in (({"final_status": "success"}, 0),
                             ({"final_status": "partial"}, 0),
                             ({"final_status": "failed"}, 1),
                             (None, 1)):
        with patch.object(forge, "execute_run", return_value=_result(report)) as run:
            assert forge.main(["an idea"]) == expected
        assert run.call_args.args[1] == "an idea"


def test_main_with_idea_maps_ctrl_c_and_errors(monkeypatch, tmp_path):
    _ready(monkeypatch, tmp_path)
    with patch.object(forge, "execute_run", side_effect=KeyboardInterrupt):
        assert forge.main(["an idea"]) == 130
    with patch.object(forge, "execute_run", side_effect=RuntimeError("boom")):
        assert forge.main(["an idea"]) == 1
