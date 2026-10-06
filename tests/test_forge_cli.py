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
