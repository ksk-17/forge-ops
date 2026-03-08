import json
from types import SimpleNamespace

import pytest

import WorkerTools


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    """
    Creates an isolated 'projects/<project_id>/' folder under tmp_path
    and monkeypatches WorkerTools' path templates to point into tmp_path.

    Also stubs out acquire_lock/remove_lock so tests don't depend on lock implementation.
    """
    project_id = "proj-1"
    base = tmp_path / "projects" / project_id
    base.mkdir(parents=True, exist_ok=True)

    # project_info.json must exist for schema reads/writes
    project_info = base / "project_info.json"
    project_info.write_text("{}", encoding="utf-8")

    # Patch the path templates to use tmp_path
    monkeypatch.setattr(
        WorkerTools,
        "project_info_path",
        str(tmp_path / "projects" / "{project_id}" / "project_info.json"),
        raising=True,
    )
    monkeypatch.setattr(
        WorkerTools,
        "project_file_path",
        str(tmp_path / "projects" / "{project_id}" / "{file_path}"),
        raising=True,
    )

    # Make locking a no-op for most tests (focus on file operations)
    monkeypatch.setattr(WorkerTools, "acquire_lock", lambda *args, **kwargs: None, raising=True)
    monkeypatch.setattr(WorkerTools, "remove_lock", lambda *args, **kwargs: None, raising=True)

    return {
        "tmp_path": tmp_path,
        "project_id": project_id,
        "base": base,
        "project_info_path": project_info,
    }


def _read_project_schema(project_env):
    """Helper: read schema json directly from disk."""
    with open(project_env["project_info_path"], "r", encoding="utf-8") as f:
        return json.load(f)


def _read_file_text(project_env, rel_file_path: str) -> str:
    abs_path = project_env["tmp_path"] / "projects" / project_env["project_id"] / rel_file_path
    return abs_path.read_text(encoding="utf-8")


def _make_edit(operation, start_line, end_line=None, new_value=""):
    """
    WorkerTools.write_file uses attribute access (edit.operation, edit.start_line, ...),
    so SimpleNamespace works even if your Edit model changes.
    """
    return SimpleNamespace(
        operation=operation,
        start_line=start_line,
        end_line=end_line,
        new_value=new_value,
        method_description_changes={},
    )


def test_get_project_schema_reads_json(project_env):
    schema = {"a.py": {"description": "test"}}
    project_env["project_info_path"].write_text(json.dumps(schema), encoding="utf-8")

    loaded = WorkerTools.get_project_schema(project_env["project_id"])
    assert loaded == schema


def test_update_file_schema_merges_fields(project_env):
    WorkerTools.update_file_schema(
        project_env["project_id"],
        "a.py",
        description="file A",
        methods={"m1": "desc"},
    )

    schema = _read_project_schema(project_env)
    assert "a.py" in schema
    assert schema["a.py"]["description"] == "file A"
    assert schema["a.py"]["methods"] == {"m1": "desc"}


def test_create_file_creates_empty_file_and_schema(project_env):
    WorkerTools.create_file(
        project_env["project_id"],
        "src/a.py",
        file_description="A file",
        worker_id="worker-1",
    )

    abs_file = project_env["base"] / "src" / "a.py"
    assert abs_file.exists()
    assert abs_file.read_text(encoding="utf-8") == ""

    schema = _read_project_schema(project_env)
    assert "src/a.py" in schema
    assert schema["src/a.py"]["file_name"] == "a.py"
    assert schema["src/a.py"]["description"] == "A file"
    assert schema["src/a.py"]["created_by"] == "worker-1"


def test_read_file_returns_line_numbered_dict(project_env):
    # Create a file to read
    abs_file = project_env["base"] / "x.txt"
    abs_file.write_text("a\nb\nc\n", encoding="utf-8")

    out = WorkerTools.read_file(project_env["project_id"], ["x.txt"])
    assert "x.txt" in out
    assert out["x.txt"] == {1: "a", 2: "b", 3: "c"}


def test_write_file_update(project_env):
    WorkerTools.create_file(project_env["project_id"], "t.txt", "test", "worker-1")
    (project_env["base"] / "t.txt").write_text("a\nb\nc", encoding="utf-8")

    edits = [_make_edit("update", start_line=2, new_value="B")]
    WorkerTools.write_file(project_env["project_id"], "worker-1", "t.txt", edits)

    assert _read_file_text(project_env, "t.txt") == "a\nB\nc"

    schema = _read_project_schema(project_env)
    assert schema["t.txt"]["modified_by"] == "worker-1"
    assert "last_modified_at" in schema["t.txt"]


def test_write_file_insert(project_env):
    WorkerTools.create_file(project_env["project_id"], "t.txt", "test", "worker-1")
    (project_env["base"] / "t.txt").write_text("a\nb\nc", encoding="utf-8")

    edits = [_make_edit("insert", start_line=2, new_value="X\nY")]
    WorkerTools.write_file(project_env["project_id"], "worker-1", "t.txt", edits)

    # Expect insertion before original line 2 ("b")
    assert _read_file_text(project_env, "t.txt") == "a\nX\nY\nb\nc"


def test_write_file_delete(project_env):
    WorkerTools.create_file(project_env["project_id"], "t.txt", "test", "worker-1")
    (project_env["base"] / "t.txt").write_text("a\nb\nc\nd", encoding="utf-8")

    edits = [_make_edit("delete", start_line=2, end_line=3)]
    WorkerTools.write_file(project_env["project_id"], "worker-1", "t.txt", edits)

    assert _read_file_text(project_env, "t.txt") == "a\nd"


def test_write_file_returns_lock_message_on_conflict(project_env, monkeypatch):
    # Simulate lock conflict
    monkeypatch.setattr(
        WorkerTools,
        "acquire_lock",
        lambda *args, **kwargs: "LOCKED",
        raising=True,
    )

    WorkerTools.create_file(project_env["project_id"], "t.txt", "test", "worker-1")
    (project_env["base"] / "t.txt").write_text("a\nb", encoding="utf-8")

    edits = [_make_edit("update", start_line=1, new_value="A")]
    result = WorkerTools.write_file(project_env["project_id"], "worker-1", "t.txt", edits)

    assert result == "LOCKED"
    # Ensure file is unchanged when locked
    assert _read_file_text(project_env, "t.txt") == "a\nb"