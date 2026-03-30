import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import WorkerTools


@pytest.fixture
def project_env(tmp_path, monkeypatch):
    project_id = "proj-1"
    base = tmp_path / "projects" / project_id
    base.mkdir(parents=True, exist_ok=True)

    project_info = base / "project_info.json"
    project_info.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        WorkerTools,
        "project_info_path",
        str(tmp_path / "projects" / "{project_id}" / "project_info.json"),
    )
    monkeypatch.setattr(
        WorkerTools,
        "project_file_path",
        str(tmp_path / "projects" / "{project_id}" / "{file_path}"),
    )

    monkeypatch.setattr(WorkerTools, "acquire_lock", lambda *a, **kw: None)
    monkeypatch.setattr(WorkerTools, "remove_lock", lambda *a, **kw: None)

    return {
        "tmp_path": tmp_path,
        "project_id": project_id,
        "base": base,
        "project_info_path": project_info,
    }

def _read_schema(project_env) -> dict:
    return json.loads(project_env["project_info_path"].read_text(encoding="utf-8"))


def _read_file_text(project_env, rel_path: str) -> str:
    return (
        project_env["base"] / rel_path
    ).read_text(encoding="utf-8")


def _seed_file(project_env, rel_path: str, content: str) -> None:
    """Write content to a project file, creating it in the schema too."""
    WorkerTools.create_file(
        project_env["project_id"], rel_path, "test file", "worker-1"
    )
    (project_env["base"] / rel_path).write_text(content, encoding="utf-8")


def _make_edit(operation, start_line, end_line=None, new_value="", method_changes=None):
    return SimpleNamespace(
        operation=operation,
        start_line=start_line,
        end_line=end_line,
        new_value=new_value,
        method_description_changes=method_changes or {},
    )

class TestGetProjectSchema:
    def test_returns_parsed_json(self, project_env):
        schema = {"a.py": {"description": "alpha"}}
        project_env["project_info_path"].write_text(json.dumps(schema), encoding="utf-8")

        result = WorkerTools.get_project_schema(project_env["project_id"])

        assert result == schema

    def test_raises_on_missing_file(self, project_env, monkeypatch):
        monkeypatch.setattr(
            WorkerTools,
            "project_info_path",
            str(project_env["tmp_path"] / "projects" / "{project_id}" / "nonexistent.json"),
        )

        with pytest.raises(FileNotFoundError):
            WorkerTools.get_project_schema(project_env["project_id"])

class TestUpdateFileSchema:
    def test_creates_new_entry(self, project_env):
        WorkerTools.update_file_schema(
            project_env["project_id"], "b.py", description="B", methods={}
        )

        schema = _read_schema(project_env)
        assert schema["b.py"]["description"] == "B"

    def test_merges_without_overwriting_other_keys(self, project_env):
        # Seed an existing entry
        project_env["project_info_path"].write_text(
            json.dumps({"b.py": {"description": "old", "created_by": "worker-0"}}),
            encoding="utf-8",
        )

        WorkerTools.update_file_schema(
            project_env["project_id"], "b.py", description="new"
        )

        schema = _read_schema(project_env)
        # updated
        assert schema["b.py"]["description"] == "new"
        # preserved
        assert schema["b.py"]["created_by"] == "worker-0"

    def test_multiple_keys_at_once(self, project_env):
        WorkerTools.update_file_schema(
            project_env["project_id"],
            "c.py",
            description="C",
            methods={"fn": "does something"},
            modified_by="worker-2",
        )

        schema = _read_schema(project_env)
        assert schema["c.py"]["methods"] == {"fn": "does something"}
        assert schema["c.py"]["modified_by"] == "worker-2"

class TestCreateFile:
    def test_creates_empty_file_on_disk(self, project_env):
        result = WorkerTools.create_file(
            project_env["project_id"], "src/a.py", "module A", "worker-1"
        )

        assert result is None
        assert (project_env["base"] / "src" / "a.py").read_text(encoding="utf-8") == ""

    def test_registers_schema_entry(self, project_env):
        WorkerTools.create_file(
            project_env["project_id"], "src/a.py", "module A", "worker-1"
        )

        schema = _read_schema(project_env)
        entry = schema["src/a.py"]
        assert entry["file_name"] == "a.py"
        assert entry["description"] == "module A"
        assert entry["created_by"] == "worker-1"
        assert entry["modified_by"] == "worker-1"
        assert entry["methods"] == {}
        assert "created_at" in entry
        assert "last_modified_at" in entry

    def test_creates_nested_directories(self, project_env):
        WorkerTools.create_file(
            project_env["project_id"], "a/b/c/deep.py", "deep", "worker-1"
        )

        assert (project_env["base"] / "a" / "b" / "c" / "deep.py").exists()

    def test_returns_error_on_duplicate(self, project_env):
        WorkerTools.create_file(
            project_env["project_id"], "dup.py", "first", "worker-1"
        )
        result = WorkerTools.create_file(
            project_env["project_id"], "dup.py", "second", "worker-1"
        )

        assert result is not None
        assert "already exists" in result

class TestReadFile:
    def test_returns_line_numbered_dict(self, project_env):
        (project_env["base"] / "r.txt").write_text("x\ny\nz", encoding="utf-8")

        result = WorkerTools.read_file(project_env["project_id"], ["r.txt"])

        assert result == {"r.txt": {1: "x", 2: "y", 3: "z"}}

    def test_reads_multiple_files(self, project_env):
        (project_env["base"] / "f1.txt").write_text("a", encoding="utf-8")
        (project_env["base"] / "f2.txt").write_text("b\nc", encoding="utf-8")

        result = WorkerTools.read_file(project_env["project_id"], ["f1.txt", "f2.txt"])

        assert result["f1.txt"] == {1: "a"}
        assert result["f2.txt"] == {1: "b", 2: "c"}

    def test_raises_file_not_found(self, project_env):
        with pytest.raises(FileNotFoundError, match="ghost.txt"):
            WorkerTools.read_file(project_env["project_id"], ["ghost.txt"])

    def test_empty_file_returns_empty_dict(self, project_env):
        (project_env["base"] / "empty.txt").write_text("", encoding="utf-8")

        result = WorkerTools.read_file(project_env["project_id"], ["empty.txt"])

        assert result["empty.txt"] == {}

class TestWriteFileUpdate:
    def test_single_line_update(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb\nc")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=2, new_value="B")],
        )

        assert _read_file_text(project_env, "t.txt") == "a\nB\nc"

    def test_multiline_update_same_line_count(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb\nc\nd")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=2, end_line=3, new_value="B\nC")],
        )

        assert _read_file_text(project_env, "t.txt") == "a\nB\nC\nd"

    def test_update_first_line(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=1, new_value="A")],
        )

        assert _read_file_text(project_env, "t.txt") == "A\nb"

    def test_update_last_line(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=2, new_value="B")],
        )

        assert _read_file_text(project_env, "t.txt") == "a\nB"

    def test_update_returns_error_when_start_line_out_of_bounds(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")

        result = WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=99, new_value="Z")],
        )

        assert result is not None
        assert "beyond the end" in result
        # File must be unchanged
        assert _read_file_text(project_env, "t.txt") == "a\nb"

    def test_update_returns_error_on_overflow_past_end_line(self, project_env):
        """new_value has more lines than the declared start_line–end_line range."""
        _seed_file(project_env, "t.txt", "a\nb\nc\nd")

        result = WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=2, end_line=3, new_value="X\nY\nZ")],
        )

        assert result is not None
        assert "covers only" in result
        assert _read_file_text(project_env, "t.txt") == "a\nb\nc\nd"

class TestWriteFileInsert:
    def test_insert_before_line(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb\nc")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("insert", start_line=2, new_value="X\nY")],
        )

        assert _read_file_text(project_env, "t.txt") == "a\nX\nY\nb\nc"

    def test_insert_before_first_line(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("insert", start_line=1, new_value="PRE")],
        )

        assert _read_file_text(project_env, "t.txt") == "PRE\na\nb"

    def test_insert_append_at_end(self, project_env):
        """start_line == total_lines + 1 should append."""
        _seed_file(project_env, "t.txt", "a\nb")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("insert", start_line=3, new_value="c")],
        )

        assert _read_file_text(project_env, "t.txt") == "a\nb\nc"

    def test_insert_returns_error_when_start_line_too_far(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")

        result = WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("insert", start_line=99, new_value="Z")],
        )

        assert result is not None
        assert "beyond the end" in result
        assert _read_file_text(project_env, "t.txt") == "a\nb"

class TestWriteFileDelete:
    def test_delete_middle_lines(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb\nc\nd")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("delete", start_line=2, end_line=3)],
        )

        assert _read_file_text(project_env, "t.txt") == "a\nd"

    def test_delete_single_line(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb\nc")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("delete", start_line=2, end_line=2)],
        )

        assert _read_file_text(project_env, "t.txt") == "a\nc"

    def test_delete_all_lines(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb\nc")

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("delete", start_line=1, end_line=3)],
        )

        assert _read_file_text(project_env, "t.txt") == ""

    def test_delete_returns_error_when_start_line_out_of_bounds(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")

        result = WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("delete", start_line=99, end_line=100)],
        )

        assert result is not None
        assert "beyond the end" in result
        assert _read_file_text(project_env, "t.txt") == "a\nb"

    def test_delete_returns_error_when_end_line_out_of_bounds(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")

        result = WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("delete", start_line=1, end_line=99)],
        )

        assert result is not None
        assert "beyond the end" in result
        assert _read_file_text(project_env, "t.txt") == "a\nb"

class TestWriteFileSequentialEdits:
    def test_multiple_edits_applied_in_order(self, project_env):
        """Delete line 2, then insert at line 2 — net effect is a replacement."""
        _seed_file(project_env, "t.txt", "a\nb\nc")

        edits = [
            _make_edit("delete", start_line=2, end_line=2),
            _make_edit("insert", start_line=2, new_value="B_new"),
        ]
        WorkerTools.write_file(project_env["project_id"], "worker-1", "t.txt", edits)

        assert _read_file_text(project_env, "t.txt") == "a\nB_new\nc"

    def test_stops_on_first_error(self, project_env):
        """Second edit is invalid; first edit must NOT have been committed."""
        _seed_file(project_env, "t.txt", "a\nb\nc")

        edits = [
            _make_edit("update", start_line=1, new_value="A"),   # valid
            _make_edit("update", start_line=99, new_value="Z"),  # invalid
        ]
        result = WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt", edits
        )

        assert result is not None
        # The file is only written to disk after ALL edits succeed, so it must
        # still have the original content.
        assert _read_file_text(project_env, "t.txt") == "a\nb\nc"

    def test_unknown_operation_returns_error(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")

        result = WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("teleport", start_line=1, new_value="x")],
        )

        assert result is not None
        assert "teleport" in result

class TestWriteFileSchemaTracking:
    def test_updates_modified_by_and_timestamp(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")
        before = time.time()

        WorkerTools.write_file(
            project_env["project_id"], "worker-2", "t.txt",
            [_make_edit("update", start_line=1, new_value="A")],
        )

        schema = _read_schema(project_env)
        assert schema["t.txt"]["modified_by"] == "worker-2"
        assert schema["t.txt"]["last_modified_at"] >= before

    def test_method_description_changes_are_merged(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb\nc")

        edits = [
            _make_edit(
                "update", start_line=1, new_value="A",
                method_changes={"fn_a": "returns A"},
            ),
            _make_edit(
                "update", start_line=2, new_value="B",
                method_changes={"fn_b": "returns B"},
            ),
        ]
        WorkerTools.write_file(project_env["project_id"], "worker-1", "t.txt", edits)

        schema = _read_schema(project_env)
        methods = schema["t.txt"]["methods"]
        assert methods["fn_a"] == "returns A"
        assert methods["fn_b"] == "returns B"

    def test_existing_methods_not_wiped_when_no_method_changes(self, project_env):
        _seed_file(project_env, "t.txt", "a\nb")
        # Manually add existing method entries to schema
        WorkerTools.update_file_schema(
            project_env["project_id"], "t.txt",
            methods={"existing_fn": "already tracked"},
        )

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=1, new_value="A")],
        )

        schema = _read_schema(project_env)
        # Must still be there
        assert schema["t.txt"]["methods"]["existing_fn"] == "already tracked"

    def test_method_changes_accumulate_on_top_of_existing(self, project_env):
        """New method entries merge with, rather than replace, existing ones."""
        _seed_file(project_env, "t.txt", "a\nb")
        WorkerTools.update_file_schema(
            project_env["project_id"], "t.txt",
            methods={"old_fn": "old description"},
        )

        WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=1, new_value="A",
                        method_changes={"new_fn": "new description"})],
        )

        schema = _read_schema(project_env)
        methods = schema["t.txt"]["methods"]
        assert methods["old_fn"] == "old description"
        assert methods["new_fn"] == "new description"

class TestWriteFileLocking:
    def test_returns_lock_message_and_leaves_file_unchanged(self, project_env, monkeypatch):
        monkeypatch.setattr(WorkerTools, "acquire_lock", lambda *a, **kw: "LOCKED by worker-9")

        _seed_file(project_env, "t.txt", "a\nb")
        result = WorkerTools.write_file(
            project_env["project_id"], "worker-1", "t.txt",
            [_make_edit("update", start_line=1, new_value="A")],
        )

        assert result == "LOCKED by worker-9"
        assert _read_file_text(project_env, "t.txt") == "a\nb"

    def test_lock_released_on_successful_write(self, project_env, monkeypatch):
        released = []
        monkeypatch._patches = []  # reset — use direct patch below

        _seed_file(project_env, "t.txt", "a")

        with patch.object(WorkerTools, "acquire_lock", return_value=None), \
             patch.object(WorkerTools, "remove_lock", side_effect=lambda *a, **kw: released.append(a)):
            WorkerTools.write_file(
                project_env["project_id"], "worker-1", "t.txt",
                [_make_edit("update", start_line=1, new_value="A")],
            )

        assert len(released) == 1, "remove_lock must be called exactly once on success"

    def test_lock_released_even_when_edit_fails(self, project_env):
        """try/finally must guarantee remove_lock is called regardless of errors."""
        released = []
        _seed_file(project_env, "t.txt", "a")

        with patch.object(WorkerTools, "acquire_lock", return_value=None), \
             patch.object(WorkerTools, "remove_lock", side_effect=lambda *a, **kw: released.append(a)):
            result = WorkerTools.write_file(
                project_env["project_id"], "worker-1", "t.txt",
                [_make_edit("update", start_line=999, new_value="Z")],  # bad edit
            )

        assert result is not None, "Expected an error string from the bad edit"
        assert len(released) == 1, "remove_lock must still be called after a failed edit"

    def test_lock_released_even_when_file_not_found(self, project_env, monkeypatch):
        """Lock must be released even if the target file doesn't exist on disk."""
        released = []

        with patch.object(WorkerTools, "acquire_lock", return_value=None), \
             patch.object(WorkerTools, "remove_lock", side_effect=lambda *a, **kw: released.append(a)):
            result = WorkerTools.write_file(
                project_env["project_id"], "worker-1", "does_not_exist.txt",
                [_make_edit("update", start_line=1, new_value="x")],
            )

        assert result is not None
        assert len(released) == 1