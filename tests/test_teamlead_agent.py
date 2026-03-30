from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock, patch, call

import pytest

import TeamLeadAgent as tla
from TeamLeadAgent import (
    MAX_RETRY_CYCLES,
    INTEGRATION_PASS_THRESHOLD,
    BatchReview,
    ProjectReport,
    TaskRecord,
    TeamLeadState,
    _counts,
    _load_registry,
    _parse_list_section,
    _parse_score,
    _parse_summary,
    _ready_tasks,
    _save_registry,
    _tasks_path,
    _update_task_status,
    build_teamlead_graph,
    collect_reports,
    decompose_tasks,
    dispatch_workers,
    finalize,
    handle_batch_review,
    review_batch,
    route_after_review,
    route_after_schedule,
    run_teamlead,
    schedule_iteration,
)

PROJECT_ID = "test-proj"

@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """
    Redirect TASKS_FILE and PROJECT_REPORT_FILE into tmp_path and create the
    project directory structure. Returns the project base directory.
    """
    base = tmp_path / "projects" / PROJECT_ID
    base.mkdir(parents=True)

    monkeypatch.setattr(
        tla, "TASKS_FILE",
        str(tmp_path / "projects" / "{project_id}" / "tasks.json"),
    )
    monkeypatch.setattr(
        tla, "PROJECT_REPORT_FILE",
        str(tmp_path / "projects" / "{project_id}" / "project_report.json"),
    )
    return base


def _make_record(
    task_id: str,
    status: str = "Open",
    deps: List[str] = None,
    retry_count: int = 0,
    review_notes: str = "",
    report: Dict = None,
) -> TaskRecord:
    return {
        "task_id": task_id,
        "project_id": PROJECT_ID,
        "desc": f"Implement {task_id}",
        "status": status,
        "worker_id": "",
        "dependency_list": deps or [],
        "retry_count": retry_count,
        "report": report,
        "review_notes": review_notes,
    }


def _seed_registry(tmp_project: Path, records: List[TaskRecord]) -> None:
    tasks_file = tmp_project / "tasks.json"
    tasks_file.write_text(json.dumps(records), encoding="utf-8")


def _load_registry_raw(tmp_project: Path) -> List[Dict]:
    return json.loads((tmp_project / "tasks.json").read_text(encoding="utf-8"))


def _make_worker_report(
    task_id: str,
    status: str = "completed",
    score: int = 9,
    touched: List[str] = None,
    blockers: List[str] = None,
) -> Dict:
    touched_files = [f"src/{task_id}.py"] if touched is None else touched
    return {
        "task_id": task_id,
        "worker_id": f"worker-{task_id}",
        "status": status,
        "summary": f"Done with {task_id}",
        "touched_files": touched_files,
        "test_file_path": f"tests/test_{task_id}.py",
        "review_score": score,
        "review_feedback": f"SCORE: {score}\nVERDICT: PASS",
        "blockers": [] if blockers is None else blockers,
        "suggestions": [],
        "execution_notes": "All good.",
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }


def _text_response(text: str) -> SimpleNamespace:
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], stop_reason="end_turn")


def _base_state(**overrides) -> TeamLeadState:
    state: TeamLeadState = {
        "project_id": PROJECT_ID,
        "architecture_spec": "Build a simple REST API.",
        "all_task_ids": [],
        "current_batch": [],
        "iteration": 0,
        "worker_reports": {},
        "all_touched_files": [],
        "batch_reviews": [],
        "latest_review": None,
        "retry_queue": [],
        "messages": [],
        "project_report": None,
    }
    state.update(overrides)
    return state

class TestTasksPath:
    def test_returns_correct_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tla, "TASKS_FILE", str(tmp_path / "projects" / "{project_id}" / "tasks.json"))
        result = _tasks_path("my-proj")
        assert result == tmp_path / "projects" / "my-proj" / "tasks.json"


class TestLoadRegistry:
    def test_returns_empty_dict_when_file_missing(self, tmp_project):
        result = _load_registry(PROJECT_ID)
        assert result == {}

    def test_loads_and_indexes_by_task_id(self, tmp_project):
        records = [_make_record("t1"), _make_record("t2")]
        _seed_registry(tmp_project, records)

        result = _load_registry(PROJECT_ID)

        assert set(result.keys()) == {"t1", "t2"}
        assert result["t1"]["status"] == "Open"


class TestSaveRegistry:
    def test_round_trip(self, tmp_project):
        registry = {
            "t1": _make_record("t1", status="Closed"),
            "t2": _make_record("t2", status="InProgress"),
        }
        _save_registry(PROJECT_ID, registry)
        loaded = _load_registry(PROJECT_ID)

        assert loaded["t1"]["status"] == "Closed"
        assert loaded["t2"]["status"] == "InProgress"

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        deep = tmp_path / "deep" / "nested" / "projects" / "{project_id}" / "tasks.json"
        monkeypatch.setattr(tla, "TASKS_FILE", str(deep))

        _save_registry("new-proj", {"t1": _make_record("t1")})

        assert (tmp_path / "deep" / "nested" / "projects" / "new-proj" / "tasks.json").exists()


class TestUpdateTaskStatus:
    def test_updates_status(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="Open")])
        _update_task_status(PROJECT_ID, "t1", "Closed")
        registry = _load_registry(PROJECT_ID)
        assert registry["t1"]["status"] == "Closed"

    def test_sets_report(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])
        report = {"summary": "done"}
        _update_task_status(PROJECT_ID, "t1", "Closed", report=report)
        assert _load_registry(PROJECT_ID)["t1"]["report"] == report

    def test_sets_review_notes(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])
        _update_task_status(PROJECT_ID, "t1", "ReworkRequired", review_notes="Fix imports")
        assert _load_registry(PROJECT_ID)["t1"]["review_notes"] == "Fix imports"

    def test_sets_retry_count(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", retry_count=0)])
        _update_task_status(PROJECT_ID, "t1", "Open", retry_count=1)
        assert _load_registry(PROJECT_ID)["t1"]["retry_count"] == 1

    def test_does_not_overwrite_omitted_optional_fields(self, tmp_project):
        """Passing review_notes='' must NOT clear an existing note."""
        _seed_registry(tmp_project, [_make_record("t1", review_notes="existing note")])
        _update_task_status(PROJECT_ID, "t1", "Open")  # no review_notes kwarg
        assert _load_registry(PROJECT_ID)["t1"]["review_notes"] == "existing note"


class TestReadyTasks:
    def test_no_deps_task_is_immediately_ready(self):
        registry = {"t1": _make_record("t1")}
        assert [r["task_id"] for r in _ready_tasks(registry)] == ["t1"]

    def test_task_with_closed_dep_is_ready(self):
        registry = {
            "t1": _make_record("t1", status="Closed"),
            "t2": _make_record("t2", deps=["t1"]),
        }
        ready_ids = [r["task_id"] for r in _ready_tasks(registry)]
        assert ready_ids == ["t2"]

    def test_task_with_open_dep_is_not_ready(self):
        # t1 is InProgress (not Closed), so t2 must not be ready.
        # t1 is also not Open so it won't appear in the ready list itself.
        registry = {
            "t1": _make_record("t1", status="InProgress"),
            "t2": _make_record("t2", deps=["t1"]),
        }
        assert _ready_tasks(registry) == []

    def test_partial_deps_blocks_task(self):
        # t1 is Closed (satisfied), t2 is InProgress (not Closed yet).
        # t3 depends on both — it must not be ready because t2 is not Closed.
        # t2 is InProgress so it also won't appear in the ready list.
        registry = {
            "t1": _make_record("t1", status="Closed"),
            "t2": _make_record("t2", status="InProgress"),
            "t3": _make_record("t3", deps=["t1", "t2"]),
        }
        assert _ready_tasks(registry) == []

    def test_already_closed_tasks_excluded(self):
        """Closed tasks must not appear in ready batch even with no deps."""
        registry = {"t1": _make_record("t1", status="Closed")}
        assert _ready_tasks(registry) == []

    def test_mixed_statuses_only_returns_open_ready(self):
        registry = {
            "t1": _make_record("t1", status="Closed"),
            "t2": _make_record("t2", status="Open", deps=["t1"]),
            "t3": _make_record("t3", status="InProgress"),
            "t4": _make_record("t4", status="ReworkRequired"),
        }
        ready_ids = [r["task_id"] for r in _ready_tasks(registry)]
        assert ready_ids == ["t2"]

    def test_empty_registry_returns_empty(self):
        assert _ready_tasks({}) == []


class TestCounts:
    def test_empty_registry(self):
        assert _counts({}) == {}

    def test_single_status(self):
        registry = {"t1": _make_record("t1", status="Closed")}
        assert _counts(registry) == {"Closed": 1}

    def test_multiple_statuses(self):
        registry = {
            "t1": _make_record("t1", status="Closed"),
            "t2": _make_record("t2", status="Closed"),
            "t3": _make_record("t3", status="Open"),
        }
        counts = _counts(registry)
        assert counts == {"Closed": 2, "Open": 1}

class TestParseScore:
    def test_extracts_integer(self):
        assert _parse_score("SCORE: 8\nVERDICT: pass") == 8

    def test_case_insensitive(self):
        assert _parse_score("score: 5") == 5

    def test_clamps_above_10(self):
        assert _parse_score("SCORE: 15") == 10

    def test_clamps_below_0(self):
        assert _parse_score("SCORE: -2") == 0

    def test_returns_0_when_missing(self):
        assert _parse_score("VERDICT: pass\nSUMMARY: all good") == 0

    def test_returns_0_for_non_integer(self):
        assert _parse_score("SCORE: great") == 0

    def test_handles_extra_whitespace(self):
        assert _parse_score("  SCORE:   7  ") == 7


class TestParseListSection:
    def test_extracts_items(self):
        text = "ISSUES:\n- Missing import\n- Wrong type\nSUMMARY:\nDone."
        items = _parse_list_section(text, "ISSUES")
        assert items == ["Missing import", "Wrong type"]

    def test_none_sentinel_returns_empty(self):
        text = "ISSUES:\n- None\nSUMMARY:\nDone."
        assert _parse_list_section(text, "ISSUES") == []

    def test_stops_at_next_section(self):
        text = "ISSUES:\n- Issue A\nSUGGESTIONS:\n- Suggest B"
        issues = _parse_list_section(text, "ISSUES")
        suggestions = _parse_list_section(text, "SUGGESTIONS")
        assert issues == ["Issue A"]
        assert suggestions == ["Suggest B"]

    def test_returns_empty_when_section_missing(self):
        assert _parse_list_section("SCORE: 8", "ISSUES") == []

    def test_case_insensitive_section_header(self):
        text = "issues:\n- Problem X"
        assert _parse_list_section(text, "ISSUES") == ["Problem X"]


class TestParseSummary:
    def test_extracts_inline_text(self):
        text = "SCORE: 9\nSUMMARY: Everything looks great."
        assert _parse_summary(text) == "Everything looks great."

    def test_extracts_multiline_text(self):
        text = "SUMMARY:\nLine one.\nLine two."
        result = _parse_summary(text)
        assert "Line one." in result
        assert "Line two." in result

    def test_returns_empty_when_missing(self):
        assert _parse_summary("SCORE: 8\nVERDICT: pass") == ""

class TestDecomposeTasks:
    def _llm_tasks(self, tasks: List[Dict]) -> SimpleNamespace:
        return _text_response(json.dumps(tasks))

    def test_creates_task_registry_on_disk(self, tmp_project):
        tasks = [
            {"task_id": "auth", "desc": "Auth module", "dependency_list": []},
            {"task_id": "api",  "desc": "API routes",  "dependency_list": ["auth"]},
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_tasks(tasks)

        state = _base_state()
        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "get_project_schema", return_value={}):
            result = decompose_tasks(state)

        assert set(result["all_task_ids"]) == {"auth", "api"}
        registry = _load_registry(PROJECT_ID)
        assert registry["auth"]["status"] == "Open"
        assert registry["api"]["dependency_list"] == ["auth"]

    def test_strips_markdown_fences(self, tmp_project):
        raw_with_fences = '```json\n[{"task_id":"t1","desc":"T1","dependency_list":[]}]\n```'
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(raw_with_fences)

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "get_project_schema", return_value={}):
            result = decompose_tasks(_base_state())

        assert "t1" in result["all_task_ids"]

    def test_invalid_json_produces_empty_task_list(self, tmp_project):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("not json at all")

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "get_project_schema", return_value={}):
            result = decompose_tasks(_base_state())

        assert result["all_task_ids"] == []

    def test_dangling_dependency_references_are_pruned(self, tmp_project):
        tasks = [{"task_id": "t1", "desc": "T1", "dependency_list": ["ghost-task"]}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_tasks(tasks)

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "get_project_schema", return_value={}):
            decompose_tasks(_base_state())

        registry = _load_registry(PROJECT_ID)
        assert registry["t1"]["dependency_list"] == []

    def test_self_referencing_dependency_is_pruned(self, tmp_project):
        tasks = [{"task_id": "t1", "desc": "T1", "dependency_list": ["t1"]}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_tasks(tasks)

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "get_project_schema", return_value={}):
            decompose_tasks(_base_state())

        assert _load_registry(PROJECT_ID)["t1"]["dependency_list"] == []

    def test_state_fields_initialised(self, tmp_project):
        tasks = [{"task_id": "t1", "desc": "T", "dependency_list": []}]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_tasks(tasks)

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "get_project_schema", return_value={}):
            result = decompose_tasks(_base_state())

        assert result["iteration"] == 0
        assert result["worker_reports"] == {}
        assert result["retry_queue"] == []
        assert result["project_report"] is None


class TestScheduleIteration:
    def test_emits_ready_tasks_as_current_batch(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1"), _make_record("t2")])
        state = _base_state(iteration=0)

        result = schedule_iteration(state)

        assert set(result["current_batch"]) == {"t1", "t2"}
        assert result["iteration"] == 1

    def test_only_emits_tasks_with_satisfied_deps(self, tmp_project):
        records = [
            _make_record("t1", status="Closed"),
            _make_record("t2", deps=["t1"]),
            _make_record("t3", deps=["t2"]),  # t2 not yet closed
        ]
        _seed_registry(tmp_project, records)

        result = schedule_iteration(_base_state())

        assert result["current_batch"] == ["t2"]

    def test_promotes_retry_queue_to_open(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="ReworkRequired")])
        state = _base_state(retry_queue=["t1"])

        schedule_iteration(state)

        assert _load_registry(PROJECT_ID)["t1"]["status"] == "Open"

    def test_prepends_review_notes_to_desc_on_retry(self, tmp_project):
        _seed_registry(tmp_project, [_make_record(
            "t1", status="ReworkRequired",
            review_notes="Fix the import",
        )])
        state = _base_state(retry_queue=["t1"])

        schedule_iteration(state)

        desc = _load_registry(PROJECT_ID)["t1"]["desc"]
        assert "RETRY" in desc
        assert "Fix the import" in desc

    def test_clears_retry_queue_from_state(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])
        state = _base_state(retry_queue=["t1"])

        result = schedule_iteration(state)

        assert result["retry_queue"] == []

    def test_increments_iteration_counter(self, tmp_project):
        _seed_registry(tmp_project, [])
        result = schedule_iteration(_base_state(iteration=3))
        assert result["iteration"] == 4


class TestDispatchWorkers:
    def _make_state_with_batch(self, batch_ids: List[str], existing_reports=None) -> TeamLeadState:
        return _base_state(
            current_batch=batch_ids,
            worker_reports=existing_reports or {},
        )

    def test_completed_worker_marks_task_closed(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])
        report = _make_worker_report("t1", status="completed")

        with patch.object(tla, "run_worker", return_value=report):
            dispatch_workers(self._make_state_with_batch(["t1"]))

        assert _load_registry(PROJECT_ID)["t1"]["status"] == "Closed"

    def test_failed_worker_marks_task_rework_required(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])
        report = _make_worker_report("t1", status="failed", score=2)

        with patch.object(tla, "run_worker", return_value=report):
            dispatch_workers(self._make_state_with_batch(["t1"]))

        assert _load_registry(PROJECT_ID)["t1"]["status"] == "ReworkRequired"

    def test_blocked_worker_marks_task_rework_required(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])
        report = _make_worker_report("t1", status="blocked", blockers=["dep missing"])

        with patch.object(tla, "run_worker", return_value=report):
            dispatch_workers(self._make_state_with_batch(["t1"]))

        assert _load_registry(PROJECT_ID)["t1"]["status"] == "ReworkRequired"

    def test_completed_with_issues_marks_pending_review(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])
        report = _make_worker_report("t1", status="completed_with_issues", score=7)

        with patch.object(tla, "run_worker", return_value=report):
            dispatch_workers(self._make_state_with_batch(["t1"]))

        assert _load_registry(PROJECT_ID)["t1"]["status"] == "PendingReview"

    def test_worker_crash_produces_synthetic_failed_report(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])

        with patch.object(tla, "run_worker", side_effect=RuntimeError("boom")):
            result = dispatch_workers(self._make_state_with_batch(["t1"]))

        assert result["worker_reports"]["t1"]["status"] == "failed"
        assert "boom" in result["worker_reports"]["t1"]["blockers"][0]

    def test_deduplicates_touched_files_across_workers(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1"), _make_record("t2")])
        r1 = _make_worker_report("t1", touched=["shared.py", "a.py"])
        r2 = _make_worker_report("t2", touched=["shared.py", "b.py"])

        with patch.object(tla, "run_worker", side_effect=[r1, r2]):
            result = dispatch_workers(self._make_state_with_batch(["t1", "t2"]))

        assert result["all_touched_files"].count("shared.py") == 1
        assert "a.py" in result["all_touched_files"]
        assert "b.py" in result["all_touched_files"]

    def test_merges_with_existing_reports_from_prior_iterations(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t2")])
        prior = {"t1": _make_worker_report("t1")}
        report = _make_worker_report("t2")

        with patch.object(tla, "run_worker", return_value=report):
            result = dispatch_workers(self._make_state_with_batch(["t2"], existing_reports=prior))

        assert "t1" in result["worker_reports"]
        assert "t2" in result["worker_reports"]

    def test_empty_batch_returns_existing_reports_unchanged(self, tmp_project):
        existing = {"t1": _make_worker_report("t1")}
        state = _base_state(current_batch=[], worker_reports=existing)

        result = dispatch_workers(state)

        assert result["worker_reports"] == existing


class TestCollectReports:
    def test_queues_retryable_rework_tasks(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="ReworkRequired", retry_count=0),
        ])
        result = collect_reports(_base_state())
        assert "t1" in result["retry_queue"]

    def test_does_not_queue_exhausted_tasks(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="ReworkRequired", retry_count=MAX_RETRY_CYCLES),
        ])
        result = collect_reports(_base_state())
        assert "t1" not in result["retry_queue"]

    def test_closed_tasks_not_queued(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="Closed")])
        result = collect_reports(_base_state())
        assert result["retry_queue"] == []

    def test_mixed_registry_only_queues_retryable(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="Closed"),
            _make_record("t2", status="ReworkRequired", retry_count=0),
            _make_record("t3", status="ReworkRequired", retry_count=MAX_RETRY_CYCLES),
        ])
        result = collect_reports(_base_state())
        assert result["retry_queue"] == ["t2"]


class TestReviewBatch:
    def _review_text(self, score: int = 9, verdict: str = "pass",
                     issues: str = "- None", suggestions: str = "- None") -> str:
        return (
            f"SCORE: {score}\nVERDICT: {verdict}\n"
            f"ISSUES:\n{issues}\n"
            f"SUGGESTIONS:\n{suggestions}\n"
            "SUMMARY:\nAll integrated correctly."
        )

    def test_parses_passing_review(self, tmp_project):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(self._review_text(score=9))

        reports = {"t1": _make_worker_report("t1", touched=["src/t1.py"])}
        state = _base_state(current_batch=["t1"], worker_reports=reports)

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "read_file", return_value={"src/t1.py": {1: "x = 1"}}):
            result = review_batch(state)

        review = result["latest_review"]
        assert review["score"] == 9
        assert review["verdict"] == "pass"
        assert review["issues"] == []

    def test_parses_failing_review_with_issues(self, tmp_project):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(
            self._review_text(score=4, verdict="fail", issues="- Bad import\n- Missing glue")
        )

        reports = {"t1": _make_worker_report("t1", touched=["src/t1.py"])}
        state = _base_state(current_batch=["t1"], worker_reports=reports)

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "read_file", return_value={"src/t1.py": {1: "x = 1"}}):
            result = review_batch(state)

        assert result["latest_review"]["verdict"] == "fail"
        assert "Bad import" in result["latest_review"]["issues"]

    def test_file_read_failure_handled_gracefully(self, tmp_project):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(self._review_text())

        reports = {"t1": _make_worker_report("t1", touched=["missing.py"])}
        state = _base_state(current_batch=["t1"], worker_reports=reports)

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "read_file", side_effect=FileNotFoundError("gone")):
            result = review_batch(state)  # must not raise

        assert "latest_review" in result

    def test_appends_to_batch_reviews_list(self, tmp_project):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(self._review_text())

        prior_review: BatchReview = {
            "score": 8, "verdict": "pass", "issues": [], "suggestions": [], "summary": "ok"
        }
        state = _base_state(
            current_batch=["t1"],
            worker_reports={"t1": _make_worker_report("t1")},
            batch_reviews=[prior_review],
        )

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "read_file", return_value={}):
            result = review_batch(state)

        assert len(result["batch_reviews"]) == 2

    def test_score_below_threshold_gives_fail_verdict(self, tmp_project):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(
            self._review_text(score=INTEGRATION_PASS_THRESHOLD - 1, verdict="fail")
        )

        state = _base_state(
            current_batch=["t1"],
            worker_reports={"t1": _make_worker_report("t1")},
        )

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client), \
             patch.object(tla, "read_file", return_value={}):
            result = review_batch(state)

        assert result["latest_review"]["verdict"] == "fail"

    def test_no_files_produces_placeholder_block(self, tmp_project):
        """When no files are touched, files_block must be the placeholder string
        and the LLM must be called with a message that contains it."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(self._review_text())

        # touched=[] — _make_worker_report now correctly sets touched_files=[]
        # because we use 'if touched is None' not 'touched or default'.
        state = _base_state(
            current_batch=["t1"],
            worker_reports={"t1": _make_worker_report("t1", touched=[])},
        )

        # read_file is never called when touched_files=[] — no need to patch it.
        with patch("TeamLeadAgent.Anthropic", return_value=mock_client):
            result = review_batch(state)

        assert "latest_review" in result

        # Extract the user message from the mock call and verify the placeholder.
        call_args = mock_client.messages.create.call_args
        messages_kwarg = call_args.kwargs.get("messages") or call_args[0][4]
        user_msg = messages_kwarg[0]["content"]
        assert "(no files in this batch)" in user_msg


class TestHandleBatchReview:
    def test_fail_verdict_writes_review_notes_to_registry(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="ReworkRequired")])

        failing_review: BatchReview = {
            "score": 4,
            "verdict": "fail",
            "issues": ["Missing import X", "Type mismatch in handler"],
            "suggestions": [],
            "summary": "Integration broken.",
        }
        state = _base_state(latest_review=failing_review, retry_queue=["t1"])

        handle_batch_review(state)

        rec = _load_registry(PROJECT_ID)["t1"]
        assert "Missing import X" in rec["review_notes"]
        assert "Type mismatch" in rec["review_notes"]

    def test_fail_verdict_increments_retry_count(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="ReworkRequired", retry_count=0)])

        failing_review: BatchReview = {
            "score": 3, "verdict": "fail",
            "issues": ["Bad issue"], "suggestions": [], "summary": "fail",
        }
        state = _base_state(latest_review=failing_review, retry_queue=["t1"])

        handle_batch_review(state)

        assert _load_registry(PROJECT_ID)["t1"]["retry_count"] == 1

    def test_pass_verdict_does_not_write_review_notes(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="Closed")])

        passing_review: BatchReview = {
            "score": 9, "verdict": "pass",
            "issues": [], "suggestions": [], "summary": "All good.",
        }
        state = _base_state(latest_review=passing_review, retry_queue=[])

        handle_batch_review(state)

        assert _load_registry(PROJECT_ID)["t1"]["review_notes"] == ""

    def test_no_review_is_a_no_op(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1")])
        state = _base_state(latest_review=None, retry_queue=[])
        handle_batch_review(state)  # must not raise


class TestFinalize:
    def test_writes_project_report_to_disk(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="Closed",
            report={"summary": "done"})])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("Project completed successfully.")

        state = _base_state(
            all_touched_files=["src/t1.py"],
            batch_reviews=[{"score": 9, "verdict": "pass", "issues": [],
                            "suggestions": [], "summary": "ok"}],
        )

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client):
            finalize(state)

        report_file = tmp_project / "project_report.json"
        assert report_file.exists()
        report = json.loads(report_file.read_text())
        assert report["project_id"] == PROJECT_ID

    def test_all_closed_gives_success_status(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="Closed", report={"summary": "ok"}),
            _make_record("t2", status="Closed", report={"summary": "ok"}),
        ])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("All done.")

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client):
            result = finalize(_base_state())

        assert result["project_report"]["final_status"] == "success"
        assert result["project_report"]["completed_tasks"] == 2

    def test_partial_closed_gives_partial_status(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="Closed", report={"summary": "ok"}),
            _make_record("t2", status="ReworkRequired", retry_count=MAX_RETRY_CYCLES),
        ])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("Partial.")

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client):
            result = finalize(_base_state())

        assert result["project_report"]["final_status"] == "partial"

    def test_none_closed_gives_failed_status(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="ReworkRequired", retry_count=MAX_RETRY_CYCLES),
        ])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("Failed.")

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client):
            result = finalize(_base_state())

        assert result["project_report"]["final_status"] == "failed"

    def test_report_contains_all_required_fields(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="Closed",
            report={"summary": "ok"})])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("Done.")

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client):
            result = finalize(_base_state(
                all_touched_files=["src/t1.py"],
                batch_reviews=[{"score": 9, "verdict": "pass",
                                "issues": [], "suggestions": [], "summary": "ok"}],
            ))

        report = result["project_report"]
        required = {
            "project_id", "total_tasks", "completed_tasks", "failed_tasks",
            "blocked_tasks", "all_touched_files", "batch_review_scores",
            "final_status", "summary", "completed_at",
        }
        assert required.issubset(set(report.keys()))

    def test_completed_at_is_iso8601(self, tmp_project):
        _seed_registry(tmp_project, [_make_record("t1", status="Closed",
            report={"summary": "ok"})])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("Done.")

        with patch("TeamLeadAgent.Anthropic", return_value=mock_client):
            result = finalize(_base_state())

        ts = result["project_report"]["completed_at"]
        assert ts.endswith("Z")
        datetime.fromisoformat(ts.rstrip("Z"))

class TestRouteAfterReview:
    def test_routes_to_schedule_when_open_tasks_exist(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="Closed"),
            _make_record("t2", status="Open", deps=["t1"]),
        ])
        assert route_after_review(_base_state()) == "schedule_iteration"

    def test_routes_to_schedule_when_retryable_rework_exists(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="ReworkRequired", retry_count=0),
        ])
        assert route_after_review(_base_state()) == "schedule_iteration"

    def test_routes_to_finalize_when_all_closed(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="Closed"),
            _make_record("t2", status="Closed"),
        ])
        assert route_after_review(_base_state()) == "finalize"

    def test_routes_to_finalize_when_all_retries_exhausted(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="ReworkRequired", retry_count=MAX_RETRY_CYCLES),
        ])
        assert route_after_review(_base_state()) == "finalize"

    def test_routes_to_schedule_when_pending_review_tasks_exist(self, tmp_project):
        _seed_registry(tmp_project, [
            _make_record("t1", status="PendingReview"),
        ])
        assert route_after_review(_base_state()) == "schedule_iteration"


class TestRouteAfterSchedule:
    def test_routes_to_dispatch_when_batch_nonempty(self):
        state = _base_state(current_batch=["t1", "t2"])
        assert route_after_schedule(state) == "dispatch_workers"

    def test_routes_to_finalize_when_batch_empty(self):
        state = _base_state(current_batch=[])
        assert route_after_schedule(state) == "finalize"

class TestBuildTeamleadGraph:
    def test_all_nodes_registered(self):
        graph = build_teamlead_graph()
        expected = {
            "decompose_tasks", "schedule_iteration", "dispatch_workers",
            "collect_reports", "review_batch", "handle_batch_review", "finalize",
        }
        assert expected.issubset(set(graph.nodes.keys()))

    def test_compiles_without_error(self):
        assert build_teamlead_graph().compile() is not None

class TestRunTeamlead:
    def test_returns_project_report(self, tmp_project):
        """
        Minimal smoke test: run_teamlead with a mocked agent that returns a
        pre-built ProjectReport. Verifies the entry point passes it through.
        """
        fake_report: ProjectReport = {
            "project_id": PROJECT_ID,
            "total_tasks": 1,
            "completed_tasks": 1,
            "failed_tasks": 0,
            "blocked_tasks": 0,
            "all_touched_files": ["src/t1.py"],
            "batch_review_scores": [9],
            "final_status": "success",
            "summary": "All done.",
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"project_report": fake_report}

        with patch.object(tla, "create_teamlead_agent", return_value=fake_agent):
            result = run_teamlead(PROJECT_ID, "Build a REST API.")

        assert result["project_id"] == PROJECT_ID
        assert result["final_status"] == "success"

    def test_produces_fallback_when_report_is_none(self, tmp_project):
        """If the graph produces no report, run_teamlead must not return None."""
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"project_report": None}

        with patch.object(tla, "create_teamlead_agent", return_value=fake_agent):
            result = run_teamlead(PROJECT_ID, "Build a REST API.")

        assert result is not None
        assert result["final_status"] == "failed"