from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock, patch, call

import pytest

# Module under test
import WorkerAgent as wa
from WorkerAgent import (
    MAX_REWORK_CYCLES,
    MAX_TOOL_ITERATIONS,
    REVIEW_PASS_THRESHOLD,
    WorkerReport,
    WorkerState,
    _dispatch_tool,
    _parse_review_score,
    _run_tool_loop,
    build_worker_graph,
    execute_task,
    generate_tests,
    plan_task,
    report_to_teamlead,
    review_task,
    rework_task,
    run_worker,
    should_rework,
)
from models import Task

@pytest.fixture
def sample_task() -> Task:
    return Task(
        task_id="task-001",
        project_id="proj-1",
        desc="Implement a hello_world() function in src/hello.py",
        status="InProgress",
        worker_id="worker-1",
        dependency_list=[],
    )


@pytest.fixture
def base_state(sample_task) -> WorkerState:
    """Minimal valid WorkerState for node unit tests."""
    return WorkerState(
        task=sample_task,
        worker_id="worker-1",
        plan="1. Create src/hello.py\n2. Add hello_world()",
        touched_files=["src/hello.py"],
        execution_notes="Created src/hello.py",
        review_score=8,
        review_feedback="SCORE: 8\nVERDICT: PASS\nISSUES:\n- None\nSUGGESTIONS:\n- None\nSUMMARY:\nLooks good.",
        rework_count=0,
        test_file_path=None,
        test_code="",
        messages=[],
        report=None,
    )

def _make_content_block(text: str = "", block_type: str = "text", **kwargs) -> SimpleNamespace:
    """Build a minimal content block matching the Anthropic SDK shape."""
    block = SimpleNamespace(type=block_type, text=text if block_type == "text" else None, **kwargs)
    return block

def _make_tool_use_block(tool_id: str, name: str, input_: Dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=input_, text=None)

def _make_api_response(
    content: List[SimpleNamespace],
    stop_reason: str = "end_turn",
) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)

def _text_response(text: str) -> SimpleNamespace:
    """Convenience: a response with a single text block and stop_reason=end_turn."""
    return _make_api_response([_make_content_block(text)])

def _tool_response(tool_id: str, name: str, input_: Dict) -> SimpleNamespace:
    """Convenience: a response that calls one tool."""
    return _make_api_response(
        [_make_tool_use_block(tool_id, name, input_)],
        stop_reason="tool_use",
    )

class TestParseReviewScore:
    def test_extracts_score_from_well_formed_review(self):
        text = "SCORE: 8\nVERDICT: PASS\nSUMMARY: Good job."
        assert _parse_review_score(text) == 8

    def test_case_insensitive(self):
        assert _parse_review_score("score: 5") == 5

    def test_clamps_above_10(self):
        assert _parse_review_score("SCORE: 15") == 10

    def test_clamps_below_0(self):
        assert _parse_review_score("SCORE: -3") == 0

    def test_returns_0_when_no_score_line(self):
        assert _parse_review_score("VERDICT: PASS\nSUMMARY: All good.") == 0

    def test_returns_0_on_non_integer_value(self):
        assert _parse_review_score("SCORE: excellent") == 0

    def test_handles_extra_whitespace(self):
        assert _parse_review_score("  SCORE:   7  ") == 7

    def test_picks_first_score_line(self):
        assert _parse_review_score("SCORE: 9\nSCORE: 3") == 9

    def test_empty_string_returns_0(self):
        assert _parse_review_score("") == 0

class TestDispatchTool:
    def test_get_project_schema_returns_json(self):
        schema = {"main.py": {"description": "entry"}}
        with patch.object(wa, "get_project_schema", return_value=schema):
            result = _dispatch_tool("get_project_schema", {"project_id": "p1"})
        assert json.loads(result) == schema

    def test_read_file_converts_int_keys_to_strings(self):
        with patch.object(wa, "read_file", return_value={"f.py": {1: "line1", 2: "line2"}}):
            result = _dispatch_tool("read_file", {"project_id": "p1", "file_paths": ["f.py"]})
        parsed = json.loads(result)
        # Keys must be strings (JSON only supports string keys)
        assert "1" in parsed["f.py"]
        assert parsed["f.py"]["1"] == "line1"

    def test_create_file_success(self):
        with patch.object(wa, "create_file", return_value=None):
            result = _dispatch_tool("create_file", {
                "project_id": "p1", "file_path": "a.py",
                "file_description": "desc", "worker_id": "w1",
            })
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["error"] is None

    def test_create_file_returns_error_string(self):
        with patch.object(wa, "create_file", return_value="File already exists"):
            result = _dispatch_tool("create_file", {
                "project_id": "p1", "file_path": "a.py",
                "file_description": "desc", "worker_id": "w1",
            })
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert parsed["error"] == "File already exists"

    def test_write_file_success(self):
        with patch.object(wa, "write_file", return_value=None):
            result = _dispatch_tool("write_file", {
                "project_id": "p1", "worker_id": "w1", "file_path": "a.py",
                "edits": [{"operation": "insert", "start_line": 1, "new_value": "x"}],
            })
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_write_file_constructs_edit_objects_correctly(self):
        """Verify that raw dict edits are converted to Edit dataclass instances."""
        captured = []
        def fake_write(project_id, worker_id, file_path, edits):
            captured.extend(edits)
            return None

        with patch.object(wa, "write_file", side_effect=fake_write):
            _dispatch_tool("write_file", {
                "project_id": "p1", "worker_id": "w1", "file_path": "a.py",
                "edits": [{
                    "operation": "update",
                    "start_line": 3,
                    "end_line": 5,
                    "new_value": "new content",
                    "method_description_changes": {"fn": "desc"},
                }],
            })

        assert len(captured) == 1
        edit = captured[0]
        assert edit.operation == "update"
        assert edit.start_line == 3
        assert edit.end_line == 5
        assert edit.new_value == "new content"
        assert edit.method_description_changes == {"fn": "desc"}

    def test_write_file_error(self):
        with patch.object(wa, "write_file", return_value="lock error"):
            result = _dispatch_tool("write_file", {
                "project_id": "p1", "worker_id": "w1", "file_path": "a.py",
                "edits": [{"operation": "delete", "start_line": 1, "end_line": 2}],
            })
        parsed = json.loads(result)
        assert parsed["success"] is False

    def test_unknown_tool_returns_error(self):
        result = _dispatch_tool("teleport", {})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "teleport" in parsed["error"]

    def test_exception_is_caught_and_returned_as_error(self):
        with patch.object(wa, "get_project_schema", side_effect=RuntimeError("boom")):
            result = _dispatch_tool("get_project_schema", {"project_id": "p1"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "boom" in parsed["error"]

class TestRunToolLoop:
    def _make_client(self, responses: List) -> MagicMock:
        client = MagicMock()
        client.messages.create.side_effect = responses
        return client

    def test_exits_immediately_when_no_tool_calls(self):
        """stop_reason=end_turn on first call → loop runs exactly once."""
        client = self._make_client([_text_response("All done.")])
        messages, touched, notes = _run_tool_loop(client, "sys", [{"role": "user", "content": "go"}])

        assert client.messages.create.call_count == 1
        assert "All done." in notes
        assert touched == []

    def test_multi_turn_tool_loop(self):
        """
        Turn 1: LLM calls create_file → tool result injected.
        Turn 2: LLM calls write_file → tool result injected.
        Turn 3: LLM sends end_turn text → loop exits.
        """
        turn1 = _tool_response("t1", "create_file", {"file_path": "a.py", "project_id": "p", "file_description": "d", "worker_id": "w"})
        turn2 = _tool_response("t2", "write_file", {"file_path": "a.py", "project_id": "p", "worker_id": "w", "edits": [{"operation": "insert", "start_line": 1, "new_value": "x"}]})
        turn3 = _text_response("Done implementing.")

        client = self._make_client([turn1, turn2, turn3])

        with patch.object(wa, "_dispatch_tool", return_value='{"success": true, "error": null}'):
            messages, touched, notes = _run_tool_loop(client, "sys", [])

        assert client.messages.create.call_count == 3
        # Both file paths should be in touched_files
        assert "a.py" in touched
        # Notes from the final text turn
        assert "Done implementing." in notes

    def test_tracks_touched_files_without_duplicates(self):
        """Same file written twice → only one entry in touched_files."""
        turn1 = _tool_response("t1", "write_file", {"file_path": "dup.py", "project_id": "p", "worker_id": "w", "edits": []})
        turn2 = _tool_response("t2", "write_file", {"file_path": "dup.py", "project_id": "p", "worker_id": "w", "edits": []})
        turn3 = _text_response("Done.")

        client = self._make_client([turn1, turn2, turn3])

        with patch.object(wa, "_dispatch_tool", return_value='{"success": true, "error": null}'):
            _, touched, _ = _run_tool_loop(client, "sys", [])

        assert touched.count("dup.py") == 1

    def test_iteration_cap_appends_warning(self):
        """If max_iterations is hit, a warning note is appended."""
        always_tool = _tool_response("tx", "get_project_schema", {"project_id": "p"})

        client = self._make_client([always_tool] * 5)

        with patch.object(wa, "_dispatch_tool", return_value="{}"):
            _, _, notes = _run_tool_loop(client, "sys", [], max_iterations=3)

        assert "WARNING" in notes
        assert client.messages.create.call_count == 3

    def test_tool_results_are_injected_as_user_turn(self):
        """
        After a tool_use response the loop must inject a user turn carrying the
        tool_result before making the next API call.

        Message order on the 2nd API call:
          [0] assistant  serialized tool_use block from turn 1
          [1] user       tool_result turn (what we are testing)
          [2] assistant  serialized text from turn 2 (appended before stop_reason
                         check fires, so it appears in the list too)

        We search for the tool_result user turn rather than relying on position.
        """
        tool_resp = _tool_response("id-1", "get_project_schema", {"project_id": "p"})
        done_resp = _text_response("Done.")

        client = self._make_client([tool_resp, done_resp])

        with patch.object(wa, "_dispatch_tool", return_value='{"schema": "x"}'):
            _run_tool_loop(client, "sys", [])

        second_call = client.messages.create.call_args_list[1]
        second_call_messages = second_call.kwargs["messages"]

        # Find the user turn that carries tool_result content
        tool_result_turns = [
            m for m in second_call_messages
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
            and any(c.get("type") == "tool_result" for c in m["content"])
        ]
        assert len(tool_result_turns) == 1, (
            f"Expected 1 tool_result user turn, found {len(tool_result_turns)}"
        )
        tool_turn = tool_result_turns[0]
        assert tool_turn["content"][0]["type"] == "tool_result"
        assert tool_turn["content"][0]["tool_use_id"] == "id-1"

class TestShouldRework:
    def _state(self, score: int, rework_count: int) -> dict:
        return {"review_score": score, "rework_count": rework_count}

    def test_low_score_first_cycle_routes_to_rework(self):
        assert should_rework(self._state(score=4, rework_count=0)) == "rework_task"

    def test_low_score_second_cycle_routes_to_rework(self):
        assert should_rework(self._state(score=4, rework_count=1)) == "rework_task"

    def test_low_score_at_max_rework_routes_to_generate_tests(self):
        """Once rework_count == MAX_REWORK_CYCLES, stop retrying."""
        assert should_rework(self._state(score=4, rework_count=MAX_REWORK_CYCLES)) == "generate_tests"

    def test_score_at_threshold_routes_to_generate_tests(self):
        assert should_rework(self._state(score=REVIEW_PASS_THRESHOLD, rework_count=0)) == "generate_tests"

    def test_score_above_threshold_routes_to_generate_tests(self):
        assert should_rework(self._state(score=9, rework_count=0)) == "generate_tests"

    def test_score_one_below_threshold_routes_to_rework(self):
        assert should_rework(self._state(score=REVIEW_PASS_THRESHOLD - 1, rework_count=0)) == "rework_task"

class TestPlanTask:
    def test_returns_plan_from_llm(self, base_state):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("Step 1: do X\nStep 2: do Y")

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "get_project_schema", return_value={"a.py": {}}):
            result = plan_task(base_state)

        assert result["plan"] == "Step 1: do X\nStep 2: do Y"
        assert result["rework_count"] == 0
        assert result["touched_files"] == []
        assert result["report"] is None

    def test_schema_load_failure_is_graceful(self, base_state):
        """If get_project_schema raises, plan_task should still succeed."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("plan text")

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "get_project_schema", side_effect=FileNotFoundError("no schema")):
            result = plan_task(base_state)

        # Should still produce a plan — the error is embedded in the system prompt
        assert "plan text" in result["plan"]

    def test_empty_llm_response_falls_back_to_placeholder(self, base_state):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_api_response([])

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "get_project_schema", return_value={}):
            result = plan_task(base_state)

        assert result["plan"] == "(empty plan)"

    def test_messages_added_to_state(self, base_state):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("my plan")

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "get_project_schema", return_value={}):
            result = plan_task(base_state)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

class TestExecuteTask:
    def test_delegates_to_run_tool_loop(self, base_state):
        expected_touched = ["src/main.py"]
        expected_notes = "Implemented main."

        with patch("WorkerAgent.Anthropic") as mock_anthropic, \
             patch.object(wa, "_run_tool_loop", return_value=([], expected_touched, expected_notes)) as mock_loop:
            mock_anthropic.return_value = MagicMock()
            result = execute_task(base_state)

        assert result["touched_files"] == expected_touched
        assert result["execution_notes"] == expected_notes
        assert mock_loop.call_count == 1

    def test_starts_with_fresh_message_list(self, base_state):
        """
        execute_task must NOT forward state["messages"] to _run_tool_loop.

        LangGraph converts plain message dicts in state into its own message
        objects (HumanMessage / AIMessage) which the Anthropic API rejects with
        a 400 "messages.0.role: Field required" error. execute_task therefore
        always starts a fresh single-user-message list so only plain dicts reach
        the API, regardless of what is stored in state["messages"].
        """
        # Even with prior messages in state, execute_task must ignore them.
        base_state["messages"] = [{"role": "user", "content": "prior msg that must be ignored"}]

        captured_initial = []
        def fake_loop(client, system, initial_messages, **kw):
            captured_initial.extend(initial_messages)
            return [], [], ""

        with patch("WorkerAgent.Anthropic"), \
             patch.object(wa, "_run_tool_loop", side_effect=fake_loop):
            execute_task(base_state)

        # Must be exactly 1 message — the fresh task-execution prompt.
        # The prior state message must NOT be present.
        assert len(captured_initial) == 1, (
            f"Expected 1 initial message, got {len(captured_initial)}: {captured_initial}"
        )
        assert captured_initial[0]["role"] == "user"
        assert "prior msg" not in captured_initial[0]["content"]

    def test_plan_injected_into_system_prompt(self, base_state):
        base_state["plan"] = "UNIQUE_PLAN_CONTENT_XYZ"
        captured_system = []

        def fake_loop(client, system, initial_messages, **kw):
            captured_system.append(system)
            return [], [], ""

        with patch("WorkerAgent.Anthropic"), \
             patch.object(wa, "_run_tool_loop", side_effect=fake_loop):
            execute_task(base_state)

        assert "UNIQUE_PLAN_CONTENT_XYZ" in captured_system[0]

class TestReviewTask:
    def _review_response(self, score: int, verdict: str = "PASS") -> SimpleNamespace:
        text = (
            f"SCORE: {score}\n"
            f"VERDICT: {verdict}\n"
            "ISSUES:\n- None\n"
            "SUGGESTIONS:\n- None\n"
            "SUMMARY:\nLooks good."
        )
        return _text_response(text)

    def test_parses_score_correctly(self, base_state):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._review_response(9)

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", return_value={"src/hello.py": {1: "def hello(): pass"}}):
            result = review_task(base_state)

        assert result["review_score"] == 9

    def test_file_read_failure_is_graceful(self, base_state):
        """If a touched file can't be read, review continues with error note."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._review_response(7)

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", side_effect=FileNotFoundError("gone")):
            result = review_task(base_state)

        # Should still return a score — not crash
        assert "review_score" in result

    def test_rework_context_injected_on_second_cycle(self, base_state):
        base_state["rework_count"] = 1
        base_state["review_feedback"] = "SCORE: 5\nVERDICT: FAIL\nISSUES:\n- Missing tests"

        captured_user_messages = []
        mock_client = MagicMock()

        def capture_and_respond(**kwargs):
            captured_user_messages.append(kwargs["messages"][0]["content"])
            return self._review_response(8)

        mock_client.messages.create.side_effect = capture_and_respond

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", return_value={"src/hello.py": {1: "x"}}):
            review_task(base_state)

        assert "rework cycle 1" in captured_user_messages[0].lower()

    def test_no_rework_context_on_first_cycle(self, base_state):
        base_state["rework_count"] = 0
        captured = []
        mock_client = MagicMock()

        def capture(**kwargs):
            captured.append(kwargs["messages"][0]["content"])
            return self._review_response(8)

        mock_client.messages.create.side_effect = capture

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", return_value={"src/hello.py": {1: "x"}}):
            review_task(base_state)

        assert "rework cycle" not in captured[0].lower()

    def test_no_touched_files_still_succeeds(self, base_state):
        base_state["touched_files"] = []
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._review_response(6)

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = review_task(base_state)

        assert result["review_score"] == 6

class TestReworkTask:
    def test_increments_rework_count(self, base_state):
        base_state["rework_count"] = 1

        with patch("WorkerAgent.Anthropic"), \
             patch.object(wa, "_run_tool_loop", return_value=([], ["new.py"], "fixed it")):
            result = rework_task(base_state)

        assert result["rework_count"] == 2

    def test_merges_touched_files_without_duplicates(self, base_state):
        base_state["touched_files"] = ["existing.py"]

        with patch("WorkerAgent.Anthropic"), \
             patch.object(wa, "_run_tool_loop", return_value=([], ["existing.py", "new.py"], "")):
            result = rework_task(base_state)

        assert result["touched_files"].count("existing.py") == 1
        assert "new.py" in result["touched_files"]

    def test_appends_notes_with_cycle_header(self, base_state):
        base_state["execution_notes"] = "initial notes"
        base_state["rework_count"] = 0

        with patch("WorkerAgent.Anthropic"), \
             patch.object(wa, "_run_tool_loop", return_value=([], [], "rework notes")):
            result = rework_task(base_state)

        assert "initial notes" in result["execution_notes"]
        assert "Rework cycle 1" in result["execution_notes"]
        assert "rework notes" in result["execution_notes"]

    def test_feedback_injected_into_system_prompt(self, base_state):
        base_state["review_feedback"] = "UNIQUE_FEEDBACK_STRING_ABC"
        captured_system = []

        def fake_loop(client, system, initial_messages, **kw):
            captured_system.append(system)
            return [], [], ""

        with patch("WorkerAgent.Anthropic"), \
             patch.object(wa, "_run_tool_loop", side_effect=fake_loop):
            rework_task(base_state)

        assert "UNIQUE_FEEDBACK_STRING_ABC" in captured_system[0]

class TestGenerateTests:
    def test_saves_test_file_and_returns_path(self, base_state):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("def test_hello(): assert True")

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", return_value={"src/hello.py": {1: "def hello(): pass"}}), \
             patch.object(wa, "create_file", return_value=None), \
             patch.object(wa, "write_file", return_value=None):
            result = generate_tests(base_state)

        assert result["test_file_path"] == f"tests/test_{base_state['task'].task_id}.py"
        assert "def test_hello" in result["test_code"]

    def test_strips_markdown_fences(self, base_state):
        raw_with_fences = "```python\ndef test_x(): pass\n```"
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response(raw_with_fences)

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", return_value={"src/hello.py": {1: "x"}}), \
             patch.object(wa, "create_file", return_value=None), \
             patch.object(wa, "write_file", return_value=None):
            result = generate_tests(base_state)

        assert "```" not in result["test_code"]
        assert "def test_x" in result["test_code"]

    def test_test_file_path_is_none_when_create_file_fails(self, base_state):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("def test_x(): pass")

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", return_value={}), \
             patch.object(wa, "create_file", return_value="already exists"):
            result = generate_tests(base_state)

        assert result["test_file_path"] is None

    def test_test_file_path_is_none_when_write_file_fails(self, base_state):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("def test_x(): pass")

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", return_value={}), \
             patch.object(wa, "create_file", return_value=None), \
             patch.object(wa, "write_file", return_value="write error"):
            result = generate_tests(base_state)

        assert result["test_file_path"] is None

    def test_file_read_failure_handled_gracefully(self, base_state):
        """A touched file that can't be read shouldn't crash generate_tests."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("def test_x(): pass")

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", side_effect=FileNotFoundError("gone")), \
             patch.object(wa, "create_file", return_value=None), \
             patch.object(wa, "write_file", return_value=None):
            result = generate_tests(base_state)

        assert "test_code" in result

    def test_uses_correct_test_file_path_format(self, base_state):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("def test_x(): pass")

        created_paths = []
        def capture_create(project_id, file_path, desc, worker_id):
            created_paths.append(file_path)
            return None

        with patch("WorkerAgent.Anthropic", return_value=mock_client), \
             patch.object(wa, "read_file", return_value={}), \
             patch.object(wa, "create_file", side_effect=capture_create), \
             patch.object(wa, "write_file", return_value=None):
            generate_tests(base_state)

        assert created_paths[0] == f"tests/test_{base_state['task'].task_id}.py"

class TestReportToTeamlead:
    def _llm_json(self, summary="Done.", blockers=None, suggestions=None) -> SimpleNamespace:
        payload = {
            "summary": summary,
            "blockers": blockers or [],
            "suggestions": suggestions or [],
        }
        return _text_response(json.dumps(payload))

    def test_status_completed_when_high_score_no_blockers(self, base_state):
        base_state["review_score"] = 9
        base_state["rework_count"] = 0

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_json()

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        assert result["report"]["status"] == "completed"

    def test_status_completed_with_issues_for_borderline_score(self, base_state):
        base_state["review_score"] = 7   # threshold, but < 9
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_json()

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        assert result["report"]["status"] == "completed_with_issues"

    def test_status_blocked_when_blockers_present(self, base_state):
        base_state["review_score"] = 8
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_json(
            blockers=["Missing import in module X"]
        )

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        assert result["report"]["status"] == "blocked"
        assert len(result["report"]["blockers"]) == 1

    def test_status_failed_when_rework_exhausted_and_low_score(self, base_state):
        base_state["review_score"] = 4
        base_state["rework_count"] = MAX_REWORK_CYCLES
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_json()

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        assert result["report"]["status"] == "failed"

    def test_status_failed_when_low_score_rework_not_exhausted(self, base_state):
        """score < threshold with rework_count < MAX but no blockers → failed."""
        base_state["review_score"] = 3
        base_state["rework_count"] = 0
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_json()

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        assert result["report"]["status"] == "failed"

    def test_json_parse_fallback_on_malformed_llm_output(self, base_state):
        """If the LLM returns non-JSON, fall back gracefully — don't crash."""
        base_state["review_score"] = 8
        base_state["execution_notes"] = "Implementation done"
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _text_response("Sorry, I can't produce JSON right now.")

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        report = result["report"]
        assert report is not None
        assert report["status"] in ("completed", "completed_with_issues", "blocked", "failed")

    def test_report_contains_all_required_fields(self, base_state):
        base_state["review_score"] = 8
        base_state["test_file_path"] = "tests/test_task-001.py"
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_json(summary="All done.")

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        report = result["report"]
        required_fields = {
            "task_id", "worker_id", "status", "summary", "touched_files",
            "test_file_path", "review_score", "review_feedback", "blockers",
            "suggestions", "execution_notes", "completed_at",
        }
        assert required_fields.issubset(set(report.keys()))

    def test_completed_at_is_iso8601(self, base_state):
        base_state["review_score"] = 9
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_json()

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        ts = result["report"]["completed_at"]
        # Should parse without raising
        assert ts.endswith("Z")
        datetime.fromisoformat(ts.rstrip("Z"))

    def test_report_carries_correct_task_and_worker_ids(self, base_state):
        base_state["review_score"] = 9
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._llm_json()

        with patch("WorkerAgent.Anthropic", return_value=mock_client):
            result = report_to_teamlead(base_state)

        assert result["report"]["task_id"] == base_state["task"].task_id
        assert result["report"]["worker_id"] == base_state["worker_id"]

class TestBuildWorkerGraph:
    def test_all_nodes_registered(self):
        graph = build_worker_graph()
        # Access internal node map — LangGraph stores these in graph.nodes
        node_names = set(graph.nodes.keys())
        expected = {
            "plan_task", "execute_task", "review_task",
            "rework_task", "generate_tests", "report_to_teamlead",
        }
        assert expected.issubset(node_names)

    def test_graph_compiles_without_error(self):
        graph = build_worker_graph()
        compiled = graph.compile()
        assert compiled is not None

class TestRunWorker:
    """
    End-to-end tests through run_worker() with Anthropic API and WorkerTools
    patched at the module level.

    WHY NOT patch node functions directly:
    LangGraph calls graph.add_node("name", fn) which captures a direct reference
    to the function object at compile() time. patch.object(wa, "node_fn", ...)
    replaces the name in the module __dict__ AFTER the graph has already stored
    the original reference — so the patch never reaches the compiled graph.

    CORRECT approach: let the real node functions run, but patch their
    dependencies (Anthropic client, WorkerTools file I/O) so no real API calls
    or filesystem access occurs.
    """

    # ------------------------------------------------------------------
    # Shared LLM response builders
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_response():
        return _text_response("1. Create src/hello.py\n2. Implement hello_world()")

    @staticmethod
    def _exec_response():
        """Single end_turn response — no tool calls, simulates execute completing."""
        return _text_response("Implementation complete.")

    @staticmethod
    def _review_pass_response(score: int = 9):
        text = (
            f"SCORE: {score}\nVERDICT: PASS\n"
            "ISSUES:\n- None\nSUGGESTIONS:\n- None\n"
            "SUMMARY:\nAll good."
        )
        return _text_response(text)

    @staticmethod
    def _test_gen_response():
        return _text_response("def test_hello():\n    assert True")

    @staticmethod
    def _report_response():
        return _text_response(json.dumps({
            "summary": "Implemented hello_world successfully.",
            "blockers": [],
            "suggestions": [],
        }))

    @staticmethod
    def _all_llm_responses(score: int = 9):
        """
        Return responses in the exact order the graph node sequence calls the LLM:
          1. plan_task         → 1 call
          2. execute_task      → 1 call (no tool use, exits immediately)
          3. review_task       → 1 call
          4. generate_tests    → 1 call
          5. report_to_teamlead→ 1 call
        """
        return [
            TestRunWorker._plan_response(),
            TestRunWorker._exec_response(),
            TestRunWorker._review_pass_response(score),
            TestRunWorker._test_gen_response(),
            TestRunWorker._report_response(),
        ]

    # ------------------------------------------------------------------
    # Context manager: patch everything external
    # ------------------------------------------------------------------

    @staticmethod
    def _all_patches(score: int = 9):
        """
        Returns a list of patch objects. Use with contextlib.ExitStack.
        Patches:
          - WorkerAgent.Anthropic  → mock client that returns canned responses
          - WorkerTools functions  → no-ops (no filesystem access)
        """
        import contextlib

        @contextlib.contextmanager
        def cm():
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = TestRunWorker._all_llm_responses(score)

            with patch("WorkerAgent.Anthropic", return_value=mock_client), \
                 patch.object(wa, "get_project_schema", return_value={}), \
                 patch.object(wa, "read_file", return_value={}), \
                 patch.object(wa, "create_file", return_value=None), \
                 patch.object(wa, "write_file", return_value=None):
                yield mock_client

        return cm()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_run_worker_returns_worker_report(self, sample_task):
        """Full graph run returns a populated WorkerReport with correct task_id."""
        with self._all_patches(score=9):
            report = run_worker(sample_task, "worker-1")

        assert report is not None
        assert report["task_id"] == sample_task.task_id
        assert report["worker_id"] == "worker-1"
        assert report["status"] in ("completed", "completed_with_issues", "blocked", "failed")
        # Required fields always present
        for field in ("summary", "touched_files", "review_score", "review_feedback",
                      "blockers", "suggestions", "execution_notes", "completed_at"):
            assert field in report, f"Missing field: {field}"

    def test_run_worker_high_score_skips_rework(self, sample_task):
        """
        Score ≥ threshold → graph routes review → generate_tests, never rework.
        Observable proof: the LLM is called exactly 5 times (plan, execute,
        review, generate_tests, report) — NOT 7 (which would include rework+re-review).
        """
        with self._all_patches(score=9) as mock_client:
            report = run_worker(sample_task, "worker-1")

        # 5 LLM calls: plan / execute / review / gen_tests / report
        assert mock_client.messages.create.call_count == 5
        assert report["review_score"] == 9

    def test_run_worker_produces_fallback_report_when_report_is_none(self, sample_task):
        """
        If the graph produces no report (report=None in final state),
        run_worker must construct a fallback rather than returning None.
        """
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {
            "report": None,
            "touched_files": ["a.py"],
            "execution_notes": "partial",
        }

        with patch.object(wa, "create_worker_agent", return_value=fake_agent):
            report = run_worker(sample_task, "worker-1")

        assert report is not None
        assert report["status"] == "failed"
        assert len(report["blockers"]) > 0