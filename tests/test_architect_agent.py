from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest

import ArchitectAgent as aa
from ArchitectAgent import (
    MAX_CLARIFICATION_ROUNDS,
    MAX_QUESTIONS_PER_ROUND,
    COMPLETENESS_THRESHOLD,
    ArchitectState,
    ArchitectureSpec,
    Question,
    RequirementsReview,
    UserAnswer,
    _format_question_for_display,
    _get_interrupt_value,
    _normalise_answer,
    _parse_questions,
    _render_understanding,
    analyse_input,
    ask_user,
    build_architect_graph,
    check_completeness,
    create_architect_agent,
    generate_questions,
    handle_user_review,
    incorporate_answers,
    present_summary,
    produce_architecture_spec,
    route_after_completeness,
    route_after_review,
)


PROJECT_ID = "test-proj-001"


def _make_question(
    qid: str = "q1",
    question: str = "What database will you use?",
    qtype: str = "multiple_choice",
    options: Optional[List[str]] = None,
    reason: str = "Affects storage layer design",
) -> Question:
    return {
        "id": qid,
        "question": question,
        "type": qtype,
        "options": options or ["PostgreSQL", "SQLite", "In-memory"],
        "reason": reason,
    }


def _make_answer(qid: str = "q1", question: str = "What DB?", answer: str = "PostgreSQL") -> UserAnswer:
    return {"question_id": qid, "question": question, "answer": answer}


def _base_state(**overrides) -> ArchitectState:
    state: ArchitectState = {
        "project_id": PROJECT_ID,
        "raw_input": "Build a simple REST API",
        "understanding": "## What is clear\nA REST API is needed.\n## What is missing\nDatabase choice.",
        "questions": [],
        "answers": [],
        "clarification_round": 0,
        "completeness_score": 0,
        "human_input": "",
        "user_review": None,
        "architecture_spec": None,
    }
    state.update(overrides)
    return state


def _text_response(text: str) -> SimpleNamespace:
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], stop_reason="end_turn")


def _mock_llm(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _text_response(text)
    return client

class TestParseQuestions:
    def _valid_json(self, n: int = 2) -> str:
        items = [
            {
                "id": f"q{i+1}",
                "question": f"Question {i+1}?",
                "type": "text",
                "options": None,
                "reason": f"Reason {i+1}",
            }
            for i in range(n)
        ]
        return json.dumps(items)

    def test_parses_valid_json_array(self):
        result = _parse_questions(self._valid_json(2))
        assert len(result) == 2
        assert result[0]["id"] == "q1"
        assert result[1]["question"] == "Question 2?"

    def test_strips_markdown_fences(self):
        raw = f"```json\n{self._valid_json(1)}\n```"
        result = _parse_questions(raw)
        assert len(result) == 1

    def test_caps_at_max_questions_per_round(self):
        result = _parse_questions(self._valid_json(MAX_QUESTIONS_PER_ROUND + 3))
        assert len(result) == MAX_QUESTIONS_PER_ROUND

    def test_returns_empty_on_invalid_json(self):
        assert _parse_questions("not valid json") == []

    def test_returns_empty_on_empty_string(self):
        assert _parse_questions("") == []

    def test_assigns_fallback_id_when_missing(self):
        raw = json.dumps([{"question": "Q?", "type": "text", "reason": "r"}])
        result = _parse_questions(raw)
        assert result[0]["id"] == "q1"

    def test_defaults_type_to_text_when_missing(self):
        raw = json.dumps([{"id": "q1", "question": "Q?", "reason": "r"}])
        result = _parse_questions(raw)
        assert result[0]["type"] == "text"

    def test_returns_empty_when_question_key_missing(self):
        raw = json.dumps([{"id": "q1", "type": "text", "reason": "r"}])
        result = _parse_questions(raw)
        assert result == []

    def test_preserves_options_for_multiple_choice(self):
        raw = json.dumps([{
            "id": "q1", "question": "Q?", "type": "multiple_choice",
            "options": ["A", "B", "C"], "reason": "r",
        }])
        result = _parse_questions(raw)
        assert result[0]["options"] == ["A", "B", "C"]

class TestNormaliseAnswer:
    def _yn_q(self) -> Question:
        return _make_question(qtype="yes_no", options=None)

    def _mc_q(self) -> Question:
        return _make_question(qtype="multiple_choice", options=["Alpha", "Beta", "Gamma"])

    def _text_q(self) -> Question:
        return _make_question(qtype="text", options=None)

    # yes_no
    def test_yes_no_y(self):
        assert _normalise_answer("y", self._yn_q()) == "yes"

    def test_yes_no_yes(self):
        assert _normalise_answer("YES", self._yn_q()) == "yes"

    def test_yes_no_1(self):
        assert _normalise_answer("1", self._yn_q()) == "yes"

    def test_yes_no_n(self):
        assert _normalise_answer("n", self._yn_q()) == "no"

    def test_yes_no_no(self):
        assert _normalise_answer("No", self._yn_q()) == "no"

    def test_yes_no_0(self):
        assert _normalise_answer("0", self._yn_q()) == "no"

    def test_yes_no_unrecognised_passthrough(self):
        assert _normalise_answer("maybe", self._yn_q()) == "maybe"

    # multiple_choice
    def test_mc_index_1_returns_first_option(self):
        assert _normalise_answer("1", self._mc_q()) == "Alpha"

    def test_mc_index_3_returns_third_option(self):
        assert _normalise_answer("3", self._mc_q()) == "Gamma"

    def test_mc_out_of_range_passthrough(self):
        assert _normalise_answer("99", self._mc_q()) == "99"

    def test_mc_non_integer_passthrough(self):
        assert _normalise_answer("Alpha", self._mc_q()) == "Alpha"

    # text
    def test_text_strips_whitespace(self):
        assert _normalise_answer("  hello  ", self._text_q()) == "hello"

    def test_text_passthrough(self):
        assert _normalise_answer("free text answer", self._text_q()) == "free text answer"

class TestFormatQuestionForDisplay:
    def test_yes_no_shows_y_n_prompt(self):
        q = _make_question(qtype="yes_no", options=None, reason="")
        output = _format_question_for_display(q, 1)
        assert "[y/n]" in output

    def test_multiple_choice_shows_numbered_options(self):
        q = _make_question(qtype="multiple_choice", options=["A", "B"])
        output = _format_question_for_display(q, 2)
        assert "1. A" in output
        assert "2. B" in output
        assert "[1-2]" in output

    def test_text_shows_free_answer_prompt(self):
        q = _make_question(qtype="text", options=None, reason="")
        output = _format_question_for_display(q, 1)
        assert "type your response" in output

    def test_reason_is_shown_when_present(self):
        q = _make_question(reason="Affects auth design")
        output = _format_question_for_display(q, 1)
        assert "Affects auth design" in output

    def test_reason_omitted_when_empty(self):
        q = _make_question(reason="")
        output = _format_question_for_display(q, 1)
        assert "Why:" not in output

    def test_question_number_in_output(self):
        q = _make_question()
        output = _format_question_for_display(q, 3)
        assert "Q3:" in output

class TestRenderUnderstanding:
    def test_contains_understanding_text(self):
        result = _render_understanding("My understanding here", [])
        assert "My understanding here" in result

    def test_shows_accept_reject_options(self):
        result = _render_understanding("understanding", [])
        assert "accept" in result.lower()
        assert "reject" in result.lower()
        assert "change" in result.lower()

    def test_shows_qa_history_when_answers_present(self):
        answers = [_make_answer(question="What DB?", answer="PostgreSQL")]
        result = _render_understanding("understanding", answers)
        assert "What DB?" in result
        assert "PostgreSQL" in result

    def test_no_qa_section_when_no_answers(self):
        result = _render_understanding("understanding", [])
        assert "CLARIFICATIONS YOU PROVIDED" not in result

    def test_multiple_answers_all_shown(self):
        answers = [
            _make_answer("q1", "Q1", "A1"),
            _make_answer("q2", "Q2", "A2"),
        ]
        result = _render_understanding("u", answers)
        assert "A1" in result
        assert "A2" in result

class TestGetInterruptValue:
    def test_returns_value_from_task_interrupts(self):
        interrupt_obj = SimpleNamespace(value="the prompt text")
        task = SimpleNamespace(interrupts=[interrupt_obj])
        state = SimpleNamespace(tasks=[task])
        assert _get_interrupt_value(state) == "the prompt text"

    def test_returns_empty_string_when_no_tasks(self):
        state = SimpleNamespace(tasks=[])
        assert _get_interrupt_value(state) == ""

    def test_returns_empty_string_when_no_interrupts(self):
        task = SimpleNamespace(interrupts=[])
        state = SimpleNamespace(tasks=[task])
        assert _get_interrupt_value(state) == ""

    def test_returns_first_interrupt_value(self):
        i1 = SimpleNamespace(value="first")
        i2 = SimpleNamespace(value="second")
        task = SimpleNamespace(interrupts=[i1, i2])
        state = SimpleNamespace(tasks=[task])
        assert _get_interrupt_value(state) == "first"

class TestAnalyseInput:
    def test_sets_understanding_from_llm(self):
        state = _base_state()
        mock_client = _mock_llm("## What is clear\nA REST API\n## What is missing\nAuth")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = analyse_input(state)

        assert "REST API" in result["understanding"]

    def test_resets_state_fields(self):
        state = _base_state(
            clarification_round=3,
            answers=[_make_answer()],
            completeness_score=5,
        )
        mock_client = _mock_llm("understanding text")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = analyse_input(state)

        assert result["clarification_round"] == 0
        assert result["answers"] == []
        assert result["completeness_score"] == 0
        assert result["user_review"] is None
        assert result["architecture_spec"] is None

    def test_raw_input_sent_to_llm(self):
        state = _base_state(raw_input="UNIQUE_RAW_INPUT_TEXT_XYZ")
        mock_client = _mock_llm("some understanding")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            analyse_input(state)

        call_args = mock_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "UNIQUE_RAW_INPUT_TEXT_XYZ" in user_msg

    def test_empty_llm_response_gives_empty_understanding(self):
        state = _base_state()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = SimpleNamespace(content=[])

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = analyse_input(state)

        assert result["understanding"] == ""

class TestGenerateQuestions:
    def _valid_questions_json(self, n: int = 2) -> str:
        return json.dumps([
            {"id": f"q{i+1}", "question": f"Q{i+1}?", "type": "text",
             "options": None, "reason": f"R{i+1}"}
            for i in range(n)
        ])

    def test_returns_parsed_questions(self):
        state = _base_state()
        mock_client = _mock_llm(self._valid_questions_json(2))

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = generate_questions(state)

        assert len(result["questions"]) == 2

    def test_prior_answers_included_in_prompt(self):
        state = _base_state(answers=[_make_answer(question="DB type?", answer="PostgreSQL")])
        captured = []
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = lambda **kw: (
            captured.append(kw["messages"][0]["content"]) or _text_response(self._valid_questions_json(1))
        )

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            generate_questions(state)

        assert "DB type?" in captured[0]
        assert "PostgreSQL" in captured[0]

    def test_understanding_sent_to_llm(self):
        state = _base_state(understanding="UNIQUE_UNDERSTANDING_ABC")
        captured = []
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = lambda **kw: (
            captured.append(kw["messages"][0]["content"]) or _text_response(self._valid_questions_json(1))
        )

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            generate_questions(state)

        assert "UNIQUE_UNDERSTANDING_ABC" in captured[0]

    def test_empty_list_on_bad_llm_output(self):
        state = _base_state()
        mock_client = _mock_llm("sorry I cannot generate questions right now")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = generate_questions(state)

        assert result["questions"] == []


class TestAskUser:
    def _make_state_with_questions(self, qs: List[Question]) -> ArchitectState:
        return _base_state(questions=qs, clarification_round=0)

    def test_interrupt_called_with_question_prompt(self):
        q = _make_question(qtype="text", options=None)
        state = self._make_state_with_questions([q])

        with patch("ArchitectAgent.interrupt", return_value="My answer") as mock_interrupt:
            ask_user(state)

        assert mock_interrupt.called
        prompt = mock_interrupt.call_args[0][0]
        assert q["question"] in prompt

    def test_single_answer_parsed_correctly(self):
        q = _make_question(qid="q1", qtype="text", options=None)
        state = self._make_state_with_questions([q])

        with patch("ArchitectAgent.interrupt", return_value="PostgreSQL"):
            result = ask_user(state)

        assert len(result["answers"]) == 1
        assert result["answers"][0]["answer"] == "PostgreSQL"
        assert result["answers"][0]["question_id"] == "q1"

    def test_multiple_answers_parsed_in_order(self):
        qs = [
            _make_question("q1", "DB?", "text", None),
            _make_question("q2", "Auth?", "yes_no", None),
        ]
        state = self._make_state_with_questions(qs)

        with patch("ArchitectAgent.interrupt", return_value="PostgreSQL\ny"):
            result = ask_user(state)

        assert result["answers"][0]["answer"] == "PostgreSQL"
        assert result["answers"][1]["answer"] == "yes"   # normalised

    def test_fewer_answers_than_questions_pads_empty(self):
        qs = [
            _make_question("q1", "Q1?", "text", None),
            _make_question("q2", "Q2?", "text", None),
        ]
        state = self._make_state_with_questions(qs)

        with patch("ArchitectAgent.interrupt", return_value="only one answer"):
            result = ask_user(state)

        assert len(result["answers"]) == 2
        assert result["answers"][1]["answer"] == ""

    def test_accumulates_on_top_of_prior_answers(self):
        prior = [_make_answer("q0", "Q0?", "prior")]
        q = _make_question("q1", "Q1?", "text", None)
        state = self._make_state_with_questions([q])
        state["answers"] = prior

        with patch("ArchitectAgent.interrupt", return_value="new answer"):
            result = ask_user(state)

        assert len(result["answers"]) == 2
        assert result["answers"][0]["answer"] == "prior"
        assert result["answers"][1]["answer"] == "new answer"

    def test_no_questions_returns_existing_answers(self):
        existing = [_make_answer()]
        state = _base_state(questions=[], answers=existing)

        # interrupt must NOT be called when there are no questions
        with patch("ArchitectAgent.interrupt") as mock_interrupt:
            result = ask_user(state)

        mock_interrupt.assert_not_called()
        assert result["answers"] == existing

    def test_human_input_cleared_after_processing(self):
        q = _make_question(qtype="text", options=None)
        state = self._make_state_with_questions([q])

        with patch("ArchitectAgent.interrupt", return_value="some answer"):
            result = ask_user(state)

        assert result["human_input"] == ""

class TestIncorporateAnswers:
    def test_updates_understanding_from_llm(self):
        state = _base_state(
            understanding="old understanding",
            answers=[_make_answer(answer="PostgreSQL")],
        )
        mock_client = _mock_llm("updated understanding with PostgreSQL info")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = incorporate_answers(state)

        assert "PostgreSQL" in result["understanding"]

    def test_increments_clarification_round(self):
        state = _base_state(clarification_round=1, answers=[_make_answer()])
        mock_client = _mock_llm("updated")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = incorporate_answers(state)

        assert result["clarification_round"] == 2

    def test_answers_sent_to_llm(self):
        state = _base_state(answers=[_make_answer(question="DB?", answer="UNIQUE_ANSWER_XYZ")])
        captured = []
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = lambda **kw: (
            captured.append(kw["messages"][0]["content"]) or _text_response("updated")
        )

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            incorporate_answers(state)

        assert "UNIQUE_ANSWER_XYZ" in captured[0]

    def test_empty_answers_still_calls_llm(self):
        state = _base_state(answers=[])
        mock_client = _mock_llm("updated with no answers")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = incorporate_answers(state)

        assert "understanding" in result

class TestCheckCompleteness:
    def test_parses_score_from_llm(self):
        state = _base_state(understanding="detailed understanding")
        mock_client = _mock_llm("SCORE: 8\nVERDICT: Requirements are sufficient.")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = check_completeness(state)

        assert result["completeness_score"] == 8

    def test_clamps_score_above_10(self):
        state = _base_state()
        mock_client = _mock_llm("SCORE: 15\nVERDICT: Perfect.")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = check_completeness(state)

        assert result["completeness_score"] == 10

    def test_clamps_score_below_0(self):
        state = _base_state()
        mock_client = _mock_llm("SCORE: -3\nVERDICT: Empty.")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = check_completeness(state)

        assert result["completeness_score"] == 0

    def test_returns_0_when_no_score_line(self):
        state = _base_state()
        mock_client = _mock_llm("VERDICT: Good enough.\nNo score here.")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = check_completeness(state)

        assert result["completeness_score"] == 0

    def test_case_insensitive_score_parsing(self):
        state = _base_state()
        mock_client = _mock_llm("score: 7\nVERDICT: Acceptable.")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = check_completeness(state)

        assert result["completeness_score"] == 7

class TestPresentSummary:
    def test_accept_decision(self):
        state = _base_state(understanding="good understanding")

        with patch("ArchitectAgent.interrupt", return_value="accept"):
            result = present_summary(state)

        assert result["user_review"]["decision"] == "accepted"
        assert result["user_review"]["change_notes"] == ""

    def test_accept_case_insensitive(self):
        state = _base_state(understanding="u")

        with patch("ArchitectAgent.interrupt", return_value="ACCEPT"):
            result = present_summary(state)

        assert result["user_review"]["decision"] == "accepted"

    def test_change_decision_extracts_notes(self):
        state = _base_state(understanding="u")

        with patch("ArchitectAgent.interrupt", return_value="change: add caching layer"):
            result = present_summary(state)

        assert result["user_review"]["decision"] == "change"
        assert "add caching layer" in result["user_review"]["change_notes"]

    def test_change_without_colon_captures_full_text(self):
        state = _base_state(understanding="u")

        with patch("ArchitectAgent.interrupt", return_value="change add Redis"):
            result = present_summary(state)

        assert result["user_review"]["decision"] == "change"

    def test_reject_decision(self):
        state = _base_state(understanding="u")

        with patch("ArchitectAgent.interrupt", return_value="reject"):
            result = present_summary(state)

        assert result["user_review"]["decision"] == "rejected"

    def test_unrecognised_input_treated_as_rejected(self):
        state = _base_state(understanding="u")

        with patch("ArchitectAgent.interrupt", return_value="hmm not sure"):
            result = present_summary(state)

        assert result["user_review"]["decision"] == "rejected"

    def test_interrupt_called_with_rendered_understanding(self):
        state = _base_state(understanding="UNIQUE_UNDERSTANDING_STRING_XYZ")

        with patch("ArchitectAgent.interrupt", return_value="accept") as mock_int:
            present_summary(state)

        prompt = mock_int.call_args[0][0]
        assert "UNIQUE_UNDERSTANDING_STRING_XYZ" in prompt

    def test_human_input_cleared(self):
        state = _base_state(understanding="u")

        with patch("ArchitectAgent.interrupt", return_value="accept"):
            result = present_summary(state)

        assert result["human_input"] == ""

class TestHandleUserReview:
    def test_change_injects_change_notes_as_answer(self):
        review: RequirementsReview = {
            "decision": "change",
            "change_notes": "Please add Redis caching",
        }
        state = _base_state(user_review=review, answers=[])

        result = handle_user_review(state)

        assert len(result["answers"]) == 1
        injected = result["answers"][0]
        assert injected["question_id"] == "user-review-change"
        assert "Redis caching" in injected["answer"]

    def test_change_appends_to_existing_answers(self):
        review: RequirementsReview = {"decision": "change", "change_notes": "Add auth"}
        prior = [_make_answer("q1", "Q?", "PostgreSQL")]
        state = _base_state(user_review=review, answers=prior)

        result = handle_user_review(state)

        assert len(result["answers"]) == 2
        assert result["answers"][0]["answer"] == "PostgreSQL"

    def test_accepted_returns_empty_dict(self):
        review: RequirementsReview = {"decision": "accepted", "change_notes": ""}
        state = _base_state(user_review=review)

        result = handle_user_review(state)

        assert result == {}

    def test_rejected_returns_empty_dict(self):
        review: RequirementsReview = {"decision": "rejected", "change_notes": ""}
        state = _base_state(user_review=review)

        result = handle_user_review(state)

        assert result == {}

    def test_change_with_empty_notes_returns_empty_dict(self):
        """If change_notes is empty there's nothing to inject."""
        review: RequirementsReview = {"decision": "change", "change_notes": ""}
        state = _base_state(user_review=review)

        result = handle_user_review(state)

        assert result == {}

    def test_no_review_returns_empty_dict(self):
        state = _base_state(user_review=None)
        assert handle_user_review(state) == {}

class TestProduceArchitectureSpec:
    def test_spec_fields_all_populated(self):
        state = _base_state(
            understanding="## Initial project name suggestion: TaskMaster\nDetails here.",
            answers=[_make_answer(question="DB?", answer="PostgreSQL")],
        )
        mock_client = _mock_llm("# TaskMaster — Architecture Specification\n## Overview\nA task API.")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = produce_architecture_spec(state)

        spec = result["architecture_spec"]
        assert spec["project_id"] == PROJECT_ID
        assert "spec_text" in spec
        assert "created_at" in spec
        assert "project_name" in spec

    def test_project_name_extracted_from_understanding(self):
        state = _base_state(
            understanding="## Initial project name suggestion: MyAwesomeApp\nSome details.",
        )
        mock_client = _mock_llm("# MyAwesomeApp — Architecture Specification")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = produce_architecture_spec(state)

        assert result["architecture_spec"]["project_name"] == "MyAwesomeApp"

    def test_project_id_used_as_fallback_name(self):
        """If no project name line in understanding, project_id is used."""
        state = _base_state(understanding="## What is clear\nA REST API")
        mock_client = _mock_llm("# REST API — Architecture Specification")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = produce_architecture_spec(state)

        assert result["architecture_spec"]["project_name"] == PROJECT_ID

    def test_answers_included_in_llm_prompt(self):
        state = _base_state(
            answers=[_make_answer(question="DB?", answer="UNIQUE_DB_CHOICE_XYZ")],
        )
        captured = []
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = lambda **kw: (
            captured.append(kw["messages"][0]["content"]) or _text_response("# Spec")
        )

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            produce_architecture_spec(state)

        assert "UNIQUE_DB_CHOICE_XYZ" in captured[0]

    def test_completed_at_is_iso8601(self):
        state = _base_state()
        mock_client = _mock_llm("# Spec content")

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = produce_architecture_spec(state)

        ts = result["architecture_spec"]["created_at"]
        assert ts.endswith("Z")
        datetime.fromisoformat(ts.rstrip("Z"))

    def test_spec_text_from_llm(self):
        state = _base_state()
        expected_spec = "# My Project — Architecture Specification\n## Overview\nDetails."
        mock_client = _mock_llm(expected_spec)

        with patch("ArchitectAgent.Anthropic", return_value=mock_client):
            result = produce_architecture_spec(state)

        assert result["architecture_spec"]["spec_text"] == expected_spec

class TestRouteAfterCompleteness:
    def _state(self, score: int, rounds: int) -> dict:
        return {"completeness_score": score, "clarification_round": rounds}

    def test_low_score_first_round_routes_to_generate_questions(self):
        assert route_after_completeness(self._state(score=4, rounds=0)) == "generate_questions"

    def test_low_score_mid_rounds_routes_to_generate_questions(self):
        assert route_after_completeness(self._state(score=5, rounds=2)) == "generate_questions"

    def test_low_score_at_max_rounds_routes_to_present_summary(self):
        result = route_after_completeness(
            self._state(score=4, rounds=MAX_CLARIFICATION_ROUNDS)
        )
        assert result == "present_summary"

    def test_score_at_threshold_routes_to_present_summary(self):
        assert route_after_completeness(
            self._state(score=COMPLETENESS_THRESHOLD, rounds=0)
        ) == "present_summary"

    def test_score_above_threshold_routes_to_present_summary(self):
        assert route_after_completeness(self._state(score=10, rounds=0)) == "present_summary"

    def test_score_one_below_threshold_routes_to_generate_questions(self):
        assert route_after_completeness(
            self._state(score=COMPLETENESS_THRESHOLD - 1, rounds=0)
        ) == "generate_questions"

class TestRouteAfterReview:
    def _state(self, decision: str) -> dict:
        return {"user_review": {"decision": decision, "change_notes": ""}}

    def test_accepted_routes_to_produce_spec(self):
        assert route_after_review(self._state("accepted")) == "produce_architecture_spec"

    def test_change_routes_to_incorporate_answers(self):
        assert route_after_review(self._state("change")) == "incorporate_answers"

    def test_rejected_routes_to_end(self):
        assert route_after_review(self._state("rejected")) == "__end__"

    def test_missing_review_defaults_to_rejected(self):
        assert route_after_review({"user_review": None}) == "__end__"

    def test_no_user_review_key_defaults_to_rejected(self):
        assert route_after_review({}) == "__end__"

class TestBuildArchitectGraph:
    def test_all_nodes_registered(self):
        graph = build_architect_graph()
        expected = {
            "analyse_input", "generate_questions", "ask_user",
            "incorporate_answers", "check_completeness",
            "present_summary", "handle_user_review", "produce_architecture_spec",
        }
        assert expected.issubset(set(graph.nodes.keys()))

    def test_graph_compiles_without_error(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_architect_graph()
        compiled = graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["ask_user", "present_summary"],
        )
        assert compiled is not None


class TestCreateArchitectAgent:
    def test_returns_compiled_agent(self):
        agent = create_architect_agent()
        assert agent is not None

    def test_accepts_custom_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver
        cp = MemorySaver()
        agent = create_architect_agent(checkpointer=cp)
        assert agent is not None
