import os
import pytest
import time
from pathlib import Path

from FileEditor import FileEditor


class SimpleEdit:
    def __init__(self, op, identifier, value, occurence=None):
        self.op = op
        self.identifier = identifier
        self.value = value
        self.occurence = occurence


def test_find_all_simple():
    content = "ababa"
    positions = FileEditor.find_all(content, "aba")
    # current implementation advances by len(key), so overlapping matches are not returned
    assert positions == [0]


def test_atomic_write_and_find_all(tmp_path):
    p = tmp_path / "f.txt"
    FileEditor.atomic_write(p, "hello\n")
    assert p.read_text(encoding="utf-8") == "hello\n"


def test_apply_patches_replace_and_insert(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("Line1\nTARGET\nLine3\n", encoding="utf-8")

    edits = [
        SimpleEdit(op="replace", identifier="TARGET", value="REPLACED"),
        SimpleEdit(op="insert_after", identifier="REPLACED", value="\nAFTER"),
    ]

    FileEditor.apply_patches(str(p), edits)
    text = p.read_text(encoding="utf-8")
    assert "REPLACED" in text
    assert "AFTER" in text


def test_apply_patches_delete_and_insert_before(tmp_path):
    p = tmp_path / "doc2.txt"
    p.write_text("HEAD\nREMOVE_ME\nTAIL\n", encoding="utf-8")

    edits = [
        SimpleEdit(op="delete", identifier="REMOVE_ME", value=""),
        SimpleEdit(op="insert_before", identifier="TAIL", value="NEWLINE\n"),
    ]

    FileEditor.apply_patches(str(p), edits)
    text = p.read_text(encoding="utf-8")
    assert "REMOVE_ME" not in text
    assert "NEWLINE" in text


def test_apply_patches_multiple_occurrence_disambiguation(tmp_path):
    p = tmp_path / "multi.txt"
    p.write_text("X A X B X\n", encoding="utf-8")

    # Replace the second occurrence (index 1)
    edits = [SimpleEdit(op="replace", identifier="X", value="Y", occurence=1)]
    FileEditor.apply_patches(str(p), edits)
    text = p.read_text(encoding="utf-8")
    # ensure one X replaced by Y
    assert text.count("Y") == 1


def test_apply_patches_errors(tmp_path):
    p = tmp_path / "err.txt"
    p.write_text("AAA\n", encoding="utf-8")

    # empty identifier
    with pytest.raises(ValueError):
        FileEditor.apply_patches(str(p), [SimpleEdit(op="replace", identifier="", value="v")])

    # identifier not found
    with pytest.raises(ValueError):
        FileEditor.apply_patches(str(p), [SimpleEdit(op="replace", identifier="NOPE", value="v")])

    # ambiguous match without occurence
    p.write_text("A B A\n", encoding="utf-8")
    with pytest.raises(ValueError):
        FileEditor.apply_patches(str(p), [SimpleEdit(op="replace", identifier="A", value="v")])

    # unsupported op
    with pytest.raises(ValueError):
        FileEditor.apply_patches(str(p), [SimpleEdit(op="unknown", identifier="A", value="v")])
