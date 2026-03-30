import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

from utils import format_content_with_line_numbers, acquire_lock, remove_lock
from models import Edit

logger = logging.getLogger(__name__)

project_info_path = "projects/{project_id}/project_info.json"
project_file_path = "projects/{project_id}/{file_path}"


def get_project_schema(project_id: str) -> Dict[str, Dict]:
    project_info_abs_path = project_info_path.format(project_id=project_id)
    with open(project_info_abs_path, "r") as f:
        return json.load(f)

def update_file_schema(project_id: str, file_path: str, **new_file_schema) -> None:
    project_schema = get_project_schema(project_id)
    file_schema = project_schema.get(file_path, {})
    file_schema.update(new_file_schema)
    project_schema[file_path] = file_schema
    project_info_abs_path = project_info_path.format(project_id=project_id)
    with open(project_info_abs_path, mode="w") as f:
        json.dump(project_schema, f, indent=2)

def create_file(
    project_id: str,
    file_path: str,
    file_description: str,
    worker_id: str,
) -> Optional[str]:
    file_abs_path = Path(
        project_file_path.format(project_id=project_id, file_path=file_path)
    )
    if file_abs_path.exists():
        return f"File '{file_path}' already exists in project '{project_id}'"

    try:
        file_abs_path.parent.mkdir(parents=True, exist_ok=True)
        file_abs_path.write_text("")
    except OSError as e:
        return f"Failed to create file '{file_path}': {e}"

    now = time.time()
    new_file_schema = {
        "file_name": file_path.split("/")[-1],
        "description": file_description,
        "methods": {},
        "created_at": now,
        "created_by": worker_id,
        "last_modified_at": now,
        "modified_by": worker_id,
    }
    update_file_schema(project_id, file_path, **new_file_schema)
    return None

def read_file(project_id: str, file_paths: List[str]) -> Dict[str, Dict[int, str]]:
    content: Dict[str, Dict[int, str]] = {}
    for fp in file_paths:
        file_abs_path = project_file_path.format(project_id=project_id, file_path=fp)
        if not os.path.exists(file_abs_path):
            raise FileNotFoundError(
                f"File '{fp}' not found in project '{project_id}'"
            )
        with open(file_abs_path, "r") as f:
            content[fp] = format_content_with_line_numbers(f.read())
    return content

def write_file(
    project_id: str,
    worker_id: str,
    file_path: str,
    edits: List[Edit],
) -> Optional[str]:
    # Acquire lock
    lock_error = acquire_lock(project_id, worker_id, file_path)
    if lock_error is not None:
        return lock_error

    try:
        # Load file
        try:
            file_content = read_file(project_id, [file_path])[file_path]
        except FileNotFoundError as e:
            return str(e)

        # Apply edits sequentially
        for edit in edits:
            result = _apply_edit(file_content, edit)
            if isinstance(result, str):
                # _apply_edit returns an error string on failure
                return result
            file_content = result

        # Write result back to disk
        file_abs_path = project_file_path.format(
            project_id=project_id, file_path=file_path
        )
        content_str = "\n".join(file_content.values())
        try:
            with open(file_abs_path, mode="w") as f:
                f.write(content_str)
        except OSError as e:
            return f"Failed to write file '{file_path}': {e}"

        # Update schema (methods, timestamps)
        merged_method_changes: Dict[str, str] = {}
        for edit in edits:
            merged_method_changes.update(edit.method_description_changes)

        now = time.time()
        schema_update: Dict = {
            "last_modified_at": now,
            "modified_by": worker_id,
        }
        if merged_method_changes:
            # Fetch current methods and apply the delta.
            project_schema = get_project_schema(project_id)
            current_methods = project_schema.get(file_path, {}).get("methods", {})
            current_methods.update(merged_method_changes)
            schema_update["methods"] = current_methods

        update_file_schema(project_id, file_path, **schema_update)

    finally:
        # Always release the lock — even if an error caused an early return.
        release_error = remove_lock(project_id, worker_id, file_path)
        if release_error:
            logger.error(
                "Failed to release lock for '%s' held by '%s': %s",
                file_path,
                worker_id,
                release_error,
            )

    return None

def _apply_edit(
    file_content: Dict[int, str], edit: Edit
) -> "Dict[int, str] | str":
    total_lines = len(file_content)

    if edit.operation == "insert":
        # Insert new_value lines *before* start_line.
        # start_line == total_lines + 1 means append at end.
        if edit.start_line > total_lines + 1:
            return (
                f"insert: start_line {edit.start_line} is beyond the end of the file "
                f"({total_lines} lines)"
            )
        new_file_content: Dict[int, str] = {}
        new_line_count = 1
        inserted = False
        for ind, value in enumerate(file_content.values()):
            if (ind + 1) == edit.start_line:
                for new_line in edit.new_value.split("\n"):
                    new_file_content[new_line_count] = new_line
                    new_line_count += 1
                inserted = True
            new_file_content[new_line_count] = value
            new_line_count += 1
        if not inserted:
            # start_line is past the last line — append
            for new_line in edit.new_value.split("\n"):
                new_file_content[new_line_count] = new_line
                new_line_count += 1
        return new_file_content

    elif edit.operation == "update":
        if edit.start_line > total_lines:
            return (
                f"update: start_line {edit.start_line} is beyond the end of the file "
                f"({total_lines} lines)"
            )
        new_lines = edit.new_value.split("\n")
        # Validate that the replacement doesn't silently overflow past end_line.
        if edit.end_line is not None:
            expected_line_count = edit.end_line - edit.start_line + 1
            if len(new_lines) > expected_line_count:
                return (
                    f"update: new_value has {len(new_lines)} lines but the range "
                    f"{edit.start_line}–{edit.end_line} covers only "
                    f"{expected_line_count} line(s). Use a delete+insert pair to "
                    f"replace a range with a different number of lines."
                )
        # Clone the dict and overwrite the target lines in place.
        new_file_content = dict(file_content)
        for ind, new_line in enumerate(new_lines):
            target = edit.start_line + ind
            if target > total_lines:
                return (
                    f"update: replacement line {target} is beyond the end of the "
                    f"file ({total_lines} lines)"
                )
            new_file_content[target] = new_line
        return new_file_content

    elif edit.operation == "delete":
        # edit.end_line is guaranteed non-None by Edit.__post_init__
        if edit.start_line > total_lines:
            return (
                f"delete: start_line {edit.start_line} is beyond the end of the file "
                f"({total_lines} lines)"
            )
        if edit.end_line > total_lines:
            return (
                f"delete: end_line {edit.end_line} is beyond the end of the file "
                f"({total_lines} lines)"
            )
        new_file_content = {}
        new_line_count = 1
        for ind, value in enumerate(file_content.values()):
            line_no = ind + 1
            if edit.start_line <= line_no <= edit.end_line:
                continue
            new_file_content[new_line_count] = value
            new_line_count += 1
        return new_file_content

    else:
        return f"Unknown operation '{edit.operation}'. Must be insert, update, or delete."