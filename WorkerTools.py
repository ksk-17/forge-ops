from langchain.tools import tool
from utils import format_content_with_line_numbers, acquire_lock, remove_lock
from models import Edit

import os
import json
import time
from pathlib import Path
from typing import List, Dict

project_info_path = "projects/{project_id}/project_info.json"
project_file_path = "projects/{project_id}/{file_path}"

def get_project_schema(project_id: str) -> Dict[str, Dict[str, str]]:
    project_info_abs_path = project_info_path.format(project_id=project_id)
    with open(project_info_abs_path, "r") as f:
        content = json.load(f)
    return content

def update_file_schema(project_id: str, file_path: str, **new_file_schema):
    project_schema = get_project_schema(project_id)
    file_schema = project_schema.get(file_path, {})
    for k, v in new_file_schema.items():
        file_schema[k] = v
    project_schema[file_path] = file_schema
    project_info_abs_path = project_info_path.format(project_id=project_id)
    with open(project_info_abs_path, mode="w") as f:
        json.dump(project_schema, f, indent=2)

def create_file(project_id: str, file_path: str, file_description: str, worker_id: str):
    file_abs_path = Path(project_file_path.format(project_id = project_id, file_path = file_path))
    file_abs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_abs_path, "w") as f:
        f.write("")
    new_file_schema = {
        'file_name': file_path.split('/')[-1],
        'description': file_description,
        'methods': {},
        'created_at': time.time(),
        'created_by': worker_id,
        'last_modified_at': time.time(),
        'modified_by': worker_id
    }
    update_file_schema(project_id, file_path, **new_file_schema)

def read_file(project_id: str, file_paths: List[str]):
    content = {}
    for file_path in file_paths:
        file_abs_path = project_file_path.format(project_id = project_id, file_path = file_path)
        with open(file_abs_path, "r") as f:
            file_content = f.read()
            content[file_path] = format_content_with_line_numbers(file_content)
    return content

def write_file(project_id: str, worker_id: str, file_path: str, edits: List[Edit]):
    lock_info = acquire_lock(project_id, worker_id, file_path)

    if lock_info is not None:
        return lock_info
    file_content = read_file(project_id, [file_path])[file_path]
    for edit in edits:
        if edit.operation == "insert":
            new_file_content = {}
            new_line_count = 1
            for ind, value in enumerate(file_content.values()):
                if (ind + 1 )== edit.start_line:
                    for new_value in edit.new_value.split('\n'):
                        new_file_content[new_line_count] = new_value
                        new_line_count += 1
                new_file_content[new_line_count] = value
                new_line_count += 1
            file_content = new_file_content
        elif edit.operation == "update":
            start_line = edit.start_line
            for ind, new_value in enumerate(edit.new_value.split('\n')):
                file_content[start_line + ind] = new_value
        elif edit.operation == "delete":
            new_file_content = {}
            new_line_count = 1
            for ind, value in enumerate(file_content.values()):
                if (ind + 1) >= edit.start_line and (ind + 1) <= edit.end_line:
                    continue
                new_file_content[new_line_count] = value
                new_line_count += 1
            file_content = new_file_content
        else:
            return f"Invalid operation: {edit.operation}"
        
    file_abs_path = project_file_path.format(project_id = project_id, file_path = file_path)
    content_str = '\n'.join([value for value in file_content.values()])
    with open(file_abs_path, mode="w") as f:
        f.write(content_str)

    new_file_schema = {
        'methods': {},
        'last_modified_at': time.time(),
        'modified_by': worker_id
    }
    update_file_schema(project_id, file_path, **new_file_schema)
    remove_lock(project_id, worker_id, file_path)

