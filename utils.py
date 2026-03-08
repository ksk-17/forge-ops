import json
import time
from typing import Dict

lock_file_path = "projects/{project_id}/locks.json"

def check_lock(locks_json, file_path) -> Dict[str, str] | None:
    return locks_json.get(file_path, None)

def acquire_lock(project_id, worker_id, file_path):
    lock_abs_file_path = lock_file_path.format(project_id=project_id)
    with open(lock_abs_file_path, 'r') as f:
        locks_json = json.load(f)
    lock_info = check_lock(locks_json, file_path)
    if lock_info:
        return f"The file is locked by the following agent: {lock_info['worker_id']} at {lock_info['time']}"
    locks_json[file_path] = {
        'worker_id': worker_id,
        'time': time.time()
    }
    with open(lock_abs_file_path, 'w') as f:
        json.dump(locks_json, f, indent=2)

def remove_lock(project_id, worker_id, file_path):
    lock_abs_file_path = lock_file_path.format(project_id=project_id)
    with open(lock_abs_file_path, "r") as f:
        locks_json = json.load(f)
    lock_info = check_lock(locks_json, file_path)
    if lock_info['worker_id'] != worker_id:
        return "The worker doesn\'t have the lock for the file"
    del locks_json[file_path]
    with open(lock_abs_file_path, 'w') as f:
        json.dump(locks_json, f, indent=2)

def format_content_with_line_numbers(text: str) -> Dict[int, str]:
    lines = text.splitlines()
    return {i+1: lines[i] for i in range(len(lines))}