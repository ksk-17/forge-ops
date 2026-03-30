import json
import time
import fcntl
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

lock_file_path = "projects/{project_id}/locks.json"

def _load_locks(lock_abs_file_path: str) -> Dict:
    with open(lock_abs_file_path, "r") as f:
        return json.load(f)


def _save_locks(lock_abs_file_path: str, locks_json: Dict) -> None:
    with open(lock_abs_file_path, "w") as f:
        json.dump(locks_json, f, indent=2)


def check_lock(locks_json: Dict, file_path: str) -> Optional[Dict]:
    return locks_json.get(file_path, None)


def acquire_lock(project_id: str, worker_id: str, file_path: str) -> Optional[str]:
    lock_abs_file_path = lock_file_path.format(project_id=project_id)

    with open(lock_abs_file_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            locks_json = json.load(f)
            lock_info = check_lock(locks_json, file_path)
            if lock_info is not None:
                return (
                    f"File is already locked by worker '{lock_info['worker_id']}' "
                    f"since {lock_info['time']:.2f}"
                )
            locks_json[file_path] = {
                "worker_id": worker_id,
                "time": time.time(),
            }
            f.seek(0)
            f.truncate()
            json.dump(locks_json, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    return None  # success


def remove_lock(project_id: str, worker_id: str, file_path: str) -> Optional[str]:
    lock_abs_file_path = lock_file_path.format(project_id=project_id)

    with open(lock_abs_file_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            locks_json = json.load(f)
            lock_info = check_lock(locks_json, file_path)

            if lock_info is None:
                return f"No lock exists for file '{file_path}'"

            if lock_info["worker_id"] != worker_id:
                return (
                    f"Worker '{worker_id}' cannot release a lock held by "
                    f"'{lock_info['worker_id']}'"
                )

            del locks_json[file_path]
            f.seek(0)
            f.truncate()
            json.dump(locks_json, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    return None  # success


def format_content_with_line_numbers(text: str) -> Dict[int, str]:
    lines = text.splitlines()
    return {i + 1: lines[i] for i in range(len(lines))}