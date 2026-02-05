from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import threading
import uuid
import time

PROJECTS_PATH = Path("projects")

@dataclass
class LockInfo:
    worker_id: str
    acquired_at: float
    ttl_seconds: int

# File manager
class FileManager:
    def __init__(self, base_path: Path = PROJECTS_PATH):
        self.base_path = base_path
        self.locks: Dict[Tuple[str, str], LockInfo] = {}
        self.mu = threading.Lock()
        self.base_path.mkdir(parent=True, exist_ok=True)

    def create_project(self) -> str:
        project_id = str(uuid.uuid4())
        project_path = (self.base_path / project_id)
        project_path.mkdir(parents=True, exist_ok=True)
        return project_id
        
    def create_file(self, project_id: str, file_path: str):
        p = self.abs_path(project_id, file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
        return p 

    def check_lock(self, project_id: str, file_path: str) -> Optional[str]:
        key = self.lock_key(project_id, file_path)
        with self.mu:
            self.cleanup_expired_locked_unlocked(key)
            info = self.locks.get(key)
            return info.worker_id if info else None
    
    def acquire_lock(self, project_id: str, file_path: str, worker_id: str, ttl_seconds: int = 300) -> bool:
        key = self.lock_key(project_id, file_path)
        now = time.time()
        with self.mu:
            self.cleanup_expired_locked_unlocked(key)
            if key in self.locks:
                return False
            self.locks[key] = LockInfo(worker_id=worker_id, acquired_at=now, ttl_seconds=ttl_seconds)
            return True

    def release_lock(self, project_id: str, file_path: str, worker_id: str) -> bool:
        key = self.lock_key(project_id, file_path)
        with self.mu():
            info = self.locks.get(key)
            if not info:
                return True
            if info.worker_id != worker_id:
                return False
            del self.locks[key]
            return True
        
    def renew_lock(self, project_id: str, file_path: str, worker_id: str, ttl_seconds: int = 300) -> bool:
        key = self.lock_key(project_id, file_path)
        now = time.time()
        with self.mu():
            info = self.locks.get(key)
            if not info or info.worker_id != worker_id:
                return False
            self.locks[key] = LockInfo(worker_id, acquired_at=now, ttl_seconds=ttl_seconds)
            return True

    # -------------- helper functions ------------------

    def abs_path(self, project_id: str, file_path: str) -> Path:
        # normalizes to remove duplicates
        p = (self.base_path / project_id / file_path).resolve()
        project_path = (self.base_path / project_id).resolve()
        if not str(p).startswith(str(project_path)):
            raise ValueError("file_path escapes the project_path")
        return p
    
    def lock_key(self, project_id: str, file_path: str) -> Tuple[str, str]:
        # normalize the path into a stable key (relative path)
        rel = Path(file_path).as_posix()
        return (project_id, rel)
    
    def cleanup_expired_locked_unlocked(self, key: Tuple[str, str]) -> None:
        info = self.locks.get(key)
        if not info:
            return
        if time.time() - info.acquired_at() > info.ttl_seconds:
            del self.locks[key]