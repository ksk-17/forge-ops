from typing import Dict, Optional
from dataclasses import dataclass, field
from FileEditor import FileEditor
from FileManager import FileManager
from models import Task
from threading import Lock

@dataclass
class SharedState:
    project_id: Optional[str] = None
    tasks: Dict[str, Task] = field(default_factory=dict)
    workers: Dict[str, str] = field(default_factory=dict)
    mu: Lock = field(default_factory=Lock)

    file_manager: FileManager = field(default_factory=FileManager)
    file_editor: FileEditor = field(default_factory=FileEditor)

    def get_task(self, task_id: str) -> Optional[Task]:
        with self.mu:
            return self.tasks.get(task_id)
        
    def update_task_status(self, task_id: str, task_status: str) -> None:
        with self.mu:
            if task_id not in self.tasks:
                raise KeyError(f"Task {task_id} not found")
            self.tasks[task_id].status = task_status
        