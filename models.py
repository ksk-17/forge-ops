from dataclasses import dataclass, field
from typing import Literal, List, Dict, Optional

@dataclass
class Edit:
    operation: Literal["insert", "update", "delete"]
    start_line: int
    end_line: int | None
    new_value: str = ""
    method_description_changes: Dict[str, str] = field(default_factory=dict)

@dataclass
class Task:
    task_id: str
    project_id: str
    desc: str
    status: Literal["Open", "InProgress", "PendingReview", "ReworkRequired", "Closed"]
    worker_id: str
    dependency_list: List[str]