from dataclasses import dataclass, field
from typing import Literal, List, Dict, Optional

@dataclass
class Edit:
    operation: Literal["insert", "update", "delete"]
    start_line: int
    end_line: Optional[int]
    new_value: str = ""
    method_description_changes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.operation == "delete" and self.end_line is None:
            raise ValueError("end_line is required for delete operations")
        if self.operation == "insert" and self.end_line is not None:
            # end_line is irrelevant for insert — enforce clean usage
            self.end_line = None
        if self.start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {self.start_line}")
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )

@dataclass
class Task:
    task_id: str
    project_id: str
    desc: str
    status: Literal["Open", "InProgress", "PendingReview", "ReworkRequired", "Closed"]
    worker_id: str
    dependency_list: List[str]

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if not self.project_id:
            raise ValueError("project_id cannot be empty")