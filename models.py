from dataclasses import dataclass, field
from typing import Literal, List, Dict, Optional, TypedDict, Any

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
        
class WorkerReport(TypedDict):
    task_id: str
    worker_id: str
    status: Literal["completed", "completed_with_issues", "blocked", "failed"]
    summary: str
    touched_files: List[str]
    test_file_path: Optional[str]
    review_score: int
    review_feedback: str
    blockers: List[str]
    suggestions: List[str]
    execution_notes: str
    completed_at: str 

class TaskRecord(TypedDict):
    task_id: str
    project_id: str
    desc: str
    status: Literal["Open", "InProgress", "PendingReview", "ReworkRequired", "Closed"]
    worker_id: str
    dependency_list: List[str]
    retry_count: int
    report: Optional[Dict]
    review_notes: str  

class BatchReview(TypedDict):
    """Result of reviewing a completed batch of tasks together."""
    score: int
    verdict: Literal["pass", "fail"]
    issues: List[str]
    suggestions: List[str]
    summary: str

class ProjectReport(TypedDict):
    project_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    blocked_tasks: int
    all_touched_files: List[str]
    batch_review_scores: List[int]
    final_status: Literal["success", "partial", "failed"]
    summary: str
    completed_at: str

class Question(TypedDict):
    id: str
    question: str
    type: Literal["yes_no", "multiple_choice", "text"]
    options: Optional[List[str]]
    reason: str
 
class UserAnswer(TypedDict):
    question_id: str
    question: str
    answer: str
 
class RequirementsReview(TypedDict):
    decision: Literal["accepted", "change", "rejected"]
    change_notes: str
 
class ArchitectureSpec(TypedDict):
    project_id: str
    project_name: str
    spec_text: str
    created_at: str

class RunRecord(TypedDict):
    run_id: str
    project_id: str
    project_name: str
    raw_input: str
    architecture_spec: str
    final_status: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    batch_review_scores: List[int]
    avg_review_score: float
    total_duration_seconds: float
    started_at: str
    completed_at: str
 
 
class AgentEvent(TypedDict):
    event_id: str
    run_id: str
    project_id: str
    agent: str
    node: str
    task_id: Optional[str]
    duration_seconds: float
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]
    timestamp: str
 
 
class PromptVersion(TypedDict):
    prompt_id: str
    agent: str
    node: str
    version: int
    content: str
    avg_score: float
    run_count: int
    recorded_at: str
 
 
class WorkerScore(TypedDict):
    score_id: str
    run_id: str
    project_id: str
    task_id: str
    task_desc: str
    review_score: int
    rework_count: int
    status: str
    prompt_version_id: str
    completed_at: str