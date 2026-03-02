from dataclasses import dataclass
from typing import Literal, Optional
from pydantic import BaseModel

Ops = Literal["insert_after", "insert_before", "replace", "delete"]

@dataclass
class Edit(BaseModel):
    op: Ops
    identifier: str
    value: str
    occurence: Optional[int] = None

TASK_STATUS = Literal["Not Assigned", "Assigned", "In Progress", "Pending Review", "Done"]

class Task(BaseModel):
    task_id: str
    desc: str
    status: TASK_STATUS
    worker_id: str

class ReviewClass(BaseModel):
    satisfied: bool
    comments: str

# helper function to render the class
def render_task(task: Task):
    return(
        f"Description:\n{task.desc}"
    )

def render_review(review: ReviewClass) -> str:
    return (
        f"Review Satisfied:\n{review.satisfied}\n\n"
        f"Comments:\n{review.comments}"
    )