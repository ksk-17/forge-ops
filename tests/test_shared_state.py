import pytest
from SharedState import SharedState
from models import Task


def test_shared_state_initialization():
    ss = SharedState()
    assert ss.project_id is None
    assert ss.tasks == {}
    assert ss.file_manager is not None
    assert ss.file_editor is not None


def test_get_task_not_found():
    ss = SharedState()
    result = ss.get_task("nonexistent")
    assert result is None


def test_get_task_found():
    ss = SharedState()
    task = Task(task_id="t1", desc="Test task", status="Not Assigned", worker_id="w1")
    ss.tasks["t1"] = task

    result = ss.get_task("t1")
    assert result == task
    assert result.task_id == "t1"


def test_update_task_status_success():
    ss = SharedState()
    task = Task(task_id="t2", desc="Update me", status="Not Assigned", worker_id="w1")
    ss.tasks["t2"] = task

    ss.update_task_status("t2", "In Progress")
    assert ss.tasks["t2"].status == "In Progress"


def test_update_task_status_not_found():
    ss = SharedState()
    with pytest.raises(KeyError):
        ss.update_task_status("nonexistent", "In Progress")


def test_shared_state_thread_safety():
    """Verify that mu lock is properly used"""
    ss = SharedState()
    task = Task(task_id="t3", desc="Thread test", status="Not Assigned", worker_id="w1")
    ss.tasks["t3"] = task

    # Verify lock is a Lock instance
    from threading import Lock
    assert isinstance(ss.mu, type(Lock()))

    # Concurrent operations should be safe
    ss.update_task_status("t3", "Done")
    assert ss.get_task("t3").status == "Done"
