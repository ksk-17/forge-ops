import os
import time
import pytest
from pathlib import Path

from FileManager import FileManager

def test_create_project_and_file(tmp_path):
    fm = FileManager(base_path=tmp_path)
    project_id = fm.create_project()
    assert isinstance(project_id, str) and project_id
    project_path = tmp_path / project_id
    assert project_path.exists() and project_path.is_dir()

    p = fm.create_file(project_id, "folder/sub.txt")
    assert p.exists()
    assert (project_path / "folder" / "sub.txt").exists()


def test_abs_path_prevents_escape(tmp_path):
    fm = FileManager(base_path=tmp_path)
    project_id = fm.create_project()
    # try to escape using ../
    with pytest.raises(ValueError):
        fm.abs_path(project_id, "../outside.txt")


def test_lock_lifecycle(tmp_path):
    fm = FileManager(base_path=tmp_path)
    project_id = fm.create_project()
    file_path = "a.txt"
    worker1 = "w1"
    worker2 = "w2"

    # acquire lock by worker1
    assert fm.acquire_lock(project_id, file_path, worker1, ttl_seconds=2) is True
    # second acquire should fail
    assert fm.acquire_lock(project_id, file_path, worker2) is False
    # check_lock should show worker1
    assert fm.check_lock(project_id, file_path) == worker1

    # renew by wrong worker should fail
    assert fm.renew_lock(project_id, file_path, worker2) is False
    # renew by correct worker should succeed
    assert fm.renew_lock(project_id, file_path, worker1, ttl_seconds=2) is True

    # release by wrong worker should fail
    assert fm.release_lock(project_id, file_path, worker2) is False
    # release by correct worker should succeed
    assert fm.release_lock(project_id, file_path, worker1) is True
    # now no lock
    assert fm.check_lock(project_id, file_path) is None


def test_lock_expiration(tmp_path):
    fm = FileManager(base_path=tmp_path)
    project_id = fm.create_project()
    fp = "exp.txt"
    worker = "w"

    # acquire with short ttl
    assert fm.acquire_lock(project_id, fp, worker, ttl_seconds=1) is True
    assert fm.check_lock(project_id, fp) == worker
    # wait for expiration
    time.sleep(1.2)
    # after ttl, lock should be gone and acquire should succeed again
    assert fm.acquire_lock(project_id, fp, "other", ttl_seconds=1) is True