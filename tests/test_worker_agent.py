from dotenv import load_dotenv

loaded_env = load_dotenv()
print("Loaded API keys succesfully: ", loaded_env)

from WorkerAgent import agent as worker_agent
from SharedState import SharedState
from models import Task

def test_worker_agent():
    task = Task(task_id="t1", desc="Write a simple crud operations for a note taking app. Initialize a class for Note, ensure to have a field user_id in it. Create create, update, get_tasks, get_task_by_id, delete.", status="Assigned", worker_id="w1")
    shared_state = SharedState()
    project_id = shared_state.file_manager.create_project()
    shared_state.project_id = project_id
    shared_state.tasks[task.task_id] = task
    
    initial_state = {
        "task_id": "t1",
        "worker_id": "w1",
        "messages": [],
        "num_calls": 0,
        "last_review": None,
        "last_answer": None,
        "shared_state": shared_state
    }

    final_state = worker_agent.invoke(initial_state)

    print("\n\nnum_calls:", final_state["num_calls"])
    print("\n\nlast_review:", final_state["last_review"])
    print("\n\nlast_answer:", final_state["last_answer"])