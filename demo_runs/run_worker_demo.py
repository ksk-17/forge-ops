import sys
from pathlib import Path
import json
import logging
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("worker_demo")

PROJECT_ID = "demo-worker"
# Projects directory lives at the project root, not relative to this file
PROJECT_DIR = _PROJECT_ROOT / "projects" / PROJECT_ID


def bootstrap_project() -> None:
    """Create the minimal directory structure WorkerTools expects."""
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "src").mkdir(exist_ok=True)
    (PROJECT_DIR / "tests").mkdir(exist_ok=True)

    locks_file = PROJECT_DIR / "locks.json"
    if not locks_file.exists():
        locks_file.write_text("{}", encoding="utf-8")

    info_file = PROJECT_DIR / "project_info.json"
    if not info_file.exists():
        info_file.write_text("{}", encoding="utf-8")

    logger.info("Project directory ready at: %s", PROJECT_DIR.resolve())

def make_demo_task():
    from models import Task
    return Task(
        task_id="calc-impl",
        project_id=PROJECT_ID,
        desc=(
            "Implement a Calculator class in src/calculator.py.\n\n"
            "Requirements:\n"
            "  - Class name: Calculator\n"
            "  - Methods: add(a, b), subtract(a, b), multiply(a, b), divide(a, b)\n"
            "  - All methods accept int or float arguments and return a float.\n"
            "  - divide() must raise ValueError with message "
            "'Cannot divide by zero' when b == 0.\n"
            "  - Include a module-level docstring describing the module.\n"
            "  - Include a docstring on every method.\n\n"
            "Do NOT create any other files. The file must be importable "
            "as 'from src.calculator import Calculator'."
        ),
        status="InProgress",
        worker_id="worker-calc-impl",
        dependency_list=[],
    )

def print_report(report: dict) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print("  WORKER AGENT REPORT")
    print(sep)
    print(f"  Task ID      : {report['task_id']}")
    print(f"  Worker ID    : {report['worker_id']}")
    print(f"  Status       : {report['status'].upper()}")
    print(f"  Review score : {report['review_score']}/10")
    print(f"  Completed at : {report['completed_at']}")
    print(f"\n  Summary:\n    {report['summary']}")

    if report["touched_files"]:
        print(f"\n  Files written:")
        for fp in report["touched_files"]:
            print(f"    - {fp}")

    if report["test_file_path"]:
        print(f"\n  Test file    : {report['test_file_path']}")

    if report["blockers"]:
        print(f"\n  Blockers:")
        for b in report["blockers"]:
            print(f"    - {b}")

    if report["suggestions"]:
        print(f"\n  Suggestions:")
        for s in report["suggestions"]:
            print(f"    - {s}")

    print(f"\n  Review feedback (first 8 lines):")
    for line in report["review_feedback"].splitlines()[:8]:
        print(f"    {line}")
    if len(report["review_feedback"].splitlines()) > 8:
        print("    ...")

    print(f"{sep}\n")

if __name__ == "__main__":
    logger.info("=== WorkerAgent Demo ===")
    logger.info("Project root : %s", _PROJECT_ROOT)

    bootstrap_project()
    task = make_demo_task()
    logger.info("Running worker for task '%s'", task.task_id)

    from WorkerAgent import run_worker
    report = run_worker(task, worker_id="worker-calc-impl")

    print_report(report)

    report_path = PROJECT_DIR / "worker_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Full report saved to: %s", report_path.resolve())

    sys.exit(1 if report["status"] == "failed" else 0)