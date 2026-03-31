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
logger = logging.getLogger("teamlead_demo")

PROJECT_ID = "demo-teamlead"
PROJECT_DIR = _PROJECT_ROOT / "projects" / PROJECT_ID


def bootstrap_project() -> None:
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

ARCHITECTURE_SPEC = """
# URL Shortener Service - Architecture Specification

## Overview
A minimal in-memory URL shortener implemented as three Python modules.
No web framework, no database - pure Python, importable and testable.

## Modules

### 1. src/models.py
Defines the data model for a shortened URL entry.

  Class: UrlRecord (dataclass)
  Fields:
    - short_code: str   - the 6-character alphanumeric code (e.g. "aB3xYz")
    - original_url: str - the full URL being shortened
    - created_at: str   - ISO-8601 timestamp string (set at creation time)
    - hit_count: int    - number of times this short code has been resolved (default 0)

  No methods required - pure data container.
  Include a module-level docstring.

Dependencies: none

---

### 2. src/store.py
In-memory storage and lookup for UrlRecord objects.

  Class: InMemoryStore
  Methods:
    - save(record: UrlRecord) -> None
        Store a UrlRecord keyed by record.short_code.
        Raise ValueError("Short code already exists") if the code is taken.

    - get(short_code: str) -> UrlRecord
        Return the UrlRecord for short_code.
        Raise KeyError(short_code) if not found.

    - increment_hits(short_code: str) -> None
        Increment hit_count on the matching record by 1.
        Raise KeyError(short_code) if not found.

    - all_records() -> list[UrlRecord]
        Return a list of all stored UrlRecord objects (any order).

  Import UrlRecord from src.models.
  Include a module-level docstring and docstring on every method.

Dependencies: src/models.py must exist first

---

### 3. src/shortener.py
High-level service that ties models and store together.

  Class: Shortener
  Constructor: __init__(self, store: InMemoryStore)
    Accept an InMemoryStore instance (dependency injection).

  Methods:
    - shorten(original_url: str) -> UrlRecord
        Generate a unique 6-character alphanumeric short_code using
        random.choices from string.ascii_letters + string.digits.
        Retry up to 5 times if the code already exists in the store.
        Raise RuntimeError("Could not generate unique short code") after 5 failures.
        Create a UrlRecord with the code, url, and current UTC timestamp.
        Save it to the store and return the record.

    - resolve(short_code: str) -> str
        Call store.increment_hits(short_code), then return record.original_url.
        Let KeyError propagate from store.get if code not found.

    - stats() -> dict
        Return {"total_urls": int, "total_hits": int} aggregated across all records.

  Import InMemoryStore from src.store and UrlRecord from src.models.
  Include a module-level docstring and docstring on every method.

Dependencies: src/models.py and src/store.py must both exist first

---

## Constraints
- Pure Python standard library only (no third-party packages).
- Every class and method must have a docstring.
- Every module must have a module-level docstring.
- Files live under src/ (src/models.py, src/store.py, src/shortener.py).
- Tests live under tests/ - one test file per source file.
"""

def print_project_report(report: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TEAM LEAD AGENT - PROJECT REPORT")
    print(sep)
    print(f"  Project ID      : {report['project_id']}")
    print(f"  Final status    : {report['final_status'].upper()}")
    print(f"  Total tasks     : {report['total_tasks']}")
    print(f"  Completed       : {report['completed_tasks']}")
    print(f"  Failed          : {report['failed_tasks']}")
    print(f"  Blocked         : {report['blocked_tasks']}")
    print(f"  Completed at    : {report['completed_at']}")

    if report["batch_review_scores"]:
        scores = report["batch_review_scores"]
        avg = sum(scores) / len(scores)
        print(f"\n  Batch review scores : {scores}  (avg {avg:.1f}/10)")

    if report["all_touched_files"]:
        print(f"\n  Files produced:")
        for fp in sorted(report["all_touched_files"]):
            print(f"    - {fp}")

    print(f"\n  Executive summary:\n    {report['summary']}")
    print(f"{sep}\n")


def print_tasks_summary(project_dir: Path) -> None:
    tasks_file = project_dir / "tasks.json"
    if not tasks_file.exists():
        return

    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    sep = "-" * 60
    print(f"\n{sep}")
    print("  TASK REGISTRY SUMMARY")
    print(sep)
    icons = {
        "Closed": "OK", "ReworkRequired": "FAIL",
        "Open": "OPEN", "InProgress": "...", "PendingReview": "REVIEW",
    }
    for rec in tasks:
        status = rec["status"]
        icon = icons.get(status, "?")
        score = ""
        if rec.get("report") and rec["report"].get("review_score") is not None:
            score = f"  score={rec['report']['review_score']}/10"
        print(f"  [{icon}] {rec['task_id']}  ({status}){score}")
        if rec.get("report") and rec["report"].get("summary"):
            print(f"        {rec['report']['summary'][:80]}")
        if rec.get("retry_count", 0) > 0:
            print(f"        retried {rec['retry_count']} time(s)")
    print(f"{sep}\n")

if __name__ == "__main__":
    logger.info("=== TeamLeadAgent Demo ===")
    logger.info("Project root : %s", _PROJECT_ROOT)
    logger.info("Project      : %s", PROJECT_ID)
    logger.info("Architecture : URL Shortener (3 modules, dependency chain)")

    bootstrap_project()

    from TeamLeadAgent import run_teamlead

    logger.info("Handing architecture spec to TeamLeadAgent...")
    report = run_teamlead(
        project_id=PROJECT_ID,
        architecture_spec=ARCHITECTURE_SPEC,
    )

    print_project_report(report)
    print_tasks_summary(PROJECT_DIR)

    report_path = PROJECT_DIR / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Full project report saved to: %s", report_path.resolve())

    if report["final_status"] == "failed":
        sys.exit(1)
    elif report["final_status"] == "partial":
        sys.exit(2)
    else:
        sys.exit(0)