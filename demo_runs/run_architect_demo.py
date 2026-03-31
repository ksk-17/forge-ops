import sys
from pathlib import Path

from ArchitectAgent import run_architect_cli
from TeamLeadAgent import run_teamlead

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

import json
import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("architect_demo")

def bootstrap_project(project_id: str) -> Path:
    """Create project directory structure."""
    project_dir = _PROJECT_ROOT / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)

    locks_file = project_dir / "locks.json"
    if not locks_file.exists():
        locks_file.write_text("{}", encoding="utf-8")

    info_file = project_dir / "project_info.json"
    if not info_file.exists():
        info_file.write_text("{}", encoding="utf-8")

    logger.info("Project directory: %s", project_dir.resolve())
    return project_dir

def print_spec(spec: dict) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print("  ARCHITECTURE SPECIFICATION PRODUCED")
    print(sep)
    print(f"  Project : {spec['project_name']}")
    print(f"  ID      : {spec['project_id']}")
    print(f"  Created : {spec['created_at']}")
    print(f"\n{'-' * 64}")
    print(spec["spec_text"])
    print(f"{sep}\n")

def print_project_report(report: dict) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print("  PROJECT COMPLETION REPORT")
    print(sep)
    print(f"  Status     : {report['final_status'].upper()}")
    print(f"  Tasks      : {report['completed_tasks']}/{report['total_tasks']} completed")
    print(f"  Failed     : {report['failed_tasks']}")
    if report["batch_review_scores"]:
        avg = sum(report["batch_review_scores"]) / len(report["batch_review_scores"])
        print(f"  Avg review : {avg:.1f}/10")
    if report["all_touched_files"]:
        print(f"\n  Files produced:")
        for fp in sorted(report["all_touched_files"]):
            print(f"    - {fp}")
    print(f"\n  Summary:\n    {report['summary']}")
    print(f"{sep}\n")

DEFAULT_IDEA = """
I want to build a simple task management CLI tool in Python.
Users should be able to add tasks, list them, mark them as done,
and delete them. Tasks should persist between sessions.
"""

if __name__ == "__main__":
    logger.info("=== ArchitectAgent Demo ===")
    logger.info("Project root: %s", _PROJECT_ROOT)

    print("\n" + "=" * 64)
    print("  FORGE-OPS — Architect Agent")
    print("=" * 64)
    print("\n  Describe your project idea.")
    print("  (Press Enter twice when done, or just press Enter to use the demo idea)\n")

    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines:
                break
            if line == "" and not lines:
                # Use default
                print("  [Using default demo idea: Task Manager CLI]")
                lines = [DEFAULT_IDEA.strip()]
                break
            lines.append(line)
    except EOFError:
        lines = [DEFAULT_IDEA.strip()]

    raw_input = "\n".join(lines).strip() or DEFAULT_IDEA.strip()

    # ── Generate project ID ──────────────────────────────────────────────
    slug = raw_input[:20].lower().replace(" ", "-").replace("/", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    project_id = f"{slug}-{uuid.uuid4().hex[:6]}"
    logger.info("Project ID: %s", project_id)

    project_dir = bootstrap_project(project_id)

    # ── Run ArchitectAgent ───────────────────────────────────────────────
    print("\n  Starting requirements elicitation...\n")
    spec = run_architect_cli(project_id, raw_input)

    if spec is None:
        print("\n  [Aborted] No architecture spec produced.")
        sys.exit(1)

    print_spec(spec)

    # Save spec to disk
    spec_path = project_dir / "architecture_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    spec_txt_path = project_dir / "architecture_spec.md"
    spec_txt_path.write_text(spec["spec_text"], encoding="utf-8")
    logger.info("Spec saved to: %s", spec_path.resolve())

    # ── Ask whether to proceed to TeamLeadAgent ──────────────────────────
    print("=" * 64)
    try:
        proceed = input("  Proceed to build with TeamLeadAgent? [y/n]: ").strip().lower()
    except EOFError:
        proceed = "n"

    if proceed not in ("y", "yes"):
        print("  [Stopped after architecture spec. Run TeamLeadAgent separately.]")
        sys.exit(0)

    logger.info("Handing spec to TeamLeadAgent...")
    report = run_teamlead(
        project_id=project_id,
        architecture_spec=spec["spec_text"],
    )

    print_project_report(report)

    # Save project report
    report_path = project_dir / "project_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Project report saved to: %s", report_path.resolve())

    if report["final_status"] == "failed":
        sys.exit(1)
    elif report["final_status"] == "partial":
        sys.exit(2)
    else:
        sys.exit(0)
