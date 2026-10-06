"""
forge.py — single entry point: idea → Architect → TeamLead → Workers, with a Rich UI.

    python forge.py "build a todo CLI"
    python forge.py            # prompts for the idea
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.prompt import Prompt

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    def load_dotenv(*_a: Any, **_k: Any) -> bool:
        return False

from ArchitectAgent import run_architect
from TeamLeadAgent import run_teamlead
from forge_events import emit, get_bus, telemetry_subscriber
from forge_ui import ForgeUI

PROJECT_ROOT = Path(__file__).resolve().parent


def make_project_id(idea: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", idea[:20].lower()).strip("-") or "project"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def bootstrap_project(project_id: str) -> Path:
    project_dir = PROJECT_ROOT / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "tests").mkdir(exist_ok=True)
    for name in ("locks.json", "project_info.json"):
        path = project_dir / name
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
    return project_dir


def run_pipeline(idea: str, project_id: str, ui: ForgeUI, project_dir: Path) -> Optional[Dict[str, Any]]:
    ui.state.set_phase("architect")
    spec = run_architect(
        project_id,
        idea,
        ask_questions=lambda questions: ui.request_input("questions", {"questions": questions}),
        ask_decision=lambda values: ui.request_input(
            "decision",
            {"understanding": values.get("understanding", ""), "answers": values.get("answers", [])},
        ),
    )
    if spec is None:
        emit("run_end", agent="forge", success=False, metadata={"reason": "architect rejected"})
        return None

    (project_dir / "architecture_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (project_dir / "architecture_spec.md").write_text(spec["spec_text"], encoding="utf-8")

    ui.state.set_phase("execution")
    report = run_teamlead(project_id, spec["spec_text"])
    emit("run_end", agent="forge", success=report["final_status"] != "failed",
         metadata={"final_status": report["final_status"]})
    return report


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(prog="forge", description="Idea to code with a live agent dashboard.")
    parser.add_argument("idea", nargs="?", help="project idea (prompted if omitted)")
    parser.add_argument("--project-id", help="reuse or choose a project id")
    args = parser.parse_args(argv)

    console = Console()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY is not set.[/] Put it in .env or the environment.")
        return 2

    idea = (args.idea or Prompt.ask("Describe your project idea", console=console)).strip()
    if not idea:
        console.print("[red]No idea given.[/]")
        return 2

    project_id = args.project_id or make_project_id(idea)
    project_dir = bootstrap_project(project_id)

    # Agent logging would corrupt the live display; send it to a file instead.
    log_path = project_dir / "forge.log"
    logging.basicConfig(
        level=logging.INFO, filename=log_path, encoding="utf-8", force=True,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    bus = get_bus()
    bus.project_id = project_id
    bus.subscribe(telemetry_subscriber)
    ui = ForgeUI(bus, console)
    ui.state.project_id = project_id

    try:
        report = ui.run(lambda: run_pipeline(idea, project_id, ui, project_dir))
    except KeyboardInterrupt:
        return 130
    except Exception:
        console.print(f"Details are in {log_path}")
        return 1

    ui.print_summary(report)
    console.print(f"Log: {log_path}")
    return 0 if report and report["final_status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
