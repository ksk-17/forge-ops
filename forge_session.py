"""
forge_session.py — the interactive `forge>` prompt.

Typed text is a project idea and runs the full pipeline; slash commands are
handled locally. The session is decoupled from the pipeline: it only needs a
`run_idea(idea) -> result` callable whose result has `.input_tokens`,
`.output_tokens` and `.cost_usd`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

HELP = [
    ("/help", "show this list"),
    ("/projects", "list projects built so far"),
    ("/usage", "token and cost totals for this session"),
    ("/exit", "leave forge (also /quit, or Ctrl+D)"),
]


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class Session:
    def __init__(self, console: Console, run_idea: Callable[[str], Any], projects_dir: Path) -> None:
        self.console = console
        self._run_idea = run_idea
        self._projects_dir = projects_dir
        self.runs = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    # -- loop --------------------------------------------------------------
    def loop(self, read: Optional[Callable[[], str]] = None) -> int:
        read = read or (lambda: self.console.input("[bold cyan]forge>[/] "))
        self._banner()
        while True:
            try:
                line = read()
            except EOFError:
                self.console.print()
                return 0
            except KeyboardInterrupt:
                self.console.print(Text("\n(type /exit to quit)", style="dim"))
                continue
            if not self.handle(line):
                return 0

    def handle(self, line: str) -> bool:
        """Process one line. Returns False when the session should end."""
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            command = line.split()[0].lower()
            if command in ("/exit", "/quit"):
                return False
            if command == "/help":
                self._help()
            elif command == "/projects":
                self._projects()
            elif command == "/usage":
                self._usage()
            else:
                self.console.print(Text(f"Unknown command {command}. Type /help for the list."))
            return True
        self._run(line)
        return True

    # -- commands ----------------------------------------------------------
    def _banner(self) -> None:
        self.console.print(Text("forge — describe a project and the agents will build it.", style="bold cyan"))
        self._help()

    def _help(self) -> None:
        self.console.print(Text("Type an idea to start a project, or a command:", style="dim"))
        for cmd, desc in HELP:
            self.console.print(Text(f"  {cmd:<10} {desc}"))

    def _usage(self) -> None:
        self.console.print(Text(
            f"Session: {self.runs} run(s) · {_fmt_tokens(self.input_tokens)} in / "
            f"{_fmt_tokens(self.output_tokens)} out · est. ${self.cost_usd:.4f}"))

    def _projects(self) -> None:
        dirs = []
        if self._projects_dir.is_dir():
            dirs = [d for d in self._projects_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        if not dirs:
            self.console.print(Text("No projects yet."))
            return
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        table = Table(box=None, pad_edge=False)
        table.add_column("Project")
        table.add_column("Status")
        for d in dirs[:20]:
            table.add_row(Text(d.name), Text(self._status_of(d)))
        self.console.print(table)

    @staticmethod
    def _status_of(project_dir: Path) -> str:
        report = project_dir / "project_report.json"
        try:
            return str(json.loads(report.read_text(encoding="utf-8")).get("final_status", "-"))
        except (OSError, ValueError, AttributeError):
            return "-"

    # -- running a project -------------------------------------------------
    def _run(self, idea: str) -> None:
        try:
            result = self._run_idea(idea)
        except KeyboardInterrupt:
            self.console.print(Text("Cancelled.", style="yellow"))
            return
        except Exception as exc:
            self.console.print(Text(f"Run failed: {type(exc).__name__}: {exc}", style="red"))
            return
        self.runs += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.cost_usd += result.cost_usd
