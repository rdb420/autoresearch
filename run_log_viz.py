from __future__ import annotations

import re
from itertools import chain
from pathlib import Path
from typing import Any

from autoresearch_runtime import preferred_log_candidates

STEP_RE = re.compile(
    r"step\s+"
    r"(?P<step>\d+)\s+"
    r"\((?P<progress_percent>[\d.]+)%\)\s+\|\s+"
    r"loss:\s+(?P<loss>[\d.]+)\s+\|\s+"
    r"lrm:\s+(?P<lrm>[\d.]+)\s+\|\s+"
    r"dt:\s+(?P<dt_ms>[\d.]+)ms\s+\|\s+"
    r"tok/sec:\s+(?P<tok_per_sec>[\d,]+)\s+\|\s+"
    r"mfu:\s+(?P<mfu_percent>[\d.]+)%\s+\|\s+"
    r"epoch:\s+(?P<epoch>\d+)"
    r"(?:\s+\|\s+remaining:\s+(?P<remaining_seconds>\d+)s)?"
)

SUMMARY_RE = re.compile(r"^(?P<key>[a-z_]+):\s+(?P<value>.+?)\s*$")


def _parse_number(value: str) -> float:
    return float(value.replace(",", "").strip())


def parse_run_log_text(text: str) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    summary: dict[str, float] = {}

    for step_match in STEP_RE.finditer(text):
        groups = step_match.groupdict()
        steps.append(
            {
                "step": int(groups["step"]),
                "progress_percent": _parse_number(groups["progress_percent"]),
                "loss": _parse_number(groups["loss"]),
                "lrm": _parse_number(groups["lrm"]),
                "dt_ms": _parse_number(groups["dt_ms"]),
                "tok_per_sec": int(groups["tok_per_sec"].replace(",", "")),
                "mfu_percent": _parse_number(groups["mfu_percent"]),
                "epoch": int(groups["epoch"]),
                "remaining_seconds": int(groups["remaining_seconds"] or 0),
            }
        )

    for line in text.splitlines():
        if summary_match := SUMMARY_RE.match(line):
            key = summary_match.group("key")
            try:
                summary[key] = _parse_number(summary_match.group("value"))
            except ValueError:
                continue

    return {"steps": steps, "summary": summary}


def load_run_log(path: str | Path = "run.log") -> dict[str, Any]:
    run_log_path = Path(path)
    if not run_log_path.exists():
        return {"steps": [], "summary": {}, "path": run_log_path, "exists": False}

    parsed = parse_run_log_text(run_log_path.read_text(errors="replace"))
    parsed["path"] = run_log_path
    parsed["exists"] = True
    return parsed


def load_latest_log(paths: list[str | Path]) -> dict[str, Any]:
    existing: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for candidate in paths:
        parsed = load_run_log(candidate)
        if parsed["exists"] and (parsed["steps"] or parsed["summary"]):
            parsed["mtime"] = parsed["path"].stat().st_mtime
            existing.append(parsed)
        else:
            missing.append(parsed)

    if existing:
        return max(existing, key=lambda parsed: parsed["mtime"])
    if missing:
        return missing[0]
    return {"steps": [], "summary": {}, "path": Path("run.log"), "exists": False}


def discover_cursor_terminal_logs(workspace_root: str | Path) -> list[Path]:
    workspace_root = Path(workspace_root).resolve()
    terminals_root = Path.home() / ".cursor" / "projects"
    if not terminals_root.exists():
        return []

    matches: list[Path] = []
    for path in chain(terminals_root.glob("*/terminals/*.txt"), terminals_root.glob("*/*/terminals/*.txt")):
        try:
            header = path.read_text(errors="replace")[:1000]
        except OSError:
            continue

        if f"cwd: {workspace_root}" not in header:
            continue
        if "last_command: uv run train.py" not in header:
            continue
        matches.append(path)

    return matches


def load_workspace_log(
    workspace_root: str | Path,
    run_log_path: str | Path = "run.log",
) -> dict[str, Any]:
    workspace_root = Path(workspace_root).resolve()
    for candidate in preferred_log_candidates(workspace_root, run_log_path=run_log_path):
        parsed = load_run_log(candidate)
        if parsed["exists"] and (parsed["steps"] or parsed["summary"]):
            return parsed

    terminal_candidates = discover_cursor_terminal_logs(workspace_root)
    if terminal_candidates:
        return load_latest_log(terminal_candidates)
    return load_run_log(Path(workspace_root) / run_log_path)


def steps_to_frame(steps: list[dict[str, Any]]) -> pd.DataFrame:
    import pandas as pd

    if not steps:
        return pd.DataFrame(
            columns=[
                "step",
                "progress_percent",
                "loss",
                "lrm",
                "dt_ms",
                "tok_per_sec",
                "mfu_percent",
                "epoch",
                "remaining_seconds",
            ]
        )
    return pd.DataFrame(steps)
