from __future__ import annotations

import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"^step\s+"
    r"(?P<step>\d+)\s+"
    r"\((?P<progress_percent>[\d.]+)%\)\s+\|\s+"
    r"loss:\s+(?P<loss>[\d.]+)\s+\|\s+"
    r"lrm:\s+(?P<lrm>[\d.]+)\s+\|\s+"
    r"dt:\s+(?P<dt_ms>[\d.]+)ms\s+\|\s+"
    r"tok/sec:\s+(?P<tok_per_sec>[\d,]+)\s+\|\s+"
    r"mfu:\s+(?P<mfu_percent>[\d.]+)%\s+\|\s+"
    r"epoch:\s+(?P<epoch>\d+)\s+\|\s+"
    r"remaining:\s+(?P<remaining_seconds>\d+)s\s*$"
)

SUMMARY_RE = re.compile(r"^(?P<key>[a-z_]+):\s+(?P<value>.+?)\s*$")


def _parse_number(value: str) -> float:
    return float(value.replace(",", "").strip())


def parse_run_log_text(text: str) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    summary: dict[str, float] = {}

    for line in text.splitlines():
        step_match = STEP_RE.match(line)
        if step_match:
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
                    "remaining_seconds": int(groups["remaining_seconds"]),
                }
            )
            continue

        summary_match = SUMMARY_RE.match(line)
        if summary_match:
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
