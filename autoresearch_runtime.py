from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_manifest import (
    ensure_run_manifest,
    finalize_run,
    load_run_manifest,
    mark_run_result_recorded,
    reconcile_run,
    record_run_result,
    run_paths,
    mark_run_running,
)


CONTROL_DIR = ".autoresearch"
STATE_DIR = "state"
RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"


class TeeTextIO:
    def __init__(self, *targets: Any) -> None:
        self.targets = targets

    def write(self, text: str) -> int:
        for target in self.targets:
            target.write(text)
        return len(text)

    def flush(self) -> None:
        for target in self.targets:
            target.flush()

    def isatty(self) -> bool:
        return any(getattr(target, "isatty", lambda: False)() for target in self.targets)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload_text)
        handle.flush()
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _runtime_paths(workspace_root: str | Path) -> dict[str, Path]:
    workspace_root = Path(workspace_root).resolve()
    control_dir = workspace_root / CONTROL_DIR
    state_dir = control_dir / STATE_DIR
    return {
        "workspace_root": workspace_root,
        "control_dir": control_dir,
        "state_dir": state_dir,
        "current_run_path": state_dir / "current_run.json",
        "last_run_path": state_dir / "last_run.json",
        "results_path": workspace_root / "results.tsv",
        "canonical_run_log_path": workspace_root / "run.log",
    }


def ensure_runtime_state(workspace_root: str | Path) -> dict[str, Path]:
    paths = _runtime_paths(workspace_root)
    paths["state_dir"].mkdir(parents=True, exist_ok=True)
    if not paths["results_path"].exists() or not paths["results_path"].read_text().strip():
        paths["results_path"].write_text(RESULTS_HEADER)
    return paths


def _pointer_payload(
    workspace_root: str | Path,
    manifest: dict[str, Any],
    *,
    next_action: str,
) -> dict[str, Any]:
    paths = _runtime_paths(workspace_root)
    run_id = manifest["run_id"]
    run_layout = run_paths(paths["workspace_root"], run_id)
    return {
        "run_id": run_id,
        "status": manifest["status"],
        "pid": manifest.get("pid"),
        "manifest_path": str(run_layout.manifest_path),
        "stdout_path": str(run_layout.stdout_path),
        "canonical_log_path": str(paths["canonical_run_log_path"]),
        "updated_at": _utc_now(),
        "next_action": next_action,
    }


def load_current_run(workspace_root: str | Path) -> dict[str, Any] | None:
    return _read_json(_runtime_paths(workspace_root)["current_run_path"])


def load_last_run(workspace_root: str | Path) -> dict[str, Any] | None:
    return _read_json(_runtime_paths(workspace_root)["last_run_path"])


def _set_current_run(workspace_root: str | Path, payload: dict[str, Any]) -> None:
    _write_json(_runtime_paths(workspace_root)["current_run_path"], payload)


def _set_last_run(workspace_root: str | Path, payload: dict[str, Any]) -> None:
    _write_json(_runtime_paths(workspace_root)["last_run_path"], payload)


def _clear_current_run_if_matches(workspace_root: str | Path, run_id: str) -> None:
    current_path = _runtime_paths(workspace_root)["current_run_path"]
    current = _read_json(current_path)
    if current is not None and current.get("run_id") == run_id:
        current_path.unlink(missing_ok=True)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


def _pid_is_running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_run(
    workspace_root: str | Path,
    *,
    command: list[str],
    run_id: str | None = None,
    pid: int | None = None,
) -> dict[str, Any]:
    ensure_runtime_state(workspace_root)
    existing_current = load_current_run(workspace_root)
    if existing_current is not None:
        previous_run_id = existing_current["run_id"]
        previous_manifest = load_run_manifest(workspace_root, previous_run_id)
        if previous_manifest is not None and previous_manifest.get("status") == "running":
            if _pid_is_running(previous_manifest.get("pid")):
                raise ValueError(f"run {previous_run_id} is still active")
            reconciled = reconcile_run(workspace_root, previous_run_id)
            _set_last_run(
                workspace_root,
                _pointer_payload(
                    workspace_root,
                    reconciled,
                    next_action="review_interrupted_run_before_retry",
                ),
            )

    run_id = run_id or _default_run_id()
    ensure_run_manifest(workspace_root, run_id, command=command)
    manifest = mark_run_running(workspace_root, run_id, pid=pid)
    payload = _pointer_payload(workspace_root, manifest, next_action="monitor_current_run")
    _set_current_run(workspace_root, payload)
    return payload


def preferred_log_candidates(
    workspace_root: str | Path,
    *,
    run_log_path: str | Path = "run.log",
) -> list[Path]:
    workspace_root = Path(workspace_root).resolve()
    candidates: list[Path] = []
    for pointer in (load_current_run(workspace_root), load_last_run(workspace_root)):
        if pointer is not None:
            candidates.append(Path(pointer["stdout_path"]))
            break
    candidates.append((workspace_root / run_log_path).resolve())

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _sanitize_description(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())[:160] or "no description"


def _git_output(workspace_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return None if result.returncode != 0 else result.stdout.strip()


def _read_result_rows(results_path: Path) -> list[dict[str, str]]:
    lines = results_path.read_text().splitlines()
    if len(lines) <= 1:
        return []
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        commit, val_bpb, memory_gb, status, description = line.split("\t", 4)
        rows.append(
            {
                "commit": commit,
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "status": status,
                "description": description,
            }
        )
    return rows


def _best_non_crash_bpb(results_path: Path) -> float | None:
    best: float | None = None
    for row in _read_result_rows(results_path):
        if row["status"].lower() == "crash":
            continue
        value = float(row["val_bpb"])
        if best is None or value < best:
            best = value
    return best


def _failure_description(log_path: Path) -> str:
    from run_log_viz import load_run_log

    parsed = load_run_log(log_path)
    if parsed["summary"] and "val_bpb" not in parsed["summary"]:
        return "missing val_bpb summary"
    lines = [line.strip() for line in log_path.read_text(errors="replace").splitlines() if line.strip()]
    return _sanitize_description(lines[-1]) if lines else "training run failed"


def _result_row(
    workspace_root: Path,
    *,
    log_path: Path,
    exit_code: int,
) -> dict[str, str]:
    from run_log_viz import load_run_log

    parsed = load_run_log(log_path)
    summary = parsed["summary"]
    commit = _git_output(workspace_root, "rev-parse", "--short", "HEAD") or "unknown"
    description = _git_output(workspace_root, "log", "-1", "--format=%s") or "autonomous run"

    if exit_code != 0 or "val_bpb" not in summary:
        return {
            "commit": commit,
            "val_bpb": "0.000000",
            "memory_gb": "0.0",
            "status": "crash",
            "description": _failure_description(log_path),
        }

    val_bpb = float(summary["val_bpb"])
    peak_vram_mb = float(summary.get("peak_vram_mb", 0.0))
    previous_best = _best_non_crash_bpb(workspace_root / "results.tsv")
    status = "keep" if previous_best is None or val_bpb < previous_best else "discard"
    return {
        "commit": commit,
        "val_bpb": f"{val_bpb:.6f}",
        "memory_gb": f"{peak_vram_mb / 1024:.1f}",
        "status": status,
        "description": _sanitize_description(description),
    }


def _append_row(results_path: Path, row: dict[str, str]) -> None:
    line = "\t".join(
        [row["commit"], row["val_bpb"], row["memory_gb"], row["status"], row["description"]]
    )
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def complete_run(
    workspace_root: str | Path,
    run_id: str,
    *,
    exit_code: int,
) -> dict[str, str]:
    paths = ensure_runtime_state(workspace_root)
    finalize_run(workspace_root, run_id, exit_code=exit_code)
    log_path = run_paths(paths["workspace_root"], run_id).stdout_path
    row = _result_row(paths["workspace_root"], log_path=log_path, exit_code=exit_code)
    manifest = record_run_result(workspace_root, run_id, result=row)
    if not manifest.get("result_recorded_at"):
        _append_row(paths["results_path"], row)
        mark_run_result_recorded(workspace_root, run_id)

    manifest = load_run_manifest(workspace_root, run_id)
    next_action = "inspect_crash_and_retry" if row["status"] == "crash" else "analyze_result_and_plan_next_run"
    if manifest is not None:
        _set_last_run(workspace_root, _pointer_payload(workspace_root, manifest, next_action=next_action))
    _clear_current_run_if_matches(workspace_root, run_id)
    return row
