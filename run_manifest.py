from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTROL_DIR = ".autoresearch"
FINAL_STATUSES = {"succeeded", "failed", "interrupted"}


@dataclass(frozen=True)
class RunPaths:
    workspace_root: Path
    run_id: str
    control_dir: Path
    run_dir: Path
    manifest_path: Path
    stdout_path: Path


def _validate_run_id(run_id: str) -> str:
    if not run_id:
        raise ValueError("run_id must not be empty")
    if run_id in {".", ".."}:
        raise ValueError("run_id must not be a relative path component")
    if Path(run_id).is_absolute():
        raise ValueError("run_id must be relative")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must not contain path separators")
    return run_id


def run_paths(workspace_root: str | Path, run_id: str) -> RunPaths:
    workspace_root = Path(workspace_root).resolve()
    run_id = _validate_run_id(run_id)
    control_dir = workspace_root / CONTROL_DIR / "runs"
    run_dir = control_dir / run_id
    return RunPaths(
        workspace_root=workspace_root,
        run_id=run_id,
        control_dir=control_dir,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        stdout_path=run_dir / "stdout.log",
    )


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
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def load_run_manifest(workspace_root: str | Path, run_id: str) -> dict[str, Any] | None:
    manifest_path = run_paths(workspace_root, run_id).manifest_path
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def ensure_run_manifest(
    workspace_root: str | Path,
    run_id: str,
    *,
    command: list[str] | None = None,
) -> dict[str, Any]:
    existing = load_run_manifest(workspace_root, run_id)
    if existing is not None:
        if command is not None and existing.get("command") != command:
            raise ValueError(f"run {run_id} already exists with a different command")
        return existing

    paths = run_paths(workspace_root, run_id)
    now = _utc_now()
    manifest = {
        "run_id": run_id,
        "status": "pending",
        "command": command or [],
        "pid": None,
        "exit_code": None,
        "result": None,
        "result_recorded_at": None,
        "created_at": now,
        "updated_at": now,
        "manifest_path": str(paths.manifest_path.relative_to(paths.workspace_root)),
        "stdout_path": str(paths.stdout_path.relative_to(paths.workspace_root)),
    }
    _write_json(paths.manifest_path, manifest)
    return manifest


def _save_manifest(workspace_root: str | Path, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    paths = run_paths(workspace_root, run_id)
    manifest["updated_at"] = _utc_now()
    _write_json(paths.manifest_path, manifest)
    return manifest


def mark_run_running(workspace_root: str | Path, run_id: str, *, pid: int | None = None) -> dict[str, Any]:
    manifest = ensure_run_manifest(workspace_root, run_id)
    if manifest["status"] in FINAL_STATUSES:
        raise ValueError(f"run {run_id} is already finalized")
    manifest["status"] = "running"
    manifest["pid"] = pid
    return _save_manifest(workspace_root, run_id, manifest)


def reconcile_run(workspace_root: str | Path, run_id: str) -> dict[str, Any]:
    manifest = ensure_run_manifest(workspace_root, run_id)
    if manifest["status"] != "running":
        return manifest

    manifest["status"] = "interrupted"
    manifest["pid"] = None
    return _save_manifest(workspace_root, run_id, manifest)


def finalize_run(workspace_root: str | Path, run_id: str, *, exit_code: int) -> dict[str, Any]:
    manifest = ensure_run_manifest(workspace_root, run_id)
    final_status = "succeeded" if exit_code == 0 else "failed"
    if manifest["status"] in FINAL_STATUSES:
        if manifest["status"] == final_status and manifest.get("exit_code") == exit_code and manifest.get("pid") is None:
            return manifest
        raise ValueError(f"run {run_id} is already finalized")

    manifest["status"] = final_status
    manifest["exit_code"] = exit_code
    manifest["pid"] = None
    return _save_manifest(workspace_root, run_id, manifest)


def record_run_result(
    workspace_root: str | Path,
    run_id: str,
    *,
    result: dict[str, Any],
) -> dict[str, Any]:
    manifest = ensure_run_manifest(workspace_root, run_id)
    existing_result = manifest.get("result")
    if existing_result is not None:
        if existing_result != result:
            raise ValueError(f"run {run_id} already has a different recorded result")
        return manifest

    manifest["result"] = result
    return _save_manifest(workspace_root, run_id, manifest)


def mark_run_result_recorded(workspace_root: str | Path, run_id: str) -> dict[str, Any]:
    manifest = ensure_run_manifest(workspace_root, run_id)
    if manifest.get("result") is None:
        raise ValueError(f"run {run_id} has no result to mark as recorded")
    if manifest.get("result_recorded_at"):
        return manifest
    manifest["result_recorded_at"] = _utc_now()
    return _save_manifest(workspace_root, run_id, manifest)
