import tempfile
import unittest
from pathlib import Path

from run_manifest import (
    ensure_run_manifest,
    finalize_run,
    load_run_manifest,
    mark_run_result_recorded,
    mark_run_running,
    reconcile_run,
    record_run_result,
    run_paths,
)


class RunManifestTests(unittest.TestCase):
    def test_rejects_run_ids_that_escape_control_plane_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            for invalid_run_id in ("../escape", "nested/run", "/abs/path", ""):
                with self.assertRaises(ValueError):
                    run_paths(workspace_root, invalid_run_id)

    def test_run_paths_use_repo_local_control_plane_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            paths = run_paths(workspace_root, "run-123")

            self.assertEqual(paths.control_dir, workspace_root / ".autoresearch" / "runs")
            self.assertEqual(paths.run_dir, workspace_root / ".autoresearch" / "runs" / "run-123")
            self.assertEqual(paths.manifest_path, paths.run_dir / "manifest.json")
            self.assertEqual(paths.stdout_path, paths.run_dir / "stdout.log")

    def test_ensure_run_manifest_creates_and_recovers_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            created = ensure_run_manifest(workspace_root, "run-123", command=["uv", "run", "train.py"])
            recovered = ensure_run_manifest(workspace_root, "run-123", command=["uv", "run", "train.py"])

            self.assertEqual(created["run_id"], "run-123")
            self.assertEqual(created["status"], "pending")
            self.assertEqual(created["command"], ["uv", "run", "train.py"])
            self.assertEqual(recovered, created)

    def test_ensure_run_manifest_rejects_conflicting_command_for_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            ensure_run_manifest(workspace_root, "run-123", command=["uv", "run", "train.py"])

            with self.assertRaises(ValueError):
                ensure_run_manifest(workspace_root, "run-123", command=["uv", "run", "other.py"])

    def test_manifest_tracks_running_and_finished_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            ensure_run_manifest(workspace_root, "run-123")
            running = mark_run_running(workspace_root, "run-123", pid=4242)
            finished = finalize_run(workspace_root, "run-123", exit_code=0)

            self.assertEqual(running["status"], "running")
            self.assertEqual(running["pid"], 4242)
            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(finished["exit_code"], 0)
            self.assertIsNone(finished["pid"])
            self.assertEqual(load_run_manifest(workspace_root, "run-123"), finished)

    def test_reconcile_run_marks_stale_running_manifest_interrupted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            ensure_run_manifest(workspace_root, "run-123")
            mark_run_running(workspace_root, "run-123", pid=4242)

            reconciled = reconcile_run(workspace_root, "run-123")

            self.assertEqual(reconciled["status"], "interrupted")
            self.assertIsNone(reconciled["pid"])
            self.assertEqual(reconciled["exit_code"], None)

    def test_finalize_run_is_idempotent_for_same_exit_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            ensure_run_manifest(workspace_root, "run-123")
            first = finalize_run(workspace_root, "run-123", exit_code=0)
            second = finalize_run(workspace_root, "run-123", exit_code=0)

            self.assertEqual(first, second)

    def test_finalize_run_rejects_conflicting_exit_code_after_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            ensure_run_manifest(workspace_root, "run-123")
            finalize_run(workspace_root, "run-123", exit_code=0)

            with self.assertRaises(ValueError):
                finalize_run(workspace_root, "run-123", exit_code=1)

    def test_record_run_result_is_idempotent_for_same_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            ensure_run_manifest(workspace_root, "run-123")
            result = {"val_bpb": 1.142676, "peak_vram_mb": 4096.0}

            first = record_run_result(workspace_root, "run-123", result=result)
            second = record_run_result(workspace_root, "run-123", result=result)

            self.assertEqual(first["result"], result)
            self.assertEqual(second, first)

    def test_record_run_result_rejects_conflicting_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            ensure_run_manifest(workspace_root, "run-123")
            record_run_result(workspace_root, "run-123", result={"val_bpb": 1.1})

            with self.assertRaises(ValueError):
                record_run_result(workspace_root, "run-123", result={"val_bpb": 1.2})

    def test_mark_run_result_recorded_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            ensure_run_manifest(workspace_root, "run-123")
            record_run_result(workspace_root, "run-123", result={"val_bpb": 1.1})

            first = mark_run_result_recorded(workspace_root, "run-123")
            second = mark_run_result_recorded(workspace_root, "run-123")

            self.assertIsNotNone(first["result_recorded_at"])
            self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()
