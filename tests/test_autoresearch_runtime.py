import os
import tempfile
import unittest
from pathlib import Path

from run_manifest import ensure_run_manifest

from autoresearch_runtime import (
    TeeTextIO,
    complete_run,
    ensure_runtime_state,
    load_current_run,
    load_last_run,
    preferred_log_candidates,
    start_run,
)


SUMMARY_LOG = """step 00001 (0.0%) | loss: 8.919419 | lrm: 1.00 | dt: 145ms | tok/sec: 225,916 | mfu: 2.7% | epoch: 1 | remaining: 300s
---
val_bpb:          1.142676
training_seconds: 300.1
peak_vram_mb:     4096.0
mfu_percent:      39.8
"""


class AutoresearchRuntimeTests(unittest.TestCase):
    def test_tee_text_io_writes_to_all_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            target_a = workspace_root / "a.log"
            target_b = workspace_root / "b.log"

            with target_a.open("w", encoding="utf-8") as handle_a, target_b.open(
                "w", encoding="utf-8"
            ) as handle_b:
                stream = TeeTextIO(handle_a, handle_b)
                stream.write("hello\n")
                stream.flush()

            self.assertEqual(target_a.read_text(), "hello\n")
            self.assertEqual(target_b.read_text(), "hello\n")

    def test_ensure_runtime_state_initializes_control_plane_and_results_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            paths = ensure_runtime_state(workspace_root)

            self.assertTrue(paths["control_dir"].exists())
            self.assertTrue(paths["state_dir"].exists())
            self.assertEqual(
                paths["results_path"].read_text(),
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n",
            )

    def test_start_run_records_current_pointer_and_prefers_manifest_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            run = start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-123",
                pid=4242,
            )
            current = load_current_run(workspace_root)
            repo_run_log = workspace_root / "run.log"
            repo_run_log.write_text("repo root log\n")

            self.assertEqual(current["run_id"], "run-123")
            self.assertEqual(current["status"], "running")
            self.assertEqual(current["pid"], 4242)
            self.assertEqual(preferred_log_candidates(workspace_root)[0], Path(run["stdout_path"]))
            self.assertEqual(preferred_log_candidates(workspace_root)[1], repo_run_log)

    def test_start_run_reconciles_previous_running_pointer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            old_run = start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-old",
                pid=None,
            )
            new_run = start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-new",
                pid=2222,
            )

            last_run = load_last_run(workspace_root)
            self.assertEqual(last_run["run_id"], "run-old")
            self.assertEqual(last_run["status"], "interrupted")
            self.assertEqual(load_current_run(workspace_root)["run_id"], "run-new")
            self.assertEqual(Path(old_run["manifest_path"]).exists(), True)
            self.assertEqual(Path(new_run["manifest_path"]).exists(), True)

    def test_start_run_refuses_to_replace_a_still_live_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-live",
                pid=os.getpid(),
            )

            with self.assertRaises(ValueError):
                start_run(
                    workspace_root,
                    command=["uv", "run", "train.py"],
                    run_id="run-new",
                    pid=None,
                )

    def test_complete_run_records_first_success_as_keep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            run = start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-123",
                pid=4242,
            )
            Path(run["stdout_path"]).write_text(SUMMARY_LOG)

            row = complete_run(workspace_root, "run-123", exit_code=0)
            lines = (workspace_root / "results.tsv").read_text().strip().splitlines()

            self.assertEqual(row["status"], "keep")
            self.assertEqual(len(lines), 2)
            self.assertEqual(load_current_run(workspace_root), None)
            self.assertEqual(load_last_run(workspace_root)["run_id"], "run-123")

    def test_complete_run_records_worse_success_as_discard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            (workspace_root / "results.tsv").write_text(
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
                "aaaaaaa\t1.100000\t4.0\tkeep\tbest so far\n"
            )

            run = start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-123",
                pid=4242,
            )
            Path(run["stdout_path"]).write_text(SUMMARY_LOG)

            row = complete_run(workspace_root, "run-123", exit_code=0)

            self.assertEqual(row["status"], "discard")

    def test_complete_run_records_crash_once_across_retries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            run = start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-123",
                pid=4242,
            )
            Path(run["stdout_path"]).write_text("Traceback\nRuntimeError: boom\n")

            first = complete_run(workspace_root, "run-123", exit_code=1)
            second = complete_run(workspace_root, "run-123", exit_code=1)
            lines = (workspace_root / "results.tsv").read_text().strip().splitlines()

            self.assertEqual(first["status"], "crash")
            self.assertEqual(second, first)
            self.assertEqual(len(lines), 2)

    def test_complete_run_keeps_identical_rows_for_distinct_run_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)

            first_run = start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-123",
                pid=4242,
            )
            Path(first_run["stdout_path"]).write_text(SUMMARY_LOG)
            complete_run(workspace_root, "run-123", exit_code=0)

            second_run = start_run(
                workspace_root,
                command=["uv", "run", "train.py"],
                run_id="run-456",
                pid=4243,
            )
            Path(second_run["stdout_path"]).write_text(SUMMARY_LOG)
            complete_run(workspace_root, "run-456", exit_code=0)

            lines = (workspace_root / "results.tsv").read_text().strip().splitlines()
            self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
