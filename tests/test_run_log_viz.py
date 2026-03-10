import os
import tempfile
import unittest
from pathlib import Path

from run_log_viz import load_latest_log, parse_run_log_text


SAMPLE_LOG = """Vocab size: 8,192
Model config: {'sequence_len': 2048}
step 00000 (0.0%) | loss: 9.011034 | lrm: 1.00 | dt: 6711ms | tok/sec: 4,882 | mfu: 0.1% | epoch: 1 | remaining: 300s
step 00001 (0.0%) | loss: 8.919419 | lrm: 1.00 | dt: 145ms | tok/sec: 225,916 | mfu: 2.7% | epoch: 1 | remaining: 300s
step 00002 (0.0%) | loss: 8.702668 | lrm: 0.98 | dt: 136ms | tok/sec: 241,210 | mfu: 2.9% | epoch: 1 | remaining: 299s
---
val_bpb:          1.142676
training_seconds: 300.1
peak_vram_mb:     4096.0
mfu_percent:      39.8
"""


class ParseRunLogTextTests(unittest.TestCase):
    def test_parses_step_metrics_and_summary(self):
        parsed = parse_run_log_text(SAMPLE_LOG)

        self.assertEqual(len(parsed["steps"]), 3)
        self.assertEqual(parsed["steps"][0]["step"], 0)
        self.assertAlmostEqual(parsed["steps"][1]["loss"], 8.919419)
        self.assertEqual(parsed["steps"][1]["tok_per_sec"], 225916)
        self.assertAlmostEqual(parsed["steps"][2]["lrm"], 0.98)
        self.assertEqual(parsed["steps"][2]["remaining_seconds"], 299)

        self.assertAlmostEqual(parsed["summary"]["val_bpb"], 1.142676)
        self.assertAlmostEqual(parsed["summary"]["training_seconds"], 300.1)
        self.assertAlmostEqual(parsed["summary"]["peak_vram_mb"], 4096.0)
        self.assertAlmostEqual(parsed["summary"]["mfu_percent"], 39.8)

    def test_handles_partial_log_without_summary(self):
        partial = (
            "step 00001 (0.0%) | loss: 8.919419 | lrm: 1.00 | dt: 145ms | "
            "tok/sec: 225,916 | mfu: 2.7% | epoch: 1 | remaining: 300s\n"
        )

        parsed = parse_run_log_text(partial)

        self.assertEqual(len(parsed["steps"]), 1)
        self.assertEqual(parsed["summary"], {})

    def test_ignores_unrecognized_lines(self):
        parsed = parse_run_log_text("hello\nworld\n")
        self.assertEqual(parsed["steps"], [])
        self.assertEqual(parsed["summary"], {})

    def test_parses_terminal_style_concatenated_steps(self):
        terminal_text = (
            "---\n"
            "pid: 177216\n"
            "cwd: /tmp/project\n"
            "last_command: uv run train.py\n"
            "last_exit_code: 0\n"
            "---\n"
            "step 01259 (55.7%) | loss: 3.518488 | lrm: 0.89 | dt: 134ms | "
            "tok/sec: 243,809 | mfu: 3.0% | epoch: 1 | remaining: 133s"
            "step 01260 (55.8%) | loss: 3.527451 | lrm: 0.88 | dt: 135ms | "
            "tok/sec: 243,201 | mfu: 3.0% | epoch: 1 | remaining: 132s"
        )

        parsed = parse_run_log_text(terminal_text)

        self.assertEqual(len(parsed["steps"]), 2)
        self.assertEqual(parsed["steps"][0]["step"], 1259)
        self.assertEqual(parsed["steps"][1]["step"], 1260)
        self.assertEqual(parsed["steps"][1]["remaining_seconds"], 132)

    def test_load_latest_log_prefers_newer_live_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_log = tmp_path / "run.log"
            terminal_log = tmp_path / "terminal.txt"

            run_log.write_text(
                "step 02233 (100.0%) | loss: 3.222948 | lrm: 0.00 | dt: 136ms | "
                "tok/sec: 241,787 | mfu: 2.9% | epoch: 1 | remaining: 0s\n"
            )
            terminal_log.write_text(
                "step 00010 (0.4%) | loss: 6.239288 | lrm: 1.00 | dt: 132ms | "
                "tok/sec: 247,695 | mfu: 3.0% | epoch: 1 | remaining: 299s\n"
            )

            os.utime(run_log, (1, 1))
            os.utime(terminal_log, None)

            loaded = load_latest_log([run_log, terminal_log])

            self.assertTrue(loaded["exists"])
            self.assertEqual(loaded["path"], terminal_log)
            self.assertEqual(loaded["steps"][-1]["step"], 10)


if __name__ == "__main__":
    unittest.main()
