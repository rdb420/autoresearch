import unittest

from run_log_viz import parse_run_log_text


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


if __name__ == "__main__":
    unittest.main()
