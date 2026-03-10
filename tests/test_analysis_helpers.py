import unittest

import pandas as pd

from analysis_helpers import best_kept_row, non_crash_baseline_bpb, summarize_results


class AnalysisHelpersTests(unittest.TestCase):
    def test_crash_only_results_have_no_non_crash_baseline_or_best_kept(self):
        df = pd.DataFrame(
            [
                {
                    "commit": "c12eef7",
                    "val_bpb": 0.0,
                    "memory_gb": 0.0,
                    "status": "CRASH",
                    "description": "baseline master config OOM on 8GB GPU",
                }
            ]
        )

        summary = summarize_results(df)

        self.assertIsNone(non_crash_baseline_bpb(df))
        self.assertIsNone(best_kept_row(df))
        self.assertTrue(summary["valid"].empty)
        self.assertTrue(summary["kept"].empty)

    def test_mixed_results_return_expected_baseline_and_best_kept(self):
        df = pd.DataFrame(
            [
                {"commit": "a", "val_bpb": 0.0, "memory_gb": 0.0, "status": "CRASH", "description": "bad"},
                {"commit": "b", "val_bpb": 1.20, "memory_gb": 4.0, "status": "DISCARD", "description": "baseline"},
                {"commit": "c", "val_bpb": 1.10, "memory_gb": 4.2, "status": "KEEP", "description": "improve"},
            ]
        )

        summary = summarize_results(df)
        best = best_kept_row(df)

        self.assertEqual(non_crash_baseline_bpb(df), 1.20)
        self.assertEqual(len(summary["valid"]), 2)
        self.assertEqual(len(summary["kept"]), 1)
        self.assertEqual(best["description"], "improve")


if __name__ == "__main__":
    unittest.main()
