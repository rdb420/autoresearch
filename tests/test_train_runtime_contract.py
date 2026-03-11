import unittest
from pathlib import Path


class TrainRuntimeContractTests(unittest.TestCase):
    def test_fast_fail_uses_sys_exit_not_builtin_exit(self):
        source = Path("train.py").read_text()

        self.assertIn("sys.exit(1)", source)
        self.assertNotIn("\n        exit(1)", source)


if __name__ == "__main__":
    unittest.main()
