import os
import subprocess
import sys
import unittest


class NotebookBackendTests(unittest.TestCase):
    def test_matplotlib_inline_backend_imports(self):
        env = dict(os.environ)
        env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import matplotlib.pyplot as plt; print(plt.get_backend())",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
        self.assertIn("matplotlib_inline", result.stdout)


if __name__ == "__main__":
    unittest.main()
