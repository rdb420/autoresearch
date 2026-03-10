import json
import os
import unittest
from pathlib import Path


class VisualizeRunLogNotebookTests(unittest.TestCase):
    def test_notebook_entry_points_exist(self):
        from notebook_live_dashboard import monitor_live, render_snapshot

        self.assertTrue(callable(render_snapshot))
        self.assertTrue(callable(monitor_live))

    def test_snapshot_cell_runs_in_fresh_namespace(self):
        os.environ["MPLBACKEND"] = "Agg"
        notebook = json.loads(Path("visualize_run_log.ipynb").read_text())
        snapshot_source = "".join(notebook["cells"][3]["source"])
        namespace = {}

        exec(compile(snapshot_source, "snapshot_cell", "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
