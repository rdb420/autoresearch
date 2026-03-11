import json
import os
import unittest
from pathlib import Path

from notebook_live_dashboard import _status_markdown


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

    def test_waiting_message_uses_direct_train_command(self):
        message = _status_markdown({"exists": False, "summary": {}, "steps": []})

        self.assertIn("`uv run train.py`", message)
        self.assertNotIn("> run.log 2>&1", message)


if __name__ == "__main__":
    unittest.main()
