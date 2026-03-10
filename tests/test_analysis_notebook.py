import json
import os
import unittest
from pathlib import Path


class AnalysisNotebookTests(unittest.TestCase):
    def test_plot_cell_uses_f_string_for_truncation(self):
        notebook = json.loads(Path("analysis.ipynb").read_text())
        plot_source = "".join(notebook["cells"][5]["source"])

        self.assertNotIn('desc = desc[:42] + "..."', plot_source)
        self.assertIn('desc = f"{desc[:42]}..."', plot_source)

    def test_summary_cell_does_not_use_unused_enumerate(self):
        notebook = json.loads(Path("analysis.ipynb").read_text())
        summary_source = "".join(notebook["cells"][7]["source"])

        self.assertNotIn("for i, (_, row) in enumerate(kept_sorted.iterrows()):", summary_source)
        self.assertIn("for _, row in kept_sorted.iterrows():", summary_source)

    def test_summary_cell_runs_in_fresh_namespace(self):
        os.environ["MPLBACKEND"] = "Agg"
        notebook = json.loads(Path("analysis.ipynb").read_text())
        summary_source = "".join(notebook["cells"][7]["source"])
        namespace = {}

        exec(compile(summary_source, "analysis_summary_cell", "exec"), namespace)

    def test_top_hits_cell_runs_in_fresh_namespace(self):
        notebook = json.loads(Path("analysis.ipynb").read_text())
        top_hits_source = "".join(notebook["cells"][9]["source"])
        namespace = {}

        exec(compile(top_hits_source, "analysis_top_hits_cell", "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
