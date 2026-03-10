# Live Run Log Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a live-updating notebook dashboard that visualizes `run.log` while training is in progress.

**Architecture:** Put log parsing and summarization in a small Python helper module so it can be tested outside Jupyter. Keep the notebook focused on configuration, polling, and plotting with `matplotlib` and IPython display utilities.

**Tech Stack:** Python 3.10+, `matplotlib`, `pandas`, built-in `re`, built-in `unittest`, Jupyter notebook cells

---

### Task 1: Add parser tests

**Files:**

- Create: `tests/test_run_log_viz.py`
- Modify: none
- Test: `tests/test_run_log_viz.py`

**Step 1: Write the failing test**

```python
import unittest

from run_log_viz import parse_run_log_text


class ParseRunLogTextTests(unittest.TestCase):
    def test_parses_step_metrics_and_summary(self):
        text = """Vocab size: 8,192
step 00001 (0.0%) | loss: 8.919419 | lrm: 1.00 | dt: 145ms | tok/sec: 225,916 | mfu: 2.7% | epoch: 1 | remaining: 300s
---
val_bpb:          1.142676
peak_vram_mb:     4096.0
"""
        parsed = parse_run_log_text(text)
        self.assertEqual(len(parsed["steps"]), 1)
        self.assertEqual(parsed["summary"]["val_bpb"], 1.142676)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_run_log_viz.py -v`
Expected: FAIL because `run_log_viz` does not exist yet.

**Step 3: Write minimal implementation**

```python
def parse_run_log_text(text):
    return {"steps": [], "summary": {}}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_run_log_viz.py -v`
Expected: PASS after the parser is fully implemented.

**Step 5: Commit**

```bash
git add tests/test_run_log_viz.py run_log_viz.py
git commit -m "feat: add run log parser for live visualization"
```

### Task 2: Implement tested parser helpers

**Files:**

- Create: `run_log_viz.py`
- Modify: none
- Test: `tests/test_run_log_viz.py`

**Step 1: Extend the failing tests**

```python
def test_handles_partial_log_without_summary(self):
    parsed = parse_run_log_text("step 00001 (0.0%) | loss: 8.1 | lrm: 1.00 | dt: 145ms | tok/sec: 200,000 | mfu: 2.7% | epoch: 1 | remaining: 300s")
    self.assertEqual(parsed["summary"], {})
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_run_log_viz.py -v`
Expected: FAIL because partial parsing is not implemented yet.

**Step 3: Write minimal implementation**

```python
# Add regex-based parsing for step lines and footer metrics.
```

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_run_log_viz.py -v`
Expected: PASS with all parser tests green.

**Step 5: Commit**

```bash
git add tests/test_run_log_viz.py run_log_viz.py
git commit -m "feat: implement tested run log parsing helpers"
```

### Task 3: Build the notebook dashboard

**Files:**

- Modify: `visualize_run_log.ipynb`
- Test: manual notebook run using the existing `run.log`

**Step 1: Add notebook setup cell**

```python
from pathlib import Path
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import pandas as pd
import time

from run_log_viz import load_run_log
```

**Step 2: Add live rendering cell**

```python
# Poll run.log, redraw loss/throughput/MFU/LRM charts, and show latest status.
```

**Step 3: Add friendly empty-log and missing-file handling**

```python
# Show a waiting message instead of throwing when run.log is absent or incomplete.
```

**Step 4: Run the notebook against `run.log`**

Run the cells in `visualize_run_log.ipynb`.
Expected: A live dashboard appears and updates while the log grows.

**Step 5: Commit**

```bash
git add visualize_run_log.ipynb
git commit -m "feat: add live notebook dashboard for training log"
```

### Task 4: Verify end-to-end behavior

**Files:**

- Modify: none
- Test: `tests/test_run_log_viz.py`, `visualize_run_log.ipynb`

**Step 1: Run automated tests**

Run: `python3 -m unittest tests/test_run_log_viz.py -v`
Expected: PASS.

**Step 2: Validate notebook behavior**

Run the notebook cells against:

- a complete `run.log`
- a growing `run.log`
- an empty or missing `run.log`

Expected: Charts render, polling continues cleanly, and summary metrics appear when available.

**Step 3: Check for diagnostics**

Run relevant lints/diagnostics for the new Python helper if available.
Expected: No new errors introduced.

**Step 4: Commit**

```bash
git add run_log_viz.py tests/test_run_log_viz.py visualize_run_log.ipynb docs/plans/2026-03-11-live-run-log-visualization.md
git commit -m "feat: add live training log visualization workflow"
```
