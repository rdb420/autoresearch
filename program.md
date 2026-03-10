# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```bash
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```bash
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```bash
commit val_bpb memory_gb status description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```bash
commit val_bpb memory_gb status description
a1b2c3d 0.997900 44.0 keep baseline
b2c3d4e 0.993200 44.2 keep increase LR to 0.04
c3d4e5f 1.005000 44.0 discard switch to GeLU activation
d4e5f6g 0.000000 0.0 crash double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

---

Deep Machine Learning Reference Library:

Use these research resources when brainstorming, when stuck, or researching the best next steps.

- **[Data Science and Machine Learning](Library/Deep-Machine_Learning/Data_Science_And_Machine_Learning/Data_Science_And_Machine_Learning.md)**  
  Covers:  
  - The foundational concepts of data science  
  - Machine learning algorithms  
  - Data preprocessing and feature engineering  
  - Supervised and unsupervised learning  
  - Model evaluation and tuning  
  - Practical case studies and applications

- **[Deep Learning with Python](Library/Deep-Machine_Learning/DEEP_LEARNIN_With_Python/DEEP_LEARNIN_With_Python.md)**  
  Covers:  
  - Introduction to neural networks and deep learning  
  - Keras and TensorFlow basics  
  - Convolutional networks, recurrent networks, and use cases  
  - Practical projects in computer vision and natural language processing  
  - Tips for model optimization and tuning

- **[Deep Learning](Library/Deep-Machine_Learning/Deep_Learning/Deep_Learning.md)**  
  Covers:  
  - Fundamentals of deep learning and representation learning  
  - Mathematical and conceptual building blocks  
  - Common architectures: MLPs, CNNs, RNNs  
  - Optimization, regularization, and unsupervised learning strategies  
  - Applications to vision, language, and generative models

- **[Deep Learning (2008-12-01 version)](Library/Deep-Machine_Learning/DEEP_LEARNING_2008-12-01/DEEP_LEARNING_2008-12-01.md)**  
  Covers:  
  - Early foundational perspective on deep learning  
  - Historical context and development of neural network models  
  - Key algorithms and architectures as of 2008  
  - Insights and trends in neural computation

- **[Deep Learning Methods and Applications](Library/Deep-Machine_Learning/Deep_Learning_Methods_And_Applications/Deep_Learning_Methods_And_Applications.md)**  
  Covers:  
  - Survey of deep learning techniques  
  - Application areas: speech recognition, computer vision, NLP  
  - Case studies and system design  
  - Comparison of classical and deep learning approaches

- **[Foundations Of Machine Learning](Library/Deep-Machine_Learning/Foundations_Of_Machine_Learning/Foundations_Of_Machine_Learning.md)**  
  Covers:  
  - Theoretical underpinnings of ML  
  - Statistical learning theory  
  - PAC learning, VC dimensions  
  - Linear and nonlinear models  
  - Algorithmic frameworks and proofs

- **[Introduction To Machine Learning](Library/Deep-Machine_Learning/Introduction_To_Machine_Learning/Introduction_To_Machine_Learning.md)**  
  Covers:  
  - Beginner-friendly intro to ML concepts  
  - Overview of supervised and unsupervised methods  
  - Core algorithms (SVM, decision trees, clustering)  
  - Model selection and validation  
  - Hands-on practical advice

- **[Machine Learning Handbook](Library/Deep-Machine_Learning/Machine_Learning_Handbook/Machine_Learning_Handbook.md)**  
  Covers:  
  - Comprehensive algorithms reference  
  - Feature extraction and data representation  
  - Ensemble models and hybrid strategies  
  - Practical programming guides and recipes

- **[Pattern Recognition and Machine Learning (C. M. Bishop)](Library/Deep-Machine_Learning/PATTERN_RECOGNITION_AND_MACHINE_LEARNING_CHRISTOPHER_M._BISHOP/PATTERN_RECOGNITION_AND_MACHINE_LEARNING_CHRISTOPHER_M._BISHOP.md)**  
  Covers:  
  - Probability and statistics for ML  
  - Bayesian networks  
  - Kernel methods  
  - Graphical models  
  - Pattern recognition theory and application

- **[Practical Machine Learning: A Beginner’s Guide With Ethical Considerations](Library/Deep-Machine_Learning/Practical_Machine_Learning_A_Beginner’s_Guide_With_Ethical/Practical_Machine_Learning_A_Beginner’s_Guide_With_Ethical.md)**  
  Covers:  
  - ML project workflow from data collection to deployment  
  - Real-world examples and best practices  
  - Ethical concerns in ML and responsible AI  
  - Model transparency and fairness

- **[The Hundred-Page Machine Learning Book](Library/Deep-Machine_Learning/The_Hundred-page/The_Hundred-page.md)**  
  Covers:  
  - Compressed yet thorough overview of ML fundamentals  
  - Algorithms and math essentials  
  - Neural networks, SVMs, decision trees, boosting  
  - Popular models and their use-cases  
  - Interview-style Q&A and summaries

- **[Understanding Machine Learning: From Theory to Algorithms](Library/Deep-Machine_Learning/Understanding_Machine_Learning_From_Theory_To_Algorithms/Understanding_Machine_Learning_From_Theory_To_Algorithms.md)**  
  Covers:  
  - Mathematical foundation of ML  
  - Generalization, capacity, and learnability  
  - Major algorithms and proof techniques  
  - Optimization, convexity, and practical considerations

// Each entry includes a high-level chapter/topic list (from the book's table of contents or preface) and a link to a `.md` file in the corresponding folder.  
