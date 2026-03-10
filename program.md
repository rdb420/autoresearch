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

Once you get confirmation, kick off the research phase.

## Research & Strategy

Before touching any code, invest time in understanding **what is likely to produce the best results**. This is not optional — blind hyperparameter sweeping is wasteful. You are a researcher, not a grid search.

### Research step

1. **Read the reference library**: Skim the Deep Machine Learning Reference Library (linked below) for techniques relevant to small-model, short-budget language model pretraining. Focus on:
   - Optimizer innovations (Muon, schedule-free, etc.)
   - Architecture variants that improve parameter efficiency at small scale
   - Regularization strategies for short training runs
   - Learning rate scheduling best practices
   - Initialization schemes
2. **Search for state-of-the-art**: Use web search, sequential thinking, and the reference library to find recent papers/results on efficient small-scale LM training. Look at competitions like the NanoGPT speedrun for what configurations win.
3. **Analyze the existing codebase**: Understand what `train.py` already does well and where the bottlenecks likely are. Read `prepare.py` for constraints (sequence length, eval metric, time budget). Understand the MuonAdamW optimizer, the architecture choices (value embeddings, sliding windows, residual lambdas), and the training loop.
4. **Analyze prior results**: Read `results.tsv` to understand what has already been tried and what worked/didn't work in previous sessions. Learn from this history — don't repeat failed experiments.
5. **Form a research plan**: Before running any experiment, write a brief research plan as a comment at the top of your first commit message. What are your top 3-5 hypotheses? What order will you test them? Why?

### Use brainstorming and sequential thinking

You have access to brainstorming and sequential thinking tools. **Use them.** Before each experiment (or batch of related experiments), think through:

- What do I expect to happen and why?
- What evidence from prior experiments or literature supports this?
- What's the expected magnitude of improvement?
- What could go wrong?

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**

- Modify `train.py` — **the entire file is in scope.** You can and should make sweeping changes when the evidence supports it: rewrite the architecture, overhaul the optimizer configuration, change the training loop structure, adjust multiple hyperparameters simultaneously, add or remove components. Do NOT limit yourself to tweaking one hyperparameter at a time — that is tedious and slow. A well-reasoned change that touches 10 things at once based on research is far better than 10 isolated +/-10% hyperparameter sweeps.
- **Create or modify any other file in the repo** (except the three protected files listed below). You can write helper scripts, analysis tools, data exploration utilities, configuration files, notes — whatever helps you research and iterate.
- **Use any tool, API, or capability at your disposal** to do research and inform your experiments. This includes but is not limited to:
  - Web search to find papers, blog posts, competition results, and state-of-the-art techniques
  - MCP tools (sequential thinking, memory, fetch, etc.)
  - Writing scripts to analyze training logs, visualize loss curves, compare experiments
  - Creating embeddings or vector representations of experiment results to find patterns
  - Using the reference library (linked below) to look up techniques
  - Writing helper utilities that `train.py` can import (as long as they only use packages already in `pyproject.toml`)
  - Any other creative approach to understanding the problem space and finding better solutions
- **Think of yourself as a full research engineer**, not just a config tweaker. Your job is to innovate, iterate, and converge on the best possible training configuration through whatever means necessary.

**What you CANNOT do — three protected files:**

Only three files in this repository are OFF LIMITS for agent edits. These are edited by the human (via a separate assistant if needed):

1. `program.md` — this file. The experiment protocol. Read-only for the agent.
2. `prepare.py` — the fixed evaluation, data loading, tokenizer, and training constants. Contains the ground truth `evaluate_bpb` metric. Read-only for the agent.
3. `analysis.ipynb` — the human's analysis notebook. Read-only for the agent.

Everything else is fair game. You may read all three protected files for context — you just cannot modify them.

**Additional constraint:** Do not install new packages or add dependencies beyond what's already in `pyproject.toml`.

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

## The scientific method — hypothesis-driven experimentation

You are not a random search. You are a **scientist**. Every experiment must follow this structure:

### Before each experiment: Hypothesis

1. **State your reasoning**: Why are you trying this specific change? What evidence (from literature, prior experiments, or theoretical understanding) supports it?
2. **Form a hypothesis**: "I predict that [specific change] will [improve/degrade] val_bpb by approximately [magnitude] because [mechanistic reason]."
3. **Identify risks**: What could cause this to fail? (e.g., "increased model size might OOM", "aggressive LR might destabilize training")
4. **Use sequential thinking**: For complex multi-part changes, use the sequential thinking tool to reason through the chain of effects step by step before committing code.

### During each experiment: Execution

Follow the experiment loop below.

### After each experiment: Analysis

1. **Compare outcome to hypothesis**: Did the result match your prediction? Be specific — don't just say "it improved". Say "I predicted ~0.005 improvement but got 0.003".
2. **Explain why or why not**: If the outcome matched, what does that confirm about your model of the system? If it didn't match, what does that tell you? Was your reasoning wrong, or is there an interfering factor?
3. **Extract lessons**: What did this experiment teach you that should inform the next one?
4. **Update your mental model**: Each experiment's output should directly inform what you try next. Build a cumulative understanding — don't treat experiments as independent coin flips.

### Experiment cascading

Experiments are not independent. They form a chain of evidence:

- Experiment N's results should shape Experiment N+1's hypothesis
- If something unexpected happens, investigate why before moving on
- If two changes individually help, consider combining them
- If a change hurts unexpectedly, ask: is it the direction that's wrong, or just the magnitude?

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. **Think first**: Review the git state, prior results, and form your hypothesis (see scientific method above). Use brainstorming and sequential thinking tools.
2. **Modify `train.py`**: Make your changes. You can change anything and everything in this file — architecture, optimizer, hyperparameters, training loop, schedules, all at once if your reasoning supports it.
3. **git commit**: Write a commit message that includes your hypothesis and reasoning (not just "changed X to Y").
4. **Run the experiment**: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. **Read the results**: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. **Handle crashes**: If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. **Analyze**: Compare outcome to hypothesis. Write your analysis in the tsv description and/or commit message. What did you learn?
8. **Record the results** in `results.tsv` (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. **Keep or discard**: If val_bpb improved (lower), keep the git commit. If val_bpb is equal or worse, git reset back to where you started.
10. **Plan next**: Based on what you just learned, form the hypothesis for the next experiment. Go to step 1.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## How the trained model is evaluated

Understanding how you know if a trained model is actually better is critical. Don't just chase a number — understand what it means.

### The metric: val_bpb (validation bits per byte)

The evaluation metric is **bits per byte (BPB)** computed on a held-out validation shard. This is calculated by `evaluate_bpb()` in `prepare.py`:

1. The model runs inference on the fixed validation data (40 * 524288 tokens)
2. Per-token cross-entropy loss (in nats) is computed with `reduction='none'`
3. Special tokens (byte length 0) are excluded from both numerator and denominator
4. Total nats are converted to bits: `total_nats / (log(2) * total_bytes)`

**Lower val_bpb = better model.** A model with lower BPB is more efficiently compressing unseen text, meaning it has learned better representations of language.

### What val_bpb tells you

- **It measures generalization**: The validation shard is never seen during training, so this measures how well the model generalizes
- **It's vocab-size independent**: Unlike perplexity, BPB normalizes by byte count, making it comparable across different tokenizers
- **Magnitude matters**: A drop from 1.15 to 1.14 is meaningful. A drop from 1.15 to 1.149 is marginal. Know the difference.
- **Training loss vs val_bpb gap**: If training loss is much lower than val_bpb, the model is overfitting. If they're close, the model is efficiently using its capacity.

### What val_bpb does NOT tell you

- It doesn't measure task-specific ability (summarization, QA, coding, etc.)
- It doesn't measure coherence over long contexts
- It doesn't measure factual accuracy
- A model with val_bpb of 1.14 is not necessarily "useful" — it's just measurably better at predicting text than one at 1.15

### How to use val_bpb to guide research

- **Track trends, not just individual numbers**: Is your improvement trajectory flattening? Are you seeing diminishing returns on a particular axis (e.g., LR tuning)?
- **Look at auxiliary metrics too**: peak_vram_mb (efficiency), mfu_percent (hardware utilization), num_steps (how many gradient updates fit in 5 min), total_tokens_M (throughput). A model that processes more tokens in the same time budget often wins.
- **Watch for overfitting signals**: If you increase model size but val_bpb doesn't improve proportionally, something is wrong (too little data, too little regularization, poor optimization).
- **Sanity-check with training loss**: Read the smooth training loss from run.log. If it's still decreasing rapidly when training stops, the model would benefit from more compute — try getting more steps per time budget (smaller model, larger batch, faster kernels).

## Prior experiment history

The file `history/` contains session logs from prior agent runs. Before starting, read `results.tsv` for a summary of what has been tried. Key learnings from the Gemini 3.1 Pro session (2026-03-10):

- Baseline val_bpb: 1.222551 (DEPTH=6, default hyperparameters on 8GB GPU)
- Increasing TOTAL_BATCH_SIZE to 2^16 then back to 2^15 with other changes improved to 1.149729
- WEIGHT_DECAY=0.1 was a significant win (1.142998)
- EMBEDDING_LR=0.4 gave a small further win (1.142676) — current best
- Failed experiments: all-long windows, SiLU activation, HEAD_DIM=64, ASPECT_RATIO=128 (OOM), larger MATRIX_LR, smaller UNEMBEDDING_LR
- The agent was too cautious — it mostly tweaked one hyperparameter at a time instead of making bold, research-informed changes

The next agent (CODEX) should learn from this: be bolder, research more, change more at once.

---

## Research Library Index

Use this index during the research setup, hypothesis formation, experiment analysis, and when you get stuck. Do not just read randomly. Pick sources intentionally:

- Use optimization- and training-focused references when deciding learning rates, schedulers, regularization, initialization, and optimizer changes.
- Use theory-focused references when reasoning about why a result did or did not match the hypothesis.
- Use practical references when turning research ideas into concrete `train.py` changes.
- Record which source categories informed each major experiment.

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

Quick usage guide:

- Optimization and training recipes: `Deep Learning with Python`, `Deep Learning`, `Deep Learning Methods and Applications`
- Theory and diagnosis: `Foundations Of Machine Learning`, `Pattern Recognition and Machine Learning`, `Understanding Machine Learning`
- Broad practical refreshers: `Data Science and Machine Learning`, `Introduction To Machine Learning`, `Machine Learning Handbook`, `The Hundred-Page Machine Learning Book`
- Historical perspective and first-principles grounding: `Deep Learning (2008-12-01 version)`
- Workflow and evaluation framing: `Practical Machine Learning: A Beginner's Guide With Ethical Considerations`

Each entry above includes a high-level chapter/topic list and a direct link to the corresponding `.md` file in the library.
