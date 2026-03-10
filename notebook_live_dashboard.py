from __future__ import annotations

from pathlib import Path
from time import sleep
from typing import Any

from run_log_viz import discover_cursor_terminal_logs, load_latest_log, steps_to_frame


def load_current_log(
    run_log_path: str | Path = "run.log",
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    run_log_path = Path(run_log_path)
    workspace_root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    candidate_paths = [run_log_path, *discover_cursor_terminal_logs(workspace_root)]
    return load_latest_log(candidate_paths)


def _format_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def _status_markdown(parsed: dict, run_log_path: str | Path = "run.log") -> str:
    summary = parsed.get("summary", {})
    steps = parsed.get("steps", [])
    path = parsed.get("path", Path(run_log_path))

    if not parsed.get("exists", False):
        return (
            f"## Waiting for `{Path(path).name}`\n\n"
            "No log file exists yet. Start training in another terminal with "
            "`uv run train.py > run.log 2>&1`, then rerun or keep the live monitor cell running."
        )

    if not steps:
        return (
            f"## Watching `{Path(path).name}`\n\n"
            "The file exists, but no step lines have been parsed yet. "
            "This usually means training is still starting up."
        )

    latest = steps[-1]
    lines = [
        "## Live Training Status",
        "",
        f"- Source: `{Path(path).name}`",
        f"- Step: `{latest['step']:,}`",
        f"- Progress: `{latest['progress_percent']:.1f}%`",
        f"- Latest loss: `{latest['loss']:.6f}`",
        f"- Throughput: `{_format_int(latest['tok_per_sec'])}` tok/s",
        f"- Step time: `{latest['dt_ms']:.1f}` ms",
        f"- MFU: `{latest['mfu_percent']:.1f}%`",
        f"- LR multiplier: `{latest['lrm']:.2f}`",
        f"- Remaining: `{latest['remaining_seconds']}` s",
    ]

    if summary:
        lines.append("")
        lines.append("### Final Summary")
        if "val_bpb" in summary:
            lines.append(f"- val_bpb: `{summary['val_bpb']:.6f}`")
        if "training_seconds" in summary:
            lines.append(f"- training_seconds: `{summary['training_seconds']:.1f}`")
        if "peak_vram_mb" in summary:
            lines.append(f"- peak_vram_mb: `{summary['peak_vram_mb']:.1f}`")
        if "mfu_percent" in summary:
            lines.append(f"- mfu_percent: `{summary['mfu_percent']:.2f}`")

    return "\n".join(lines)


def _display_markdown(markdown_text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(markdown_text))
    except ModuleNotFoundError:
        print(markdown_text)


def render_dashboard(parsed: dict, recent_window: int = 150, run_log_path: str | Path = "run.log") -> None:
    import matplotlib.pyplot as plt

    _display_markdown(_status_markdown(parsed, run_log_path=run_log_path))

    steps = parsed.get("steps", [])
    if not steps:
        return

    df = steps_to_frame(steps)
    recent = df.tail(min(len(df), recent_window))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_loss, ax_recent, ax_speed, ax_eff = axes.flatten()

    ax_loss.plot(df["step"], df["loss"], color="#1f77b4", alpha=0.35, linewidth=1.2, label="Loss")
    ax_loss.plot(
        df["step"],
        df["loss"].rolling(25, min_periods=1).mean(),
        color="#0b3c6f",
        linewidth=2.2,
        label="Rolling mean (25)",
    )
    ax_loss.set_title("Loss Across Training")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")

    ax_recent.plot(recent["step"], recent["loss"], color="#d62728", linewidth=1.8)
    ax_recent.scatter(recent["step"].iloc[-1], recent["loss"].iloc[-1], color="#d62728", s=45, zorder=3)
    ax_recent.set_title(f"Recent Loss (last {len(recent)} steps)")
    ax_recent.set_xlabel("Step")
    ax_recent.set_ylabel("Loss")

    ax_speed.plot(df["step"], df["tok_per_sec"], color="#2ca02c", linewidth=1.5, label="tok/sec")
    ax_speed.set_title("Throughput")
    ax_speed.set_xlabel("Step")
    ax_speed.set_ylabel("Tokens / sec")
    ax_speed.ticklabel_format(style="plain", axis="y")

    ax_dt = ax_speed.twinx()
    ax_dt.plot(df["step"], df["dt_ms"], color="#9467bd", alpha=0.55, linewidth=1.2, label="dt (ms)")
    ax_dt.set_ylabel("Step time (ms)")

    speed_handles, speed_labels = ax_speed.get_legend_handles_labels()
    dt_handles, dt_labels = ax_dt.get_legend_handles_labels()
    ax_speed.legend(speed_handles + dt_handles, speed_labels + dt_labels, loc="upper right")

    ax_eff.plot(df["step"], df["mfu_percent"], color="#ff7f0e", linewidth=1.6, label="MFU %")
    ax_eff.set_title("Efficiency and LR Multiplier")
    ax_eff.set_xlabel("Step")
    ax_eff.set_ylabel("MFU %")

    ax_lrm = ax_eff.twinx()
    ax_lrm.plot(df["step"], df["lrm"], color="#8c564b", alpha=0.75, linewidth=1.4, label="LR multiplier")
    ax_lrm.set_ylabel("LR multiplier")

    eff_handles, eff_labels = ax_eff.get_legend_handles_labels()
    lrm_handles, lrm_labels = ax_lrm.get_legend_handles_labels()
    ax_eff.legend(eff_handles + lrm_handles, eff_labels + lrm_labels, loc="upper right")

    latest = df.iloc[-1]
    fig.suptitle(
        "Live training dashboard"
        f" | step {int(latest['step']):,}"
        f" | loss {latest['loss']:.4f}"
        f" | tok/sec {_format_int(latest['tok_per_sec'])}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def render_snapshot(
    run_log_path: str | Path = "run.log",
    workspace_root: str | Path | None = None,
    recent_window: int = 150,
) -> dict[str, Any]:
    parsed = load_current_log(run_log_path=run_log_path, workspace_root=workspace_root)
    render_dashboard(parsed, recent_window=recent_window, run_log_path=run_log_path)
    return parsed


def monitor_live(
    run_log_path: str | Path = "run.log",
    workspace_root: str | Path | None = None,
    refresh_seconds: int = 2,
    recent_window: int = 150,
) -> None:
    try:
        from IPython.display import clear_output
    except ModuleNotFoundError:
        def clear_output(*_: Any, **__: Any) -> None:
            return None

    try:
        while True:
            clear_output(wait=True)
            parsed = load_current_log(run_log_path=run_log_path, workspace_root=workspace_root)
            render_dashboard(parsed, recent_window=recent_window, run_log_path=run_log_path)
            print(
                f"Polling live training output every {refresh_seconds}s from {Path(parsed['path']).name}. "
                "Interrupt the cell to stop the live monitor."
            )
            sleep(refresh_seconds)
    except KeyboardInterrupt:
        clear_output(wait=True)
        render_snapshot(run_log_path=run_log_path, workspace_root=workspace_root, recent_window=recent_window)
        print("Live monitor stopped.")
