from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_results(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = df[df["status"] != "CRASH"].copy().reset_index(drop=True)
    kept = df[df["status"] == "KEEP"].copy()
    return {"valid": valid, "kept": kept}


def non_crash_baseline_bpb(df: pd.DataFrame) -> float | None:
    valid = summarize_results(df)["valid"]
    return None if valid.empty else float(valid.iloc[0]["val_bpb"])


def best_kept_row(df: pd.DataFrame) -> pd.Series | None:
    kept = summarize_results(df)["kept"]
    return None if kept.empty else kept.loc[kept["val_bpb"].idxmin()]


def has_non_crash_results(df: pd.DataFrame) -> bool:
    return not summarize_results(df)["valid"].empty


def has_kept_results(df: pd.DataFrame) -> bool:
    return not summarize_results(df)["kept"].empty


def safe_total_improvement(df: pd.DataFrame) -> float | None:
    baseline = non_crash_baseline_bpb(df)
    best = best_kept_row(df)
    if baseline is None or best is None:
        return None
    return float(baseline - best["val_bpb"])


def summary_message(df: pd.DataFrame) -> str | None:
    if not has_non_crash_results(df):
        return "No non-crash experiments yet, so baseline and progress plots are unavailable."
    if not has_kept_results(df):
        return "No kept improvements yet. The notebook will show the available non-crash baseline data only."
    return None
