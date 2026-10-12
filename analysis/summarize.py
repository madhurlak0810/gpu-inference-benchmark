"""Reduce results/*/summary.csv to one row per setting: mean and std across repeats.

`python -m analysis.summarize` prints the README results table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from bench.config import REPO_ROOT

SWEEP = {"offline": "batch_size", "online": "concurrency"}
GROUP = ["run", "backend", "model", "input_tokens", "max_new_tokens"]
METRICS = ["throughput_tok_s", "latency_p95_ms", "ttft_p95_ms", "itl_mean_ms", "peak_mem_gb", "mean_util_pct"]


def load(results_dir: str | Path) -> pd.DataFrame:
    """Every successful (setting, repeat) row from every run; --smoke runs are skipped."""
    paths = sorted(p for p in Path(results_dir).glob("*/summary.csv") if not p.parent.name.startswith("_"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df[df["status"] == "ok"]


def aggregate(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One row per setting of one mode, with <metric>_mean and <metric>_std columns."""
    if df.empty:
        return df
    sweep = SWEEP[mode]
    df = df[df["mode"] == mode]
    if df.empty:
        return df
    df = df.astype({sweep: int})
    metrics = [m for m in METRICS if m in df.columns and df[m].notna().any()]
    grouped = df.groupby(GROUP + [sweep])
    out = grouped[metrics].agg(["mean", "std"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    out["repeats"] = grouped.size()
    return out.reset_index()


def _pm(mean: float, std: float, digits: int = 0) -> str:
    if pd.isna(std):
        return f"{mean:,.{digits}f}"
    return f"{mean:,.{digits}f} ± {std:,.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    lines = [
        "| Run | Backend | Model | Tokens in / out | Setting | Throughput (tok/s) | p95 latency (ms) | Peak GPU memory (GB) |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for mode, sweep in SWEEP.items():
        unit = "batch {}" if mode == "offline" else "{} users"
        for _, row in aggregate(df, mode).iterrows():
            lines.append(
                f"| {row['run']} | {row['backend']} | {row['model'].split('/')[-1]} "
                f"| {row['input_tokens']} / {row['max_new_tokens']} | {unit.format(row[sweep])} "
                f"| {_pm(row['throughput_tok_s_mean'], row['throughput_tok_s_std'])} "
                f"| {_pm(row['latency_p95_ms_mean'], row['latency_p95_ms_std'])} "
                f"| {row['peak_mem_gb_mean']:.2f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the results table as Markdown.")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    results = load(ap.parse_args().results_dir)
    sys.stdout.reconfigure(encoding="utf-8")  # so "±" survives a redirect on Windows
    print("no results yet" if results.empty else markdown_table(results))
