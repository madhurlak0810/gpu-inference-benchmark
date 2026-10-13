"""Reduce results/*/summary.csv to one row per setting: mean and std across repeats.

`python -m analysis.summarize` prints the README results tables for the
baseline runs. The optimization experiments (runs named e1_..., e2_...) are
compared against their controls by `python -m analysis.compare`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from bench.config import REPO_ROOT

SWEEP = {"offline": "batch_size", "online": "concurrency", "online_open": "request_rate"}
EXPERIMENT = re.compile(r"e\d+_")  # run names of the optimization experiments
GROUP = ["run", "backend", "model", "input_tokens", "max_new_tokens"]
METRICS = [
    "throughput_tok_s", "latency_p50_ms", "latency_p95_ms", "ttft_p50_ms", "ttft_p95_ms",
    "itl_mean_ms", "peak_mem_gb", "mean_util_pct",
    "itl_p95_ms", "max_gap_p95_ms", "cpu_server_pct", "cpu_client_pct",
    "running_mean", "waiting_mean", "waiting_max", "kv_cache_peak_pct",
]


def load(results_dir: str | Path, experiments: bool = False) -> pd.DataFrame:
    """Every successful (setting, repeat) row from the baseline runs, or from
    the optimization experiments instead; --smoke runs are skipped."""
    paths = sorted(
        p for p in Path(results_dir).glob("*/summary*.csv")
        if not p.parent.name.startswith("_") and bool(EXPERIMENT.match(p.parent.name)) == experiments
    )
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


def offline_table(df: pd.DataFrame) -> str:
    """Hugging Face and vLLM side by side for each model, prompt setting and batch size."""
    agg = aggregate(df, "offline")
    setting = ["model", "input_tokens", "max_new_tokens", "batch_size"]
    wide = agg.pivot(index=setting, columns="backend")
    lines = [
        "| Model | Tokens in / out | Batch | HF tok/s | vLLM tok/s | vLLM ÷ HF "
        "| HF p95 latency (ms) | vLLM p95 latency (ms) | HF peak memory (GB) | vLLM peak memory (GB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (model, tokens_in, tokens_out, batch), row in wide.iterrows():
        hf, vllm = row["throughput_tok_s_mean"]["hf"], row["throughput_tok_s_mean"]["vllm"]
        lines.append(
            f"| {model.split('/')[-1]} | {tokens_in:,} / {tokens_out} | {batch} "
            f"| {_pm(hf, row['throughput_tok_s_std']['hf'])} | {_pm(vllm, row['throughput_tok_s_std']['vllm'])} "
            f"| {vllm / hf:.1f}× | {row['latency_p95_ms_mean']['hf']:,.0f} | {row['latency_p95_ms_mean']['vllm']:,.0f} "
            f"| {row['peak_mem_gb_mean']['hf']:.1f} | {row['peak_mem_gb_mean']['vllm']:.1f} |"
        )
    return "\n".join(lines)


def online_table(df: pd.DataFrame) -> str:
    """One row per run and concurrency level."""
    lines = [
        "| Run | Users | Throughput (tok/s) | Time to first token p50 / p95 (ms) | Inter-token latency (ms) "
        "| End-to-end latency p50 / p95 (ms) | Peak GPU memory (GB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in aggregate(df, "online").iterrows():
        lines.append(
            f"| {row['run']} | {row['concurrency']} | {_pm(row['throughput_tok_s_mean'], row['throughput_tok_s_std'])} "
            f"| {row['ttft_p50_ms_mean']:,.0f} / {row['ttft_p95_ms_mean']:,.0f} | {row['itl_mean_ms_mean']:.1f} "
            f"| {row['latency_p50_ms_mean']:,.0f} / {row['latency_p95_ms_mean']:,.0f} | {row['peak_mem_gb_mean']:.1f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the results tables as Markdown.")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    results = load(ap.parse_args().results_dir)
    sys.stdout.reconfigure(encoding="utf-8")  # so "±" survives a redirect on Windows
    if results.empty:
        print("no results yet")
    else:
        print(offline_table(results), online_table(results), sep="\n" * 2)
