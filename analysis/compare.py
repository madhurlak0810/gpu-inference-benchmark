"""Compare an optimization experiment's runs with their control.

`python -m analysis.compare e1` prints one Markdown table per load mode: every
run named e1_..., its metrics as mean ± standard deviation across repeats, and
the change from the control run. A change counts only if it clears the noise
floor: more than twice the larger standard deviation of the two runs.

A run's control is the nearest `<prefix>_control`: e4_hf_static is compared
with e4_hf_control, e1_seqs_016 with e1_control.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from bench.config import REPO_ROOT

from .summarize import SWEEP, _pm, aggregate, load

NOISE_FACTOR = 2
SUFFIX = {"offline": "", "online": "", "online_open": "_open"}
TITLE = {"offline": "Offline", "online": "Closed loop", "online_open": "Open loop"}
SWEEP_LABEL = {"batch_size": "Batch", "concurrency": "Users", "request_rate": "Requests/s"}
# metric -> (column heading, decimals)
COLUMNS = {
    "offline": {
        "throughput_tok_s": ("Throughput (tok/s)", 0), "mean_util_pct": ("GPU utilization (%)", 0),
        "latency_p95_ms": ("p95 latency (ms)", 0), "peak_mem_gb": ("Peak memory (GB)", 1),
    },
    "online": {
        "throughput_tok_s": ("Throughput (tok/s)", 0), "ttft_p50_ms": ("TTFT p50 (ms)", 0),
        "ttft_p95_ms": ("TTFT p95 (ms)", 0), "itl_mean_ms": ("ITL mean (ms)", 1),
        "max_gap_p95_ms": ("Longest token gap p95 (ms)", 0), "latency_p95_ms": ("E2E p95 (ms)", 0),
        "waiting_mean": ("Queued requests (mean)", 1), "kv_cache_peak_pct": ("KV cache peak (%)", 1),
        "peak_mem_gb": ("Peak memory (GB)", 1),
    },
}
COLUMNS["online_open"] = COLUMNS["online"]


def control_of(run: str, runs: set[str]) -> str | None:
    parts = run.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:i]) + "_control"
        if candidate in runs:
            return None if candidate == run else candidate
    return None


def _cell(row, base, metric: str, digits: int) -> str:
    mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
    if pd.isna(mean):
        return "–"
    text = _pm(mean, std, digits)
    if base is None or pd.isna(base[f"{metric}_mean"]) or not base[f"{metric}_mean"]:
        return text
    delta = mean - base[f"{metric}_mean"]
    floor = NOISE_FACTOR * max(std if pd.notna(std) else 0, base[f"{metric}_std"] if pd.notna(base[f"{metric}_std"]) else 0)
    change = f"{delta / base[f'{metric}_mean']:+.0%}"
    # Bold marks a change beyond the noise floor.
    return f"{text} ({f'**{change}**' if abs(delta) > floor else change})"


def table(agg: pd.DataFrame, mode: str) -> str:
    sweep = SWEEP[mode]
    columns = {m: c for m, c in COLUMNS[mode].items() if f"{m}_mean" in agg and agg[f"{m}_mean"].notna().any()}
    runs = set(agg["run"])
    by_key = {(r["run"], r[sweep]): r for _, r in agg.iterrows()}
    lines = [
        f"| {SWEEP_LABEL[sweep]} | Run | " + " | ".join(heading for heading, _ in columns.values()) + " |",
        "| ---: | --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    # Controls first within each setting, then the runs compared with them.
    order = sorted(by_key, key=lambda k: (k[1], control_of(k[0], runs) or k[0], control_of(k[0], runs) is not None, k[0]))
    for run, setting in order:
        base = by_key.get((control_of(run, runs), setting))
        cells = [_cell(by_key[run, setting], base, metric, digits) for metric, (_, digits) in columns.items()]
        lines.append(f"| {setting} | {run} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def output_match(results_dir: Path, run: str, control: str, mode: str) -> str | None:
    """Share of requests (offline: batches) whose output fingerprint equals the control's."""
    frames = []
    for name in (run, control):
        path = results_dir / name / f"raw{SUFFIX[mode]}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "output_sha" not in df:
            return None
        frames.append(df)
    unit = "batch_idx" if mode == "offline" else "request"
    keys = [SWEEP[mode], "repeat", unit] + (["prompt"] if "prompt" in frames[0] else [])
    both = frames[0].merge(frames[1], on=keys, suffixes=("", "_control"))
    both = both.dropna(subset=["output_sha", "output_sha_control"])
    if both.empty:
        return None
    same = (both["output_sha"] == both["output_sha_control"]).sum()
    return f"{run} vs {control}: {same:,} of {len(both):,} outputs identical ({same / len(both):.1%})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare one experiment's runs with their control.")
    ap.add_argument("experiment", help="run-name prefix, e.g. e1")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    results_dir = Path(args.results_dir)
    df = load(results_dir, experiments=True)
    if not df.empty:
        df = df[df["run"].str.startswith(args.experiment + "_")]
    if df.empty:
        print(f"no results for {args.experiment} under {results_dir}")
        return
    for mode in SWEEP:
        agg = aggregate(df, mode)
        if agg.empty:
            continue
        runs = set(agg["run"])
        print(f"### {TITLE[mode]}\n\n{table(agg, mode)}\n")
        for run in sorted(runs):
            control = control_of(run, runs)
            line = control and output_match(results_dir, run, control, mode)
            if line:
                print(f"- {line}")
        print()


if __name__ == "__main__":
    main()
