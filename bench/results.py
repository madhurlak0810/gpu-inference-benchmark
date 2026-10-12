"""On-disk results for one run: results/<run>/{raw,summary,gpu_samples}.csv.

Rows are appended after every repeat, so a session that dies mid-run loses at
most the repeat in progress and the next invocation resumes after it.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ("raw.csv", "summary.csv", "gpu_samples.csv")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def percentiles(values_ms: list[float], name: str) -> dict:
    p50, p95, p99 = np.percentile(values_ms, [50, 95, 99])
    return {f"{name}_p50_ms": round(p50, 3), f"{name}_p95_ms": round(p95, 3), f"{name}_p99_ms": round(p99, 3)}


class RunStore:
    def __init__(self, out_dir: Path, key: str):
        self.dir = Path(out_dir)
        self.key = key  # the swept setting: batch_size (offline) or concurrency (online)
        self.dir.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        for name in FILES:
            (self.dir / name).unlink(missing_ok=True)

    def append(self, filename: str, rows: list[dict]) -> None:
        if not rows:
            return
        path = self.dir / filename
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if is_new:
                writer.writeheader()
            writer.writerows(rows)

    def _summary(self) -> pd.DataFrame:
        path = self.dir / "summary.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def completed(self) -> set[tuple[int, int]]:
        """(setting, repeat) pairs already on disk."""
        df = self._summary()
        if df.empty:
            return set()
        return {(int(s), int(r)) for s, r in zip(df[self.key], df["repeat"])}

    def spread(self) -> dict[int, float]:
        """std / mean of throughput across repeats, per setting."""
        df = self._summary()
        if df.empty:
            return {}
        grouped = df[df["status"] == "ok"].groupby(self.key)["throughput_tok_s"]
        return {int(k): float(v) for k, v in (grouped.std() / grouped.mean()).dropna().items()}

    def noisy(self, max_spread: float) -> list[int]:
        return sorted(s for s, v in self.spread().items() if v > max_spread)

    def warn_if_noisy(self, setting: int, max_spread: float) -> None:
        value = self.spread().get(setting)
        if value is not None and value > max_spread:
            print(f"  ! {self.key}={setting}: spread {value:.1%} exceeds {max_spread:.0%}; rerun with --redo-noisy")

    def drop(self, settings: list[int]) -> None:
        """Forget every row of the given settings so they run again."""
        for name in FILES:
            path = self.dir / name
            if path.exists():
                df = pd.read_csv(path)
                df[~df[self.key].isin(settings)].to_csv(path, index=False)
