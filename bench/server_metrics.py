"""Background scraper for the vLLM server's Prometheus /metrics endpoint.

Client-side latency says that a request was slow; the server's own gauges say
why: requests waiting in the scheduler's queue, requests running in the batch,
and how full the KV cache is.
"""
from __future__ import annotations

import threading
import time

import httpx

# Column -> metric names to look for. vLLM has renamed the KV cache gauge
# between releases, so both spellings are accepted.
GAUGES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_pct": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}


def parse(text: str) -> dict[str, float | None]:
    """Reduce one /metrics response to the gauges above; None where absent."""
    seen: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0]
        try:
            # A gauge has one sample per label set (engine, model); add them up.
            seen[name] = seen.get(name, 0.0) + float(value)
        except ValueError:
            continue
    out = {}
    for column, names in GAUGES.items():
        out[column] = next((seen[n] for n in names if n in seen), None)
    if out["kv_cache_pct"] is not None:
        out["kv_cache_pct"] *= 100  # reported as a fraction
    return out


class ServerMetrics:
    """Context manager that scrapes `base_url`/metrics every `interval_s` seconds.

    `samples` holds (seconds since start, parsed gauges). A scrape that fails
    is skipped, so a busy server costs samples rather than the run.
    """

    def __init__(self, base_url: str, interval_s: float = 0.5):
        self.url = f"{base_url}/metrics"
        self.interval_s = interval_s
        self.samples: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        try:
            response = self._client.get(self.url)
            response.raise_for_status()
        except httpx.HTTPError:
            return
        self.samples.append((time.perf_counter() - self._t0, parse(response.text)))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> "ServerMetrics":
        self._client = httpx.Client(timeout=2)
        self._t0 = time.perf_counter()
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._client.close()

    def _values(self, column: str) -> list[float]:
        return [s[column] for _, s in self.samples if s[column] is not None]

    def summary(self) -> dict:
        """Mean and peak of each gauge over the repeat; None if never reported."""
        out = {}
        for column in ("running", "waiting"):
            values = self._values(column)
            out[f"{column}_mean"] = round(sum(values) / len(values), 2) if values else None
            out[f"{column}_max"] = max(values) if values else None
        cache = self._values("kv_cache_pct")
        out["kv_cache_peak_pct"] = round(max(cache), 2) if cache else None
        return out
