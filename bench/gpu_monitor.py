"""Background NVML sampler for GPU memory and utilization.

Reads the whole device rather than this process, so it also covers a model
served from another process (the vLLM server in online mode).
"""
from __future__ import annotations

import threading
import time

import pynvml


class GpuMonitor:
    """Context manager that samples one GPU every `interval_s` seconds.

    `samples` holds (seconds since start, memory used in MB, utilization %).
    A sample is taken on entry and on exit, so runs shorter than the interval
    still get two.
    """

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.5):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        self.samples.append((time.perf_counter() - self._t0, mem.used / 2**20, util.gpu))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> "GpuMonitor":
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._t0 = time.perf_counter()
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()
        pynvml.nvmlShutdown()

    @property
    def peak_mem_gb(self) -> float:
        return max(s[1] for s in self.samples) / 1024

    @property
    def mean_util_pct(self) -> float:
        return sum(s[2] for s in self.samples) / len(self.samples)
