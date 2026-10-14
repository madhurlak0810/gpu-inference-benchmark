"""Host CPU utilization for groups of cores, read from /proc/stat.

The load client shares the machine with the server. With the two pinned to
separate cores, this reports how busy each side's cores were during a repeat,
so a run where the client itself was the bottleneck can be flagged.
"""
from __future__ import annotations

from pathlib import Path


def _read() -> dict[int, tuple[int, int]]:
    """Core index -> (busy jiffies, total jiffies) since boot."""
    cores = {}
    for line in Path("/proc/stat").read_text(encoding="ascii").splitlines():
        name, *fields = line.split()
        if name.startswith("cpu") and name[3:].isdigit():
            ticks = [int(f) for f in fields[:8]]  # user nice system idle iowait irq softirq steal
            total = sum(ticks)
            cores[int(name[3:])] = (total - ticks[3] - ticks[4], total)
    return cores


class CpuMonitor:
    """Context manager measuring mean busy % of each core group over its body.

    `groups` maps a name to core indices, e.g. {"server": [0, 1, 2], "client": [3]}.
    """

    def __init__(self, groups: dict[str, list[int]]):
        self.groups = groups
        self.busy_pct: dict[str, float] = {}

    def __enter__(self) -> "CpuMonitor":
        self._start = _read()
        return self

    def __exit__(self, *exc) -> None:
        end = _read()
        for name, cores in self.groups.items():
            busy = sum(end[c][0] - self._start[c][0] for c in cores)
            total = sum(end[c][1] - self._start[c][1] for c in cores)
            self.busy_pct[name] = round(100 * busy / total, 1) if total else 0.0
