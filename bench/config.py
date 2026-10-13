"""Experiment configs: one YAML per experiment.

A config may `extends:` another file and then states only what it changes, so
the diff between two experiments is the diff between two short files.
"""
from __future__ import annotations

from pathlib import Path

import os

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

_COMMON = (
    "mode", "backend", "model", "dtype", "seed", "gpu_index", "prompts",
    "max_new_tokens", "warmup_batches", "repeats", "max_spread", "monitor_interval_s",
)
_BY_MODE = {
    "offline": ("batch_sizes",),
    "online": ("concurrency", "num_requests", "server"),
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _merge(_load((path.parent / parent).resolve()), raw)


def _validate(cfg: dict) -> None:
    mode = cfg.get("mode")
    if mode not in _BY_MODE:
        raise ValueError(f"mode must be one of {sorted(_BY_MODE)}, got {mode!r}")
    missing = [k for k in _COMMON + _BY_MODE[mode] if k not in cfg]
    if missing:
        raise ValueError(f"config {cfg['name']!r} is missing: {', '.join(missing)}")
    if cfg["backend"] not in ("hf", "vllm"):
        raise ValueError(f"backend must be 'hf' or 'vllm', got {cfg['backend']!r}")
    if mode == "online" and cfg["backend"] != "vllm":
        raise ValueError("online mode drives vLLM's OpenAI-compatible server; set backend: vllm")
    if cfg["backend"] == "vllm":
        needed = cfg["prompts"]["input_tokens"] + cfg["max_new_tokens"]
        limit = cfg.get("vllm", {}).get("max_model_len")
        if limit is not None and needed > limit:
            raise ValueError(f"prompt + output is {needed} tokens but vllm.max_model_len is {limit}")
    _validate_prompts(cfg)
    if mode == "online":
        _validate_online(cfg)


def _validate_prompts(cfg: dict) -> None:
    prefix = cfg["prompts"].get("shared_prefix_tokens")
    if prefix is None:
        return
    if cfg["mode"] != "online":
        raise ValueError("a shared-prefix prompt set is built per request; it needs mode: online")
    if not 0 < prefix < cfg["prompts"]["input_tokens"]:
        raise ValueError("prompts.shared_prefix_tokens must be shorter than prompts.input_tokens")


def _validate_online(cfg: dict) -> None:
    """The optional online settings: open-loop load, CPU pinning, metrics scraping."""
    open_loop = cfg.get("open_loop")
    if open_loop is not None:
        missing = [k for k in ("request_rates", "num_requests", "warmup_requests") if k not in open_loop]
        if missing:
            raise ValueError(f"open_loop is missing: {', '.join(missing)}")
        rates = open_loop["request_rates"]
        if not all(isinstance(r, int) and r > 0 for r in rates):
            raise ValueError(f"open_loop.request_rates must be whole requests per second, got {rates}")
    elif not cfg["concurrency"]:
        raise ValueError("online mode needs concurrency levels, open_loop, or both")
    cpu = cfg.get("cpu")
    if cpu is not None:
        server, client = set(cpu["server_cores"]), set(cpu["client_cores"])
        if not server or not client or server & client:
            raise ValueError("cpu.server_cores and cpu.client_cores must be non-empty and not overlap")
        if not cfg["server"]["launch"]:
            raise ValueError("cpu pinning needs server.launch: true; an external server cannot be pinned from here")
        if hasattr(os, "sched_getaffinity") and not (server | client) <= os.sched_getaffinity(0):
            raise ValueError(f"cpu pins cores {sorted(server | client)} but this process may use "
                             f"{sorted(os.sched_getaffinity(0))}")


def load_config(path: str | Path) -> dict:
    """Resolve `extends`, name the experiment after its file, and validate."""
    path = Path(path).resolve()
    cfg = _load(path)
    cfg["name"] = path.stem
    _validate(cfg)
    return cfg
