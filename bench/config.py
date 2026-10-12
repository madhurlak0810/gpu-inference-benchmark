"""Experiment configs: one YAML per experiment.

A config may `extends:` another file and then states only what it changes, so
the diff between two experiments is the diff between two short files.
"""
from __future__ import annotations

from pathlib import Path

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


def load_config(path: str | Path) -> dict:
    """Resolve `extends`, name the experiment after its file, and validate."""
    path = Path(path).resolve()
    cfg = _load(path)
    cfg["name"] = path.stem
    _validate(cfg)
    return cfg
