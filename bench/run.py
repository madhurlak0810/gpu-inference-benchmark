"""Run one experiment: `python -m bench.run configs/<name>.yaml`.

Results land in results/<name>/. Re-running the same command resumes: settings
already on disk are skipped, so an interrupted session costs one repeat.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

from . import env_info
from .config import REPO_ROOT, load_config
from .results import RunStore

SWEEP_KEY = {"offline": "batch_size", "online": "concurrency"}


def _smoke(cfg: dict) -> dict:
    """Shrink a config to a quick end-to-end check of the same code path."""
    cfg = copy.deepcopy(cfg)
    cfg.update(repeats=1, warmup_batches=1, max_new_tokens=16)
    cfg["prompts"]["num_prompts"] = 8
    if cfg["mode"] == "offline":
        cfg["batch_sizes"] = [1, 4]
    else:
        cfg["concurrency"] = [1, 4]
        cfg["num_requests"] = 8
    return cfg


def _check_resumable(out_dir: Path, cfg: dict, env: dict) -> None:
    """Refuse to mix rows measured under a different config or environment."""
    config_path, env_path = out_dir / "config.yaml", out_dir / "env.json"
    if config_path.exists() and yaml.safe_load(config_path.read_text(encoding="utf-8")) != cfg:
        raise SystemExit(f"{config_path} differs from this config; pass --overwrite to start the run again")
    if env_path.exists():
        old = json.loads(env_path.read_text(encoding="utf-8"))
        if any(old.get(k) != env[k] for k in ("gpu", "cuda_runtime", "packages")):
            raise SystemExit(f"{env_path} was recorded on a different environment; pass --overwrite to start again")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one benchmark experiment.")
    ap.add_argument("config", help="path to an experiment YAML under configs/")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny version of the experiment, written to results/_smoke/")
    ap.add_argument("--overwrite", action="store_true", help="discard this run's existing results first")
    ap.add_argument("--redo-noisy", action="store_true",
                    help="rerun the settings whose spread across repeats exceeds max_spread")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(args.results_dir)
    if args.smoke:
        cfg = _smoke(cfg)
        results_dir = results_dir / "_smoke"

    # Kaggle hands out two T4s; keep every library on the one being measured.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_index"])

    store = RunStore(results_dir / cfg["name"], SWEEP_KEY[cfg["mode"]])
    env = env_info.collect(cfg["gpu_index"])
    if args.overwrite or args.smoke:
        store.clear()
    else:
        _check_resumable(store.dir, cfg, env)
    if args.redo_noisy:
        noisy = store.noisy(cfg["max_spread"])
        if not noisy:
            print(f"[{cfg['name']}] nothing is noisy")
            return  # before the model loads
        print(f"[{cfg['name']}] rerunning {store.key} in {noisy}")
        store.drop(noisy)

    (store.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (store.dir / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"[{cfg['name']}] {env['gpu']['name']}, {cfg['backend']} {cfg['mode']}, writing to {store.dir}")

    if cfg["mode"] == "offline":
        from . import offline as mode
    else:
        from . import online as mode
    mode.run(cfg, store)


if __name__ == "__main__":
    main()
