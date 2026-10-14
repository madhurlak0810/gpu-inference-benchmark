"""Run an optimization experiment on a Kaggle T4 from the command line.

    python scripts/kaggle_job.py push e4 --smoke   # tiny end-to-end check first
    python scripts/kaggle_job.py push e4           # the real run
    python scripts/kaggle_job.py status e4
    python scripts/kaggle_job.py fetch e4          # copy its results into results/

`push` starts a private script kernel that clones this repo at the current
commit, builds the pinned environment and runs every config named e4_*.yaml.
The commit must be on GitHub first: the kernel measures what is pushed, so
results/<run>/env.json records a commit anyone can check out.

Needs the Kaggle CLI (`pip install kaggle`, then `kaggle auth login`).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "3.13"  # the version the baseline was measured on

# The script the kernel runs. Results are copied to /kaggle/working after every
# config, so a kernel that dies part-way still returns what it finished.
KERNEL = '''\
import shutil, subprocess, sys
from pathlib import Path

REPO, COMMIT, CONFIGS, SMOKE = {repo!r}, {commit!r}, {configs!r}, {smoke!r}
work, out = Path("/tmp/bench"), Path("/kaggle/working/results")

def sh(*cmd, check=True, **kw):
    print("$", *cmd, flush=True)
    return subprocess.run(cmd, check=check, **kw).returncode

sh("nvidia-smi")
sh("git", "clone", REPO, str(work))
sh("git", "checkout", COMMIT, cwd=work)
sh(sys.executable, "-m", "pip", "install", "-q", "uv")
sh("uv", "venv", "--python", {python!r}, "/tmp/venv")
sh("uv", "pip", "install", "-q", "--python", "/tmp/venv/bin/python", "-r", "requirements.lock.txt", cwd=work)

failed = []
for config in CONFIGS:
    cmd = ["/tmp/venv/bin/python", "-m", "bench.run", f"configs/{{config}}.yaml"] + (["--smoke"] if SMOKE else [])
    if sh(*cmd, check=False, cwd=work):
        failed.append(config)
    log = work / "results" / ("_smoke" if SMOKE else "") / config / "server.log"
    if log.exists():  # the tail says why a server died; the head lists its resolved arguments
        print(f"--- last lines of {{log}}", *log.read_text(errors="replace").splitlines()[-25:], sep="\\n", flush=True)
    shutil.copytree(work / "results", out, dirs_exist_ok=True,
                    ignore=lambda d, names: [n for n in names if Path(d) == work / "results" and not (
                        n == "_smoke" if SMOKE else n in CONFIGS)])

print("failed:", failed or "none")
sys.exit(1 if failed else 0)
'''


def run(*cmd: str, capture: bool = False) -> str:
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def username() -> str:
    for line in run("kaggle", "config", "view", capture=True).splitlines():
        if line.startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("not logged in to Kaggle; run `kaggle auth login`")


def slug(experiment: str, smoke: bool) -> str:
    return f"gpu-bench-{experiment}" + ("-smoke" if smoke else "")


def configs_of(experiment: str) -> list[str]:
    """Every config of the experiment: controls first, repeated controls last."""
    names = sorted(p.stem for p in (REPO_ROOT / "configs").glob(f"{experiment}_*.yaml"))
    if not names:
        raise SystemExit(f"no configs/{experiment}_*.yaml")
    return sorted(names, key=lambda n: 0 if n.endswith("_control") else 2 if n.endswith("_control_repeat") else 1)


def push(args) -> None:
    if run("git", "status", "--porcelain", capture=True):
        raise SystemExit("uncommitted changes: the kernel runs a commit, so commit and push first")
    commit = run("git", "rev-parse", "HEAD", capture=True)
    repo = run("git", "remote", "get-url", "origin", capture=True)
    if not run("git", "branch", "-r", "--contains", commit, capture=True):
        raise SystemExit(f"commit {commit[:8]} is not on the remote; push it first")

    configs = args.configs or configs_of(args.experiment)
    name = slug(args.experiment, args.smoke)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "job.py").write_text(KERNEL.format(
            repo=repo, commit=commit, configs=configs, smoke=args.smoke, python=PYTHON), encoding="utf-8")
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{username()}/{name}", "title": name, "code_file": "job.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=2), encoding="utf-8")
        print(f"{name}: commit {commit[:8]}, configs {', '.join(configs)}" + (" (smoke)" if args.smoke else ""))
        if args.dry_run:
            print((folder / "job.py").read_text(encoding="utf-8"))
            return
        run("kaggle", "kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4")


def status(args) -> None:
    run("kaggle", "kernels", "status", f"{username()}/{slug(args.experiment, args.smoke)}")


def fetch(args) -> None:
    kernel = f"{username()}/{slug(args.experiment, args.smoke)}"
    with tempfile.TemporaryDirectory() as tmp:
        run("kaggle", "kernels", "output", kernel, "-p", tmp)
        fetched = Path(tmp) / "results"
        if not fetched.is_dir():
            raise SystemExit(f"{kernel} has no results in its output; check `kaggle kernels logs {kernel}`")
        shutil.copytree(fetched, REPO_ROOT / "results", dirs_exist_ok=True)
        print("fetched:", ", ".join(sorted(str(p.relative_to(fetched)) for p in fetched.glob("**/summary*.csv"))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, func in (("push", push), ("status", status), ("fetch", fetch)):
        p = sub.add_parser(name)
        p.add_argument("experiment", help="config-name prefix, e.g. e4")
        p.add_argument("--smoke", action="store_true", help="the tiny --smoke version, as its own kernel")
        p.set_defaults(func=func)
        if name == "push":
            p.add_argument("--configs", nargs="+", help="run only these configs instead of the whole experiment")
            p.add_argument("--dry-run", action="store_true", help="print the kernel script without starting it")
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
