"""Online serving: load against vLLM's OpenAI-compatible server.

Closed loop: `concurrency` simulated users each send a streaming completion,
wait for it to finish, then send the next, until a fixed number of requests has
been served. Open loop (`open_loop` in the config): requests arrive as a
Poisson process at a fixed rate whether or not earlier ones have finished.
Requests use the same raw prompts and forced output length as offline mode.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .cpu_monitor import CpuMonitor
from .gpu_monitor import GpuMonitor
from .prompts import prompt_source
from .results import RunStore, now, percentiles
from .server_metrics import ServerMetrics

OPEN_SUFFIX = "_open"  # the open-loop sweep's files: summary_open.csv and so on
CPU_BUSY_PCT = 90      # client cores busier than this may be limiting the load


def open_store(store: RunStore) -> RunStore:
    return store.sibling("request_rate", OPEN_SUFFIX)


def _server_cmd(cfg: dict) -> list[str]:
    server = cfg["server"]
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["model"], "--dtype", cfg["dtype"], "--seed", str(cfg["seed"]),
        "--host", server["host"], "--port", str(server["port"]),
    ]
    for key, value in cfg.get("vllm", {}).items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            cmd.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            cmd += [f"--{flag}", str(value)]
    cores = (cfg.get("cpu") or {}).get("server_cores")
    if cores:
        if shutil.which("taskset") is None:
            raise RuntimeError("cpu.server_cores needs the `taskset` command (util-linux)")
        cmd = ["taskset", "-c", ",".join(str(c) for c in cores), *cmd]
    return cmd


def _wait_healthy(base_url: str, proc: subprocess.Popen, timeout_s: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {proc.returncode}; see {log_path}")
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server not healthy after {timeout_s:.0f} s; see {log_path}")


@contextlib.contextmanager
def _server(cfg: dict, log_path: Path):
    """Yield the server's base URL, launching and stopping it if configured to."""
    server = cfg["server"]
    base_url = server.get("base_url") or f"http://{server['host']}:{server['port']}"
    if not server["launch"]:
        yield base_url
        return
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(_server_cmd(cfg), stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_healthy(base_url, proc, server["startup_timeout_s"], log_path)
            yield base_url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


async def _request(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    """Send one streaming completion and time it from the client side."""
    t0 = time.perf_counter()
    t_first = t_last = None
    max_gap = 0.0
    chunks = 0
    usage = None
    text = hashlib.sha1()
    try:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                if chunk.get("choices"):
                    t_now = time.perf_counter()
                    if t_last is not None:
                        max_gap = max(max_gap, t_now - t_last)
                    t_last = t_now
                    t_first = t_first or t_last
                    chunks += 1
                    text.update((chunk["choices"][0].get("text") or "").encode())
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if t_first is None:
        return {"ok": False, "error": "no tokens received"}

    # Under load the server can pack several tokens into one chunk, so the
    # token count comes from the usage record, not from counting chunks.
    tokens = usage["completion_tokens"] if usage else chunks
    return {
        "ok": True, "error": "", "completion_tokens": tokens,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "ttft_ms": round((t_first - t0) * 1000, 3),
        "itl_ms": round((t_last - t_first) * 1000 / (tokens - 1), 3) if tokens > 1 else None,
        # Longest wait between two streamed chunks: a stall the mean hides.
        "max_gap_ms": round(max_gap * 1000, 3),
        "latency_ms": round((t_last - t0) * 1000, 3),
        # Fingerprint of the generated text, to check that an optimization
        # leaves greedy outputs unchanged.
        "output_sha": text.hexdigest()[:12],
    }


async def _run_level(client: httpx.AsyncClient, url: str, payloads: list[tuple[int, dict]], concurrency: int):
    """Serve every (prompt id, payload) with `concurrency` users; return (results, wall seconds)."""
    pending = iter(enumerate(payloads))
    results = []

    async def user() -> None:
        for i, (prompt, payload) in pending:
            results.append({"request": i, "prompt": prompt, **await _request(client, url, payload)})

    start = time.perf_counter()
    await asyncio.gather(*(user() for _ in range(concurrency)))
    return sorted(results, key=lambda r: r["request"]), time.perf_counter() - start


async def _run_open(client: httpx.AsyncClient, url: str, payloads: list[tuple[int, dict]], rate: float,
                    rng: random.Random):
    """Send every (prompt id, payload) at Poisson arrivals of `rate` per second; return (results, wall seconds)."""
    results = []

    async def one(i: int, prompt: int, payload: dict, lag_s: float) -> None:
        results.append({"request": i, "prompt": prompt, "send_lag_ms": round(lag_s * 1000, 3),
                        **await _request(client, url, payload)})

    start = time.perf_counter()
    due = 0.0
    tasks = []
    for i, (prompt, payload) in enumerate(payloads):
        due += rng.expovariate(rate)
        delay = start + due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        # How late the request left: large values mean the client could not
        # keep up with the schedule and the offered rate was not delivered.
        tasks.append(asyncio.create_task(one(i, prompt, payload, time.perf_counter() - start - due)))
    await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["request"]), time.perf_counter() - start


async def _measure(cfg: dict, base_url: str, load) -> tuple[list[dict], float, dict, list[dict], list[dict]]:
    """Run one timed repeat under every monitor the config turns on.

    Returns (results, wall seconds, summary columns, GPU samples, server samples).
    """
    cpu = cfg.get("cpu")
    with contextlib.ExitStack() as stack:
        gpu = stack.enter_context(GpuMonitor(cfg["gpu_index"], cfg["monitor_interval_s"]))
        host = stack.enter_context(
            CpuMonitor({"server": cpu["server_cores"], "client": cpu["client_cores"]})) if cpu else None
        server = stack.enter_context(
            ServerMetrics(base_url, cfg["monitor_interval_s"])) if cfg.get("scrape_metrics") else None
        results, wall_s = await load()

    columns = {"peak_mem_gb": round(gpu.peak_mem_gb, 3), "mean_util_pct": round(gpu.mean_util_pct, 1)}
    if host:
        columns.update(cpu_server_pct=host.busy_pct["server"], cpu_client_pct=host.busy_pct["client"])
    if server:
        columns.update(server.summary())
    gpu_rows = [{"t_s": round(t, 3), "mem_used_mb": round(mem, 1), "util_pct": util} for t, mem, util in gpu.samples]
    server_rows = [{"t_s": round(t, 3), **gauges} for t, gauges in server.samples] if server else []
    return results, wall_s, columns, gpu_rows, server_rows


async def _sweep(cfg: dict, store: RunStore, base_url: str, mode: str, settings: list[int],
                 warm_up, load) -> None:
    """Warm up, time and record every repeat of every setting of one sweep.

    `warm_up(setting)` and `load(setting, repeat)` are coroutines sending the
    untimed and the timed requests.
    """
    key = store.key
    done = store.completed()
    for setting in sorted(settings):
        pending = [r for r in range(cfg["repeats"]) if (setting, r) not in done]
        if not pending:
            print(f"[{cfg['name']}] {key}={setting}: already on disk, skipping")
            continue
        label = {
            "run": cfg["name"], "mode": mode, "backend": cfg["backend"], "model": cfg["model"],
            "dtype": cfg["dtype"], "input_tokens": cfg["prompts"]["input_tokens"],
            "max_new_tokens": cfg["max_new_tokens"], key: setting,
        }
        await warm_up(setting)

        for repeat in pending:
            results, wall_s, monitored, gpu_rows, server_rows = await _measure(
                cfg, base_url, lambda: load(setting, repeat))

            ok = [r for r in results if r["ok"]]
            if not ok:
                raise RuntimeError(f"every request failed at {key}={setting}: {results[0]['error']}")
            generated = sum(r["completion_tokens"] for r in ok)
            itl = [r["itl_ms"] for r in ok if r["itl_ms"] is not None]
            lag = [r["send_lag_ms"] for r in results if "send_lag_ms" in r]
            summary = {
                **label, "repeat": repeat, "num_requests": len(results), "errors": len(results) - len(ok),
                "generated_tokens": generated, "wall_s": round(wall_s, 4),
                "throughput_tok_s": round(generated / wall_s, 2),
                "requests_per_s": round(len(ok) / wall_s, 3),
                **percentiles([r["ttft_ms"] for r in ok], "ttft"),
                "itl_mean_ms": round(sum(itl) / len(itl), 3) if itl else None,
                "itl_p95_ms": percentiles(itl, "itl")["itl_p95_ms"] if itl else None,
                "max_gap_p95_ms": percentiles([r["max_gap_ms"] for r in ok], "max_gap")["max_gap_p95_ms"],
                **percentiles([r["latency_ms"] for r in ok], "latency"),
                **({"send_lag_max_ms": max(lag)} if lag else {}),
                **monitored,
                "status": "ok", "finished_at": now(),
            }
            store.append("raw.csv", [
                {**label, "repeat": repeat, "request": r["request"], "prompt": r["prompt"],
                 "ok": r["ok"], "error": r["error"], "prompt_tokens": r.get("prompt_tokens"),
                 "completion_tokens": r.get("completion_tokens"), "ttft_ms": r.get("ttft_ms"),
                 "itl_ms": r.get("itl_ms"), "max_gap_ms": r.get("max_gap_ms"),
                 "latency_ms": r.get("latency_ms"), "output_sha": r.get("output_sha"),
                 **({"send_lag_ms": r["send_lag_ms"]} if lag else {})}
                for r in results
            ])
            store.append("gpu_samples.csv", [{key: setting, "repeat": repeat, **row} for row in gpu_rows])
            store.append("server_metrics.csv", [{key: setting, "repeat": repeat, **row} for row in server_rows])
            store.append("summary.csv", [summary])
            print(
                f"[{cfg['name']}] {key}={setting} repeat={repeat + 1}/{cfg['repeats']}  "
                f"{summary['throughput_tok_s']:,.0f} tok/s  ttft p95 {summary['ttft_p95_ms']:,.0f} ms  "
                f"latency p95 {summary['latency_p95_ms']:,.0f} ms  errors {summary['errors']}"
            )
            wrong = sum(r["prompt_tokens"] not in (None, label["input_tokens"]) for r in ok)
            if wrong:
                print(f"  ! {wrong} prompts were not {label['input_tokens']} tokens long on the server")
            if summary.get("cpu_client_pct", 0) > CPU_BUSY_PCT:
                print(f"  ! client cores were {summary['cpu_client_pct']:.0f}% busy; the client may be the bottleneck")
        store.warn_if_noisy(setting, cfg["max_spread"])


async def _run(cfg: dict, store: RunStore, base_url: str) -> None:
    take = prompt_source(cfg["prompts"], cfg["seed"])
    url = f"{base_url}/v1/completions"
    open_loop = cfg.get("open_loop")

    def payloads(n: int) -> list[tuple[int, dict]]:
        return [(prompt, {
            "model": cfg["model"], "prompt": text,
            "max_tokens": cfg["max_new_tokens"], "temperature": 0, "seed": cfg["seed"],
            "stream": True, "stream_options": {"include_usage": True},
            # vLLM extensions: force the full output length, and override the
            # repetition penalty the server adopts from the model repo.
            "ignore_eos": True, "repetition_penalty": 1.0,
        }) for prompt, text in take(n)]

    # Open-loop arrivals do not wait for a free connection, so no pool limit there.
    limits = httpx.Limits(max_connections=None if open_loop else max(cfg["concurrency"]))
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10), limits=limits) as client:
        # One warm-up "batch" online is one full wave of concurrent users.
        await _sweep(
            cfg, store, base_url, "online", cfg["concurrency"],
            warm_up=lambda c: _run_level(client, url, payloads(cfg["warmup_batches"] * c), c),
            load=lambda c, _repeat: _run_level(client, url, payloads(cfg["num_requests"]), c),
        )
        if open_loop:
            # The arrival times depend only on the seed, rate and repeat, so
            # every config faces the same schedule.
            def arrivals(rate: int, repeat) -> random.Random:
                return random.Random(f"{cfg['seed']}-{rate}-{repeat}")

            await _sweep(
                cfg, open_store(store), base_url, "online_open", open_loop["request_rates"],
                warm_up=lambda rate: _run_open(
                    client, url, payloads(cfg["warmup_batches"] * open_loop["warmup_requests"]),
                    rate, arrivals(rate, "warmup")),
                load=lambda rate, repeat: _run_open(
                    client, url, payloads(open_loop["num_requests"]), rate, arrivals(rate, repeat)),
            )


def run(cfg: dict, store: RunStore) -> None:
    with _server(cfg, store.dir / "server.log") as base_url:
        client_cores = (cfg.get("cpu") or {}).get("client_cores")
        if client_cores:
            os.sched_setaffinity(0, client_cores)  # after the server starts, so it does not inherit this
        asyncio.run(_run(cfg, store, base_url))
