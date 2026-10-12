"""Online serving: closed-loop load against vLLM's OpenAI-compatible server.

`concurrency` simulated users each send a streaming completion, wait for it to
finish, then send the next, until a fixed number of requests has been served.
Requests use the same raw prompts and forced output length as offline mode.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .gpu_monitor import GpuMonitor
from .prompts import load_prompts
from .results import RunStore, now, percentiles


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
    chunks = 0
    usage = None
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
                    t_last = time.perf_counter()
                    t_first = t_first or t_last
                    chunks += 1
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if t_first is None:
        return {"ok": False, "error": "no tokens received"}

    # Under load the server can pack several tokens into one chunk, so the
    # token count comes from the usage record, not from counting chunks.
    tokens = usage["completion_tokens"] if usage else chunks
    return {
        "ok": True, "error": "", "completion_tokens": tokens,
        "ttft_ms": round((t_first - t0) * 1000, 3),
        "itl_ms": round((t_last - t_first) * 1000 / (tokens - 1), 3) if tokens > 1 else None,
        "latency_ms": round((t_last - t0) * 1000, 3),
    }


async def _run_level(client: httpx.AsyncClient, url: str, payloads: list[dict], concurrency: int):
    """Serve every payload with `concurrency` users; return (results, wall seconds)."""
    pending = iter(enumerate(payloads))
    results = []

    async def user() -> None:
        for i, payload in pending:
            results.append({"request": i, **await _request(client, url, payload)})

    start = time.perf_counter()
    await asyncio.gather(*(user() for _ in range(concurrency)))
    return sorted(results, key=lambda r: r["request"]), time.perf_counter() - start


async def _run(cfg: dict, store: RunStore, base_url: str) -> None:
    prompts = load_prompts(cfg["prompts"])
    url = f"{base_url}/v1/completions"
    done = store.completed()

    def payloads(n: int) -> list[dict]:
        return [{
            "model": cfg["model"], "prompt": prompts[i % len(prompts)],
            "max_tokens": cfg["max_new_tokens"], "temperature": 0, "seed": cfg["seed"],
            "stream": True, "stream_options": {"include_usage": True},
            # vLLM extensions: force the full output length, and override the
            # repetition penalty the server adopts from the model repo.
            "ignore_eos": True, "repetition_penalty": 1.0,
        } for i in range(n)]

    limits = httpx.Limits(max_connections=max(cfg["concurrency"]))
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10), limits=limits) as client:
        for concurrency in sorted(cfg["concurrency"]):
            pending = [r for r in range(cfg["repeats"]) if (concurrency, r) not in done]
            if not pending:
                print(f"[{cfg['name']}] concurrency={concurrency}: already on disk, skipping")
                continue
            label = {
                "run": cfg["name"], "mode": "online", "backend": cfg["backend"], "model": cfg["model"],
                "dtype": cfg["dtype"], "input_tokens": cfg["prompts"]["input_tokens"],
                "max_new_tokens": cfg["max_new_tokens"], "concurrency": concurrency,
            }
            # One warm-up "batch" online is one full wave of concurrent users.
            await _run_level(client, url, payloads(cfg["warmup_batches"] * concurrency), concurrency)

            for repeat in pending:
                with GpuMonitor(cfg["gpu_index"], cfg["monitor_interval_s"]) as monitor:
                    results, wall_s = await _run_level(client, url, payloads(cfg["num_requests"]), concurrency)

                ok = [r for r in results if r["ok"]]
                if not ok:
                    raise RuntimeError(f"every request failed at concurrency={concurrency}: {results[0]['error']}")
                generated = sum(r["completion_tokens"] for r in ok)
                itl = [r["itl_ms"] for r in ok if r["itl_ms"] is not None]
                summary = {
                    **label, "repeat": repeat, "num_requests": len(results), "errors": len(results) - len(ok),
                    "generated_tokens": generated, "wall_s": round(wall_s, 4),
                    "throughput_tok_s": round(generated / wall_s, 2),
                    "requests_per_s": round(len(ok) / wall_s, 3),
                    **percentiles([r["ttft_ms"] for r in ok], "ttft"),
                    "itl_mean_ms": round(sum(itl) / len(itl), 3) if itl else None,
                    **percentiles([r["latency_ms"] for r in ok], "latency"),
                    "peak_mem_gb": round(monitor.peak_mem_gb, 3),
                    "mean_util_pct": round(monitor.mean_util_pct, 1),
                    "status": "ok", "finished_at": now(),
                }
                store.append("raw.csv", [
                    {**label, "repeat": repeat, "request": r["request"], "ok": r["ok"], "error": r["error"],
                     "completion_tokens": r.get("completion_tokens"), "ttft_ms": r.get("ttft_ms"),
                     "itl_ms": r.get("itl_ms"), "latency_ms": r.get("latency_ms")}
                    for r in results
                ])
                store.append("gpu_samples.csv", [
                    {"concurrency": concurrency, "repeat": repeat, "t_s": round(t, 3),
                     "mem_used_mb": round(mem, 1), "util_pct": util}
                    for t, mem, util in monitor.samples
                ])
                store.append("summary.csv", [summary])
                print(
                    f"[{cfg['name']}] concurrency={concurrency} repeat={repeat + 1}/{cfg['repeats']}  "
                    f"{summary['throughput_tok_s']:,.0f} tok/s  ttft p95 {summary['ttft_p95_ms']:,.0f} ms  "
                    f"latency p95 {summary['latency_p95_ms']:,.0f} ms  errors {summary['errors']}"
                )
            store.warn_if_noisy(concurrency, cfg["max_spread"])


def run(cfg: dict, store: RunStore) -> None:
    with _server(cfg, store.dir / "server.log") as base_url:
        asyncio.run(_run(cfg, store, base_url))
