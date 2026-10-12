"""Offline batch throughput: the same fixed prompts through Hugging Face
generate() or vLLM's LLM.generate(), one batch at a time.

Both backends are handed raw prompt strings and timed from that call to the
finished batch, so tokenization is inside the timed region for both. Both
decode greedily with no repetition penalty and are forced to produce exactly
`max_new_tokens` per prompt, so every setting generates the same token count.
"""
from __future__ import annotations

import time

from .gpu_monitor import GpuMonitor
from .prompts import load_prompts
from .results import RunStore, now, percentiles


class HFBackend:
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.manual_seed(cfg["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]))
        self.model.to("cuda").eval()

        # The model repo's generation_config turns on sampling and a repetition
        # penalty. Reset it to plain greedy decoding, and drop the EOS token so
        # generation never stops early: the same thing vLLM's ignore_eos does.
        gen = self.model.generation_config
        gen.do_sample = False
        gen.repetition_penalty = 1.0
        gen.temperature = gen.top_p = gen.top_k = None
        gen.eos_token_id = None
        gen.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompts: list[str], max_new_tokens: int) -> tuple[int, int]:
        """Run one batch; return (prompt tokens, generated tokens)."""
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        generated = (out.shape[1] - enc["input_ids"].shape[1]) * len(prompts)
        return int(enc["attention_mask"].sum()), generated

    def new_setting(self) -> None:
        # Return cached blocks so NVML memory reflects this batch size, not the last.
        self.torch.cuda.empty_cache()

    def reset_peak(self) -> None:
        self.torch.cuda.reset_peak_memory_stats()

    def peak_alloc_gb(self) -> float | None:
        return round(self.torch.cuda.max_memory_allocated() / 2**30, 3)


class VLLMBackend:
    def __init__(self, cfg: dict):
        from vllm import LLM, SamplingParams

        self._sampling_params = SamplingParams
        self.llm = LLM(model=cfg["model"], dtype=cfg["dtype"], seed=cfg["seed"], **cfg.get("vllm", {}))

    def generate(self, prompts: list[str], max_new_tokens: int) -> tuple[int, int]:
        params = self._sampling_params(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return (
            sum(len(o.prompt_token_ids) for o in outputs),
            sum(len(o.outputs[0].token_ids) for o in outputs),
        )

    def new_setting(self) -> None:
        pass

    def reset_peak(self) -> None:
        pass

    def peak_alloc_gb(self) -> float | None:
        # vLLM reserves its KV-cache pool up front; there is no per-run allocator peak.
        return None


BACKENDS = {"hf": HFBackend, "vllm": VLLMBackend}


def _is_oom(exc: Exception) -> bool:
    return type(exc).__name__ == "OutOfMemoryError"


def run(cfg: dict, store: RunStore) -> None:
    prompts = load_prompts(cfg["prompts"])
    backend = BACKENDS[cfg["backend"]](cfg)
    max_new_tokens = cfg["max_new_tokens"]
    done = store.completed()

    for batch_size in sorted(cfg["batch_sizes"]):
        pending = [r for r in range(cfg["repeats"]) if (batch_size, r) not in done]
        if not pending:
            print(f"[{cfg['name']}] batch_size={batch_size}: already on disk, skipping")
            continue
        batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
        label = {
            "run": cfg["name"], "mode": "offline", "backend": cfg["backend"], "model": cfg["model"],
            "dtype": cfg["dtype"], "input_tokens": cfg["prompts"]["input_tokens"],
            "max_new_tokens": max_new_tokens, "batch_size": batch_size,
        }
        backend.new_setting()
        try:
            for i in range(cfg["warmup_batches"]):
                backend.generate(batches[i % len(batches)], max_new_tokens)

            for repeat in pending:
                raw = []
                backend.reset_peak()
                with GpuMonitor(cfg["gpu_index"], cfg["monitor_interval_s"]) as monitor:
                    start = time.perf_counter()
                    for i, batch in enumerate(batches):
                        t0 = time.perf_counter()
                        prompt_tokens, generated = backend.generate(batch, max_new_tokens)
                        latency_ms = (time.perf_counter() - t0) * 1000
                        raw.append({
                            **label, "repeat": repeat, "batch_idx": i, "num_prompts": len(batch),
                            "prompt_tokens": prompt_tokens, "generated_tokens": generated,
                            "latency_ms": round(latency_ms, 3),
                        })
                    wall_s = time.perf_counter() - start

                generated_total = sum(r["generated_tokens"] for r in raw)
                # Static batching returns a whole batch at once, so a request's
                # end-to-end latency is its batch's latency.
                latencies = [r["latency_ms"] for r in raw for _ in range(r["num_prompts"])]
                summary = {
                    **label, "repeat": repeat, "num_prompts": len(prompts),
                    "generated_tokens": generated_total, "wall_s": round(wall_s, 4),
                    "throughput_tok_s": round(generated_total / wall_s, 2),
                    **percentiles(latencies, "latency"),
                    "peak_mem_gb": round(monitor.peak_mem_gb, 3),
                    "mean_util_pct": round(monitor.mean_util_pct, 1),
                    "peak_torch_alloc_gb": backend.peak_alloc_gb(),
                    "status": "ok", "finished_at": now(),
                }
                store.append("raw.csv", raw)
                store.append("gpu_samples.csv", [
                    {"batch_size": batch_size, "repeat": repeat, "t_s": round(t, 3),
                     "mem_used_mb": round(mem, 1), "util_pct": util}
                    for t, mem, util in monitor.samples
                ])
                store.append("summary.csv", [summary])
                print(
                    f"[{cfg['name']}] batch_size={batch_size} repeat={repeat + 1}/{cfg['repeats']}  "
                    f"{summary['throughput_tok_s']:,.0f} tok/s  p95 {summary['latency_p95_ms']:,.0f} ms  "
                    f"peak {summary['peak_mem_gb']:.2f} GB"
                )
        except Exception as exc:
            if not _is_oom(exc):
                raise
            # Larger batches would fail the same way; record where it stopped.
            store.append("summary.csv", [{
                **label, "repeat": pending[0], "num_prompts": len(prompts), "generated_tokens": 0,
                "wall_s": None, "throughput_tok_s": None,
                "latency_p50_ms": None, "latency_p95_ms": None, "latency_p99_ms": None,
                "peak_mem_gb": None, "mean_util_pct": None, "peak_torch_alloc_gb": None,
                "status": "oom", "finished_at": now(),
            }])
            print(f"[{cfg['name']}] batch_size={batch_size}: out of memory, stopping the sweep here")
            break
        store.warn_if_noisy(batch_size, cfg["max_spread"])
