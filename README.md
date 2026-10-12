# GPU Inference Benchmark

How much do serving choices change LLM inference speed on a single GPU? This
repo measures latency, throughput and GPU memory for Hugging Face `generate()`
and vLLM on one NVIDIA T4, across batch sizes and 1 to 64 concurrent users.
Every reported number comes from a CSV under `results/` and one command
reproduces it.

> **Status:** the baseline measurements below are complete. The optimization
> experiment (quantization, prefix caching or batching limits) is not done yet.

## Findings

Measured on one Tesla T4 with vLLM 0.31.0 and transformers 5.17.0. Each figure
is the mean of 3 repeats; the largest spread (standard deviation ÷ mean) in any
setting was 2.3%.

- **vLLM was 2.8× to 5.1× faster than Hugging Face `generate()` at the same
  batch size** on Qwen2.5-0.5B with short prompts: 3,755 against 1,275
  tokens/s at batch 64, and 168 against 33 at batch 1. With 1,024-token
  prompts the gap was 3.1× to 4.5×.
- **The gap narrowed on the larger model:** 1.9× to 2.3× on Qwen2.5-1.5B
  (1,458 against 621 tokens/s at batch 64).
- **Batching was the bigger lever.** Hugging Face went from 33 to 1,275
  tokens/s between batch 1 and batch 64 on short prompts, a 38× gain. With
  long prompts it stopped paying after batch 8: 206 tokens/s at batch 8 and 267
  at batch 64, while peak memory rose from 2.9 GB to 12.1 GB.
- **Under concurrent load, vLLM's server traded latency for throughput.** From
  1 to 64 users, throughput rose 22× (165 to 3,629 tokens/s, within 4% of the
  offline batch-64 figure) while p95 end-to-end latency rose 2.9× (784 to
  2,291 ms). The wait for the first token grew far more than the gap between
  tokens: p95 time to first token went from 23 to 517 ms, inter-token latency
  from 5.9 to 15.0 ms.

One measured clue to the small-batch gap: at batch 1 on the 0.5B model, Hugging
Face kept the GPU 38% busy and vLLM kept it 99% busy (`mean_util_pct` in the
CSVs). Hugging Face was waiting on work outside the GPU between tokens, which
is also why its throughput scaled almost linearly with batch size until the
GPU filled up.

## Results

Both tables are printed by `python -m analysis.summarize`. Throughput is mean ±
standard deviation over 3 repeats.

### Offline: Hugging Face against vLLM

| Model | Tokens in / out | Batch | HF tok/s | vLLM tok/s | vLLM ÷ HF | HF p95 latency (ms) | vLLM p95 latency (ms) | HF peak memory (GB) | vLLM peak memory (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 128 / 128 | 1 | 33 ± 0 | 168 ± 4 | 5.1× | 3,906 | 773 | 1.6 | 13.5 |
| Qwen2.5-0.5B-Instruct | 128 / 128 | 8 | 258 ± 1 | 1,130 ± 2 | 4.4× | 3,993 | 910 | 1.6 | 13.5 |
| Qwen2.5-0.5B-Instruct | 128 / 128 | 32 | 958 ± 2 | 2,724 ± 1 | 2.8× | 4,293 | 1,508 | 1.9 | 13.6 |
| Qwen2.5-0.5B-Instruct | 128 / 128 | 64 | 1,275 ± 4 | 3,755 ± 6 | 2.9× | 6,427 | 2,181 | 2.1 | 13.9 |
| Qwen2.5-0.5B-Instruct | 1,024 / 256 | 1 | 32 ± 0 | 145 ± 0 | 4.5× | 8,059 | 1,783 | 1.7 | 13.5 |
| Qwen2.5-0.5B-Instruct | 1,024 / 256 | 8 | 206 ± 0 | 637 ± 0 | 3.1× | 9,951 | 3,220 | 2.9 | 13.9 |
| Qwen2.5-0.5B-Instruct | 1,024 / 256 | 32 | 256 ± 0 | 964 ± 1 | 3.8× | 31,979 | 8,507 | 6.8 | 14.0 |
| Qwen2.5-0.5B-Instruct | 1,024 / 256 | 64 | 267 ± 0 | 1,052 ± 3 | 3.9× | 61,370 | 15,575 | 12.1 | 14.0 |
| Qwen2.5-1.5B-Instruct | 128 / 128 | 1 | 28 ± 0 | 53 ± 0 | 1.9× | 4,589 | 2,531 | 3.7 | 13.2 |
| Qwen2.5-1.5B-Instruct | 128 / 128 | 8 | 213 ± 0 | 436 ± 1 | 2.0× | 4,873 | 2,354 | 3.8 | 13.3 |
| Qwen2.5-1.5B-Instruct | 128 / 128 | 32 | 507 ± 1 | 1,032 ± 13 | 2.0× | 8,088 | 3,987 | 4.1 | 13.8 |
| Qwen2.5-1.5B-Instruct | 128 / 128 | 64 | 621 ± 1 | 1,458 ± 14 | 2.3× | 13,202 | 5,620 | 4.6 | 14.0 |

Two cautions when reading it. Offline, a request's latency is the time of the
batch it was in, so the latency columns restate throughput and are not an
independent result. And the two memory columns measure different things: see
[Reading the memory numbers](#reading-the-memory-numbers).

### Online: vLLM's server under concurrent users

| Run | Users | Throughput (tok/s) | Time to first token p50 / p95 (ms) | Inter-token latency (ms) | End-to-end latency p50 / p95 (ms) | Peak GPU memory (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vllm_online | 1 | 165 ± 0 | 21 / 23 | 5.9 | 775 / 784 | 13.5 |
| vllm_online | 4 | 646 ± 1 | 39 / 43 | 5.9 | 791 / 795 | 13.5 |
| vllm_online | 16 | 1,802 ± 7 | 107 / 138 | 8.1 | 1,135 / 1,150 | 13.5 |
| vllm_online | 64 | 3,629 ± 1 | 335 / 517 | 15.0 | 2,225 / 2,291 | 13.6 |

All 1,536 timed requests succeeded.

## Charts

`python -m analysis.plots` regenerates every chart from the CSVs.

![Offline throughput by batch size](analysis/figures/throughput_vs_batch_size.png)

![Peak GPU memory by batch size](analysis/figures/peak_memory_vs_batch_size.png)

![p95 end-to-end latency by concurrent users](analysis/figures/p95_latency_vs_concurrency.png)

![p95 time to first token by concurrent users](analysis/figures/ttft_vs_concurrency.png)

## Methodology

### Setup

| Piece | Choice |
| --- | --- |
| GPU | One NVIDIA Tesla T4 (15 GB) on Kaggle, driver 580.178.04, CUDA runtime 13.0 |
| Software | Python 3.13.15, torch 2.13.0, transformers 5.17.0, vLLM 0.31.0 |
| Models | Qwen2.5-0.5B-Instruct (main), Qwen2.5-1.5B-Instruct (second size) |
| Precision | fp16 (the T4 has no bf16) |
| Serving stacks | Hugging Face `transformers` `generate()`; vLLM `LLM.generate()` and its OpenAI-compatible server |
| Prompts | 64 fixed synthetic prompts per length, committed under `data/` |

Each run's GPU, driver and package versions are in `results/<run>/env.json`,
next to the resolved config it ran with. `requirements.lock.txt` lists all 207
packages of the measured environment.

### Experiments

Each experiment is one YAML file under `configs/`. A config extends another and
states only what it changes, so two experiments differ by exactly what their
files say.

| Config | What it measures |
| --- | --- |
| `hf_offline_short` | Hugging Face, 128 tokens in and 128 out, batch sizes 1, 8, 32, 64 |
| `hf_offline_long` | The same with 1,024 tokens in and 256 out |
| `vllm_offline_short`, `vllm_offline_long` | The same two settings through vLLM |
| `hf_offline_short_1p5b`, `vllm_offline_short_1p5b` | The short setting on the 1.5B model |
| `vllm_online` | vLLM's server under 1, 4, 16 and 64 concurrent users, 128 requests per level |

### Metrics

| Metric | Definition |
| --- | --- |
| Throughput (tokens/s) | Generated tokens ÷ wall-clock time of the whole repeat |
| Time to first token (ms) | Request sent → first streamed chunk received (online) |
| Inter-token latency (ms) | (Last chunk − first chunk) ÷ (generated tokens − 1), averaged over requests (online) |
| End-to-end latency p50 / p95 / p99 (ms) | Online: request sent → last token received. Offline: the time of the batch the request was in |
| Peak GPU memory (GB) | Largest NVML reading for the whole device during the repeat |
| GPU utilization (%) | Mean of NVML readings taken every 0.5 s during the repeat |

### Rules for every run

- **Warm-up:** 3 untimed batches before each setting. Online, a batch is one
  wave of requests at that concurrency.
- **Repeats:** 3 per setting, reported as mean ± standard deviation. A setting
  whose throughput spread (std ÷ mean) exceeds 10% is flagged, and
  `--redo-noisy` reruns it. No setting needed a rerun.
- **Fixed inputs:** every prompt in a set is exactly 128 or exactly 1,024
  tokens, so batches need no padding and each request is the same work.
- **One change at a time:** each config differs from its parent in one respect.

### What keeps the comparison fair

- **Same output length.** Both backends ignore the end-of-sequence token and
  generate exactly the configured number of tokens for every prompt. The raw
  CSVs confirm identical prompt and output token counts on both.
- **Same decoding.** Greedy, with no repetition penalty. The model repository's
  default generation settings enable sampling and a repetition penalty; both
  backends override them.
- **No prefix caching in the baseline.** vLLM enables it by default, and
  warm-ups and repeats resend the same prompts, so a cache hit would skip work
  that Hugging Face has to do. It is off unless an experiment turns it on.
- **Same timed region.** Both backends are handed raw strings, so tokenization
  is inside the measurement for both.
- **One GPU.** Kaggle provides two T4s; every run is pinned to the first.

### Reading the memory numbers

vLLM reserves a fixed share of GPU memory up front (`gpu_memory_utilization`,
0.90 here) for its KV cache, so its 13 to 14 GB is the size of that reservation,
not what the workload needed. The Hugging Face column is what PyTorch held at
the peak. The CSVs also record PyTorch's own peak allocation for Hugging Face
(`peak_torch_alloc_gb`), which is lower still.

## Reproduce

```bash
pip install -r requirements.txt
python -m bench.run configs/hf_offline_short.yaml    # writes results/hf_offline_short/
python -m analysis.plots                             # writes analysis/figures/
```

On Kaggle, create a notebook with the **GPU T4 x2** accelerator and internet
access turned on, clone this repo, and run the same commands from its root with
a leading `!`. The numbers above came from a clean virtual environment built
from `requirements.txt` in a fresh Kaggle session; the seven experiments took
115 minutes in total.

Useful flags for `bench.run`:

| Flag | Effect |
| --- | --- |
| `--smoke` | A tiny version of the experiment, written to `results/_smoke/`. Use it first to check that a backend installs and runs |
| `--redo-noisy` | Rerun only the settings whose spread exceeded 10% |
| `--overwrite` | Discard the run's existing results and start again |

Running a config again without flags resumes it: settings already on disk are
skipped, so a session that times out loses at most the repeat in progress.

`python -m bench.prompts` is how the prompt sets under `data/` were built. It
does not need to be run again.

## Limitations and next steps

- One GPU type (T4, fp16 only) and two small models (0.5B and 1.5B). The
  results do not predict behaviour on larger models or data-centre GPUs.
- Everything was measured in a single Kaggle session on one physical card. The
  small spread describes repeats within that session, not variation between
  sessions or machines.
- Synthetic prompts of uniform length. Real traffic has mixed lengths, which
  changes how well batching works.
- The online load is closed-loop and every request is the same length, so the
  simulated users stay in step and their requests arrive in waves. Time to
  first token at 64 users is therefore close to a worst case of 64 simultaneous
  arrivals, not what a steady arrival rate would show.
- The load client runs on the same 4-core machine as the server.
- A single node with one GPU: no tensor parallelism or multi-GPU serving.
- Offline latency percentiles are over batches, and at batch size 64 a repeat
  is a single batch.

Next: one optimization experiment (AWQ 4-bit quantization, prefix caching, or
batching limits) measured against the `vllm_online` baseline.
