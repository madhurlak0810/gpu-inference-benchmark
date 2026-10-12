# GPU Inference Benchmark

How much do serving choices change LLM inference speed on a single GPU? This
repo measures latency, throughput and GPU memory for Hugging Face `generate()`
and vLLM on one NVIDIA T4, across batch sizes and 1 to 64 concurrent users.
Every reported number comes from a committed CSV and one command reproduces it.

> **Status:** the harness is written, but no T4 measurements have been run yet.
> The findings, results table and charts below are empty until they have.

## Findings

No measurements yet.

## Results

No measurements yet. `python -m analysis.summarize` prints this table
(backend × setting, with throughput, p95 latency and peak GPU memory) from the
CSVs under `results/`.

## Charts

No measurements yet. `python -m analysis.plots` regenerates every chart into
`analysis/figures/`:

| File | Shows |
|---|---|
| `throughput_vs_batch_size.png` | Offline throughput by batch size, both backends |
| `peak_memory_vs_batch_size.png` | Peak GPU memory by batch size, both backends |
| `p95_latency_vs_concurrency.png` | p95 end-to-end latency by concurrent users |
| `ttft_vs_concurrency.png` | p95 time to first token by concurrent users |

## Methodology

### Setup

| Piece | Choice |
|---|---|
| GPU | One NVIDIA T4 (16 GB) on Kaggle |
| Models | Qwen2.5-0.5B-Instruct (main), Qwen2.5-1.5B-Instruct (second size) |
| Precision | fp16 (the T4 has no bf16) |
| Serving stacks | Hugging Face `transformers` `generate()`; vLLM `LLM.generate()` and its OpenAI-compatible server |
| Prompts | 64 fixed synthetic prompts per length, committed under `data/` |

The exact GPU, driver, CUDA runtime and package versions of each run are in
`results/<run>/env.json`, next to the resolved config it ran with.

### Experiments

Each experiment is one YAML file under `configs/`. A config extends another and
states only what it changes, so two experiments differ by exactly what their
files say.

| Config | What it measures |
|---|---|
| `hf_offline_short` | Hugging Face, 128 tokens in and 128 out, batch sizes 1, 8, 32, 64 |
| `hf_offline_long` | The same with 1,024 tokens in and 256 out |
| `vllm_offline_short`, `vllm_offline_long` | The same two settings through vLLM |
| `hf_offline_short_1p5b`, `vllm_offline_short_1p5b` | The short setting on the 1.5B model |
| `vllm_online` | vLLM's server under 1, 4, 16 and 64 concurrent users, 128 requests per level |

### Metrics

| Metric | Definition |
|---|---|
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
  `--redo-noisy` reruns it.
- **Fixed inputs:** every prompt in a set is exactly 128 or exactly 1,024
  tokens, so batches need no padding and each request is the same work.
- **One change at a time:** each config differs from its parent in one respect.

### What keeps the comparison fair

- **Same output length.** Both backends ignore the end-of-sequence token and
  generate exactly the configured number of tokens for every prompt.
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
0.90 here) for its KV cache, so its peak memory is the size of that reservation
and does not vary with batch size. For Hugging Face the CSVs also record
PyTorch's own peak allocation (`peak_torch_alloc_gb`), which is the memory the
run needed rather than the memory PyTorch held.

## Reproduce

```bash
pip install -r requirements.txt
python -m bench.run configs/hf_offline_short.yaml    # writes results/hf_offline_short/
python -m analysis.plots                             # writes analysis/figures/
```

On Kaggle, create a notebook with the **GPU T4 x2** accelerator and internet
access turned on, clone this repo, and run the same commands from its root with
a leading `!`.

Useful flags for `bench.run`:

| Flag | Effect |
|---|---|
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
- Synthetic prompts of uniform length. Real traffic has mixed lengths, which
  changes how well batching works.
- A single node with one GPU: no tensor parallelism or multi-GPU serving.
- The online load is closed-loop (each simulated user waits for a response
  before sending the next request) and the client runs on the same machine as
  the server, so the results do not show behaviour under a fixed arrival rate.
- Offline latency percentiles are over batches, and at batch size 64 a repeat
  is a single batch.

Next: one optimization experiment (AWQ 4-bit quantization, prefix caching, or
batching limits) measured against the `vllm_online` baseline.
