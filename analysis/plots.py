"""Regenerate every chart from results/: `python -m analysis.plots`.

Each baseline chart is a sweep (batch size or concurrent users) on a log-2 x
axis, one panel per model and prompt setting, with the mean across repeats as
the line and ± 1 standard deviation as the bars. Each optimization experiment
gets bar charts instead: one bar per run, one panel per swept setting.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from bench.config import REPO_ROOT

from .summarize import SWEEP, aggregate, load

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, never cycled

# A backend keeps its color in every chart.
BACKENDS = {"hf": ("Hugging Face generate()", SLOTS[0]), "vllm": ("vLLM", SLOTS[1])}

CHARTS = [
    # mode, metric, series, title, y label, file, note
    ("offline", "throughput_tok_s", "backend", "Offline throughput by batch size",
     "Generated tokens per second", "throughput_vs_batch_size.png", ""),
    ("offline", "peak_mem_gb", "backend", "Peak GPU memory by batch size",
     "Peak GPU memory (GB)", "peak_memory_vs_batch_size.png",
     "vLLM reserves its share of the card up front, so its line is the reservation, not the need."),
    ("online", "latency_p95_ms", "run", "p95 end-to-end latency by concurrent users",
     "p95 latency (ms)", "p95_latency_vs_concurrency.png", ""),
    ("online", "ttft_p95_ms", "run", "p95 time to first token by concurrent users",
     "p95 time to first token (ms)", "ttft_vs_concurrency.png", ""),
]
X_LABEL = {"offline": "Batch size", "online": "Concurrent users", "online_open": "Requests per second"}

# Experiment bar charts: experiment, mode, metric, title, x label, file, note, runs (None = all).
EXPERIMENT_CHARTS = [
    ("e4", "offline", "throughput_tok_s", "E4: throughput with and without compilation",
     "Generated tokens per second", "e4_throughput.png",
     "vLLM's eager mode turns off torch.compile and kernel warm-up as well as CUDA graphs.", None),
    ("e4", "offline", "mean_util_pct", "E4: GPU utilization with and without compilation",
     "Mean GPU utilization (%)", "e4_gpu_utilization.png", "", None),
    ("e3", "online_open", "ttft_p50_ms", "E3: time to first token at a fixed request rate",
     "Median time to first token (ms)", "e3_ttft_open_loop.png",
     "Open loop: requests arrive at the stated rate whether or not earlier ones have finished.", None),
    ("e3", "online", "ttft_p50_ms", "E3: time to first token with a fixed number of users",
     "Median time to first token (ms)", "e3_ttft_closed_loop.png",
     "Closed loop: each user sends its next request as soon as the last one finishes, "
     "so a faster server is sent more requests.", None),
    ("e3", "online", "throughput_tok_s", "E3: throughput with a fixed number of users",
     "Generated tokens per second", "e3_throughput_closed_loop.png", "", None),
    ("e1", "online", "throughput_tok_s", "E1: throughput under scheduler limits",
     "Generated tokens per second", "e1_throughput_closed_loop.png",
     "Closed loop. The control ran first and was about 5% faster than every other run, "
     "including those whose limit never binds.", None),
    ("e1", "online", "latency_p95_ms", "E1: p95 end-to-end latency under scheduler limits",
     "p95 end-to-end latency (ms)", "e1_latency_closed_loop.png", "Closed loop.", None),
    ("e1", "online", "max_gap_p95_ms", "E1: longest wait between two tokens, by token budget",
     "p95 of each request's longest gap between tokens (ms)", "e1_token_gap_closed_loop.png",
     "Closed loop.", ("e1_tokens_1024", "e1_control", "e1_tokens_4096", "e1_tokens_8192")),
]
# Experiment sweep charts, at most 4 runs each: experiment, mode, metric, title, y label, file, note, runs.
EXPERIMENT_SWEEPS = [
    ("e1", "online_open", "latency_p95_ms", "E1: p95 end-to-end latency by request rate",
     "p95 latency (ms)", "e1_latency_vs_request_rate.png",
     "Open loop: requests arrive at the stated rate whether or not earlier ones have finished.",
     ("e1_control", "e1_seqs_016", "e1_seqs_032", "e1_seqs_064")),
]
WORKLOAD = {
    "e4": "Qwen2.5-0.5B, 128 tokens in and out",
    "e3": "Qwen2.5-0.5B, 1,024 tokens in (the first 512 shared by every prompt) and 128 out",
    "e1": "Qwen2.5-0.5B, 128 tokens in and out",
}
PANEL_LABEL = {"batch_size": "Batch size {}", "concurrency": "{} concurrent users",
               "request_rate": "{} requests per second"}
# Run -> bar label, in the order the bars are drawn.
RUN_LABELS = {
    "e4_hf_control": "Hugging Face generate()",
    "e4_hf_static": "+ static KV cache",
    "e4_hf_compiled": "+ static KV cache, torch.compile",
    "e4_vllm_control": "vLLM",
    "e4_vllm_eager": "vLLM, eager mode",
    "e3_control": "Prefix caching off",
    "e3_prefix_cache": "Prefix caching on",
    "e1_control": "Default limits",
    "e1_seqs_016": "max_num_seqs 16",
    "e1_seqs_032": "max_num_seqs 32",
    "e1_seqs_064": "max_num_seqs 64",
    "e1_seqs_128": "max_num_seqs 128",
    "e1_tokens_1024": "max_num_batched_tokens 1,024",
    "e1_tokens_4096": "max_num_batched_tokens 4,096",
    "e1_tokens_8192": "max_num_batched_tokens 8,192",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",  # ships with matplotlib, so charts match on every machine
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 9.5,
})


def _styles(agg, series: str) -> dict:
    """Series value -> (legend label, color)."""
    if series == "backend":
        return BACKENDS
    names = sorted(agg[series].unique())
    if len(names) > len(SLOTS):
        raise ValueError(f"{len(names)} runs on one chart; plot at most {len(SLOTS)} or split them")
    return {name: (RUN_LABELS.get(name, name), SLOTS[i]) for i, name in enumerate(names)}


def _compact(value: float, _pos=None) -> str:
    return f"{value / 1000:g}K" if value >= 1000 else f"{value:g}"


def _value(value: float) -> str:
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def _gpu_names(results_dir: Path) -> str:
    names = set()
    for path in results_dir.glob("*/env.json"):
        if not path.parent.name.startswith("_"):
            names.add(json.loads(path.read_text(encoding="utf-8"))["gpu"]["name"])
    return ", ".join(sorted(names))


def sweep_chart(agg, sweep: str, metric: str, series: str, title: str, y_label: str,
                x_label: str, subtitle: list[str], path: Path) -> None:
    styles = _styles(agg, series)
    panels = list(agg.groupby(["model", "input_tokens", "max_new_tokens"]))
    shown = [styles[v] for v in styles if v in set(agg[series])]
    has_legend = len(shown) > 1  # one series needs no legend; the title names it

    # The header (title, subtitle, legend) is sized in inches from its contents,
    # so a longer subtitle pushes the plot down instead of running into it.
    width = max(4.4 * len(panels) + 0.8, 6.0, 2.5 * len(shown))  # room for the legend row
    lines = [line for part in subtitle for line in textwrap.wrap(part, int(width * 11.5))]
    legend_top = 0.58 + 0.18 * len(lines)
    header = legend_top + (0.38 if has_legend else 0.08)
    height = 3.5 + header
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), sharey=True, squeeze=False)

    for ax, ((model, tokens_in, tokens_out), panel) in zip(axes[0], panels):
        ends = []
        for value, rows in panel.groupby(series):
            rows = rows.sort_values(sweep)
            _, color = styles[value]
            ax.errorbar(
                rows[sweep], rows[f"{metric}_mean"], yerr=rows[f"{metric}_std"].fillna(0),
                color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                elinewidth=1, capsize=0, zorder=3,
            )
            ends.append((rows[sweep].iloc[-1], rows[f"{metric}_mean"].iloc[-1]))

        ticks = sorted(panel[sweep].unique())
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(_compact))
        ax.set_ylim(bottom=0)
        ax.margins(x=0.12)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_xlabel(x_label)
        ax.set_title(f"{model.split('/')[-1]} · {tokens_in} tokens in, {tokens_out} out",
                     loc="left", fontsize=9.5, color=INK_SECONDARY, pad=8)

        # Label each line's last value, unless two labels would sit on each other.
        top = ax.get_ylim()[1]
        if all(abs(a[1] - b[1]) > 0.07 * top for a, b in combinations(ends, 2)):
            for x, y in ends:
                ax.annotate(_value(y), (x, y), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK_SECONDARY, annotation_clip=False)

    axes[0][0].set_ylabel(y_label)
    fig.tight_layout(rect=(0, 0, 1, 1 - header / height), w_pad=3)
    left = axes[0][0].get_position().x0
    fig.text(left, 1 - 0.2 / height, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(left, 1 - 0.54 / height, "\n".join(lines), fontsize=9, color=INK_SECONDARY,
             va="top", linespacing=1.4)
    if has_legend:
        handles = [Line2D([], [], color=color, linewidth=2, marker="o", markersize=8,
                          markeredgecolor=SURFACE, markeredgewidth=2) for _, color in shown]
        fig.legend(handles, [label for label, _ in shown], loc="upper left",
                   bbox_to_anchor=(left - 0.008, 1 - legend_top / height), ncol=len(shown), frameon=False,
                   fontsize=9, labelcolor=INK_SECONDARY, handlelength=2.2, columnspacing=1.8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def bar_chart(agg, sweep: str, metric: str, title: str, x_label: str, subtitle: list[str], path: Path,
              runs: tuple[str, ...] | None = None) -> None:
    """One horizontal bar per run, colored by backend, one panel per swept setting."""
    runs = [r for r in (runs or RUN_LABELS) if r in set(agg["run"])]
    settings = sorted(agg[sweep].unique())
    backend_of = dict(zip(agg["run"], agg["backend"]))
    shown = [BACKENDS[b] for b in BACKENDS if b in backend_of.values()]

    width = 4.6 * len(settings) + 2.4
    lines = [line for part in subtitle for line in textwrap.wrap(part, int(width * 11.5))]
    legend_top = 0.58 + 0.18 * len(lines)
    header = legend_top + (0.38 if len(shown) > 1 else 0.08)
    height = 0.42 * len(runs) + 1.0 + header
    fig, axes = plt.subplots(1, len(settings), figsize=(width, height), sharey=True, squeeze=False)

    for ax, setting in zip(axes[0], settings):
        rows = agg[agg[sweep] == setting].set_index("run").reindex(runs)
        means, stds = rows[f"{metric}_mean"], rows[f"{metric}_std"].fillna(0)
        ax.barh(range(len(runs)), means, height=0.62, xerr=stds, zorder=3,
                color=[BACKENDS[backend_of[r]][1] for r in runs],
                error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 0})
        for y, (mean, std) in enumerate(zip(means, stds)):
            ax.annotate(_value(mean), (mean + std, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_SECONDARY)
        ax.set_yticks(range(len(runs)), [RUN_LABELS[r] for r in runs], fontsize=9.5, color=INK_SECONDARY)
        ax.set_xlim(0, means.max() * 1.18)
        ax.xaxis.set_major_formatter(FuncFormatter(_compact))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.set_xlabel(x_label)
        ax.set_title(PANEL_LABEL[sweep].format(setting), loc="left", fontsize=9.5, color=INK_SECONDARY, pad=8)

    axes[0][0].invert_yaxis()  # shared axis: first run on top
    fig.tight_layout(rect=(0, 0, 1, 1 - header / height), w_pad=3)
    left = 0.02
    fig.text(left, 1 - 0.2 / height, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(left, 1 - 0.54 / height, "\n".join(lines), fontsize=9, color=INK_SECONDARY,
             va="top", linespacing=1.4)
    if len(shown) > 1:
        handles = [Patch(facecolor=color) for _, color in shown]
        fig.legend(handles, [label for label, _ in shown], loc="upper left",
                   bbox_to_anchor=(left - 0.008, 1 - legend_top / height), ncol=len(shown), frameon=False,
                   fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.2, columnspacing=1.8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate every chart from the committed CSVs.")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "figures"))
    args = ap.parse_args()

    results_dir, out = Path(args.results_dir), Path(args.out)
    df = load(results_dir)
    if df.empty:
        print(f"no results under {results_dir}; run an experiment first")
        return
    out.mkdir(parents=True, exist_ok=True)
    gpu = _gpu_names(results_dir)
    for mode, metric, series, title, y_label, filename, note in CHARTS:
        agg = aggregate(df, mode)
        if agg.empty or f"{metric}_mean" not in agg:
            print(f"skipped {filename}: no {mode} results")
            continue
        subtitle = [f"{gpu}. Mean across repeats; bars show ± 1 standard deviation."]
        if note:
            subtitle.append(note)
        sweep_chart(agg, SWEEP[mode], metric, series, title, y_label, X_LABEL[mode], subtitle, out / filename)
        print(f"wrote {out / filename}")

    experiments = load(results_dir, experiments=True)
    for kind, charts in (("bar", EXPERIMENT_CHARTS), ("sweep", EXPERIMENT_SWEEPS)):
        for experiment, mode, metric, title, label, filename, note, runs in charts:
            rows = experiments[experiments["run"].str.startswith(experiment + "_")] if not experiments.empty else experiments
            agg = aggregate(rows, mode)
            if agg.empty or f"{metric}_mean" not in agg:
                print(f"skipped {filename}: no {experiment} results")
                continue
            subtitle = [f"{gpu}, {WORKLOAD[experiment]}. Mean across repeats; bars show ± 1 standard deviation."]
            if note:
                subtitle.append(note)
            if kind == "bar":
                bar_chart(agg, SWEEP[mode], metric, title, label, subtitle, out / filename, runs)
            else:
                sweep_chart(agg[agg["run"].isin(runs)], SWEEP[mode], metric, "run", title, label,
                            X_LABEL[mode], subtitle, out / filename)
            print(f"wrote {out / filename}")


if __name__ == "__main__":
    main()
