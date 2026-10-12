"""Regenerate every chart from results/: `python -m analysis.plots`.

Each chart is a sweep (batch size or concurrent users) on a log-2 x axis, one
panel per model and prompt setting, with the mean across repeats as the line
and ± 1 standard deviation as the bars.
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
X_LABEL = {"offline": "Batch size", "online": "Concurrent users"}

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
    return {name: (name, SLOTS[i]) for i, name in enumerate(names)}


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
    width = max(4.4 * len(panels) + 0.8, 6.0)
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


if __name__ == "__main__":
    main()
