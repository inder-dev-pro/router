"""Matplotlib visualisations for the RouterBench evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ensure_mpl() -> None:
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib")


def _bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    fmt: str = ".4f",
    color: str = "#4C78A8",
    pct: bool = False,
) -> None:
    """Generic grouped bar chart helper."""
    _ensure_mpl()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    finite_vals = [v if not math.isnan(v) else 0.0 for v in values]
    bars = ax.bar(labels, finite_vals, color=color, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        if not math.isnan(val):
            label = f"{val:{fmt}}" if not pct else f"{val:.1%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {output_path}")


def plot_utility_by_config(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [s.get("configuration", s.get("mode", "")) for s in summaries]
    router_utils = [s.get("avg_router_utility", float("nan")) for s in summaries]
    oracle_utils = [s.get("avg_oracle_utility", float("nan")) for s in summaries]

    _ensure_mpl()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    x = range(len(labels))
    width = 0.35
    r_vals = [v if not math.isnan(v) else 0 for v in router_utils]
    o_vals = [v if not math.isnan(v) else 0 for v in oracle_utils]
    ax.bar([i - width / 2 for i in x], r_vals, width, label="Router", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], o_vals, width, label="Oracle", color="#F58518")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Average Utility")
    ax.set_title("Average Utility: Router vs Oracle", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "utility_by_config.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {path}")


def plot_regret_by_config(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [s.get("configuration", s.get("mode", "")) for s in summaries]
    regrets = [s.get("avg_regret", float("nan")) for s in summaries]
    _bar_chart(labels, regrets, "Average Regret by Configuration", "Average Regret",
               output_dir / "regret_by_config.png", color="#E45756")


def plot_cost_by_config(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [s.get("configuration", s.get("mode", "")) for s in summaries]
    costs = [s.get("avg_actual_cost", float("nan")) for s in summaries]
    _bar_chart(labels, costs, "Average Routed Cost by Configuration", "Avg Cost (USD)",
               output_dir / "cost_by_config.png", fmt=".6f", color="#72B7B2")


def plot_success_rate_by_config(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [s.get("configuration", s.get("mode", "")) for s in summaries]
    rates = [s.get("routed_task_success_rate", float("nan")) for s in summaries]
    _bar_chart(labels, rates, "Task Success Rate by Configuration", "Success Rate",
               output_dir / "success_rate_by_config.png", color="#54A24B", pct=True)


def plot_model_selection_frequency(
    model_stats: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    """Stacked/grouped bar chart of model selection frequency across configs."""
    _ensure_mpl()

    # Collect all models across configs
    all_models: set[str] = set()
    for stats in model_stats.values():
        for ms in stats:
            all_models.add(ms["model_name"])
    all_models_sorted = sorted(all_models)

    configs = list(model_stats.keys())
    fig, ax = plt.subplots(figsize=(max(10, len(all_models_sorted) * 1.5), 6))

    width = 0.8 / max(len(configs), 1)
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2"]

    for ci, (config_label, stats) in enumerate(model_stats.items()):
        freq_map = {ms["model_name"]: ms["selections"] for ms in stats}
        values = [freq_map.get(m, 0) for m in all_models_sorted]
        offsets = [i + ci * width for i in range(len(all_models_sorted))]
        ax.bar(offsets, values, width, label=config_label,
               color=colors[ci % len(colors)], edgecolor="white", linewidth=0.5)

    ax.set_xticks([i + width * (len(configs) - 1) / 2 for i in range(len(all_models_sorted))])
    ax.set_xticklabels([m.split("/")[-1] for m in all_models_sorted], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Selections")
    ax.set_title("Model Selection Frequency", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "model_selection_frequency.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {path}")


def generate_all_plots(
    summaries: list[dict[str, Any]],
    model_stats: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Generate all visualisations."""
    if not HAS_MPL:
        print("  ⚠ matplotlib not available — skipping plots")
        return

    print("\nGenerating plots...")
    plot_utility_by_config(summaries, output_dir)
    plot_regret_by_config(summaries, output_dir)
    plot_cost_by_config(summaries, output_dir)
    plot_success_rate_by_config(summaries, output_dir)
    plot_model_selection_frequency(model_stats, output_dir)
