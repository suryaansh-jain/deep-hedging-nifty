from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from black_scholes import atm_call_premium
from config import Config, cost_rate_from_bps, load_config, load_fitted_params, resolve_sigma0, seed_everything
from evaluate import (
    DEEP_STRATEGIES,
    STRATEGY_LABELS,
    StrategyResult,
    load_band_widths,
    paired_bootstrap_cvar_difference,
    run_strategies,
    strategies_for,
    summarize,
)
from simulators import build_simulator

COMPARISON_BASELINES = ("bs_delta", "delta_band")
STRATEGY_COLORS = {"unhedged": "#9e9e9e", "bs_delta": "#1f77b4", "delta_band": "#2ca02c", "mlp": "#ff7f0e", "ntb": "#d62728", "ntb_tail": "#9467bd"}


def evaluate_simulator(config: Config, simulator_name: str, premium: float, band_widths: dict) -> tuple[list[dict], list[dict], dict]:
    fitted = load_fitted_params(config)
    simulator = build_simulator(simulator_name, config, fitted)
    test = simulator.generate(config.simulation.n_test_paths, config.simulation.test_seed)
    prices, volatility = test.to_torch(device="cpu")
    metric_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    saved: dict[str, np.ndarray] = {
        "times": test.times,
        "sample_prices": test.prices[: config.evaluation.sample_paths_saved],
        "sample_volatility": test.volatility[: config.evaluation.sample_paths_saved],
    }
    for cost_bps in config.costs.cost_grid:
        strategies = strategies_for(config, simulator_name, cost_bps, band_widths)
        results = run_strategies(strategies, prices, volatility, test.times, premium, config.option.r, cost_rate_from_bps(cost_bps))
        for name, result in results.items():
            metric_rows.append({"simulator": simulator_name, "cost_bps": cost_bps, "strategy": name, **summarize(result, premium)})
            saved[f"pnl__{name}__{cost_bps:g}"] = result.pnl.astype(np.float32)
            saved[f"positions__{name}__{cost_bps:g}"] = result.positions[: config.evaluation.sample_paths_saved]
        for deep_name in DEEP_STRATEGIES:
            if deep_name not in results:
                continue
            for baseline in COMPARISON_BASELINES:
                interval = paired_bootstrap_cvar_difference(
                    -results[deep_name].pnl,
                    -results[baseline].pnl,
                    config.risk.alpha,
                    config.evaluation.n_bootstrap,
                    config.evaluation.bootstrap_seed,
                )
                bootstrap_rows.append({"simulator": simulator_name, "cost_bps": cost_bps, "strategy": deep_name, "versus": baseline, **interval})
        print(f"{simulator_name} {cost_bps:>4} bps evaluated", flush=True)
    return metric_rows, bootstrap_rows, saved


def add_premium_units(metrics: pd.DataFrame, premium: float) -> pd.DataFrame:
    scaled = metrics.copy()
    for column in ["mean_pnl", "std_pnl", "cvar95", "cvar99", "mean_total_cost"]:
        scaled[f"{column}_x_premium"] = scaled[column] / premium
    return scaled


def plot_histograms(saved: dict, simulator_name: str, cost_bps: float, premium: float, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    names = [key.split("__")[1] for key in saved if key.startswith("pnl__") and key.endswith(f"__{cost_bps:g}")]
    bins = np.linspace(-4, 2, 121)
    for name in names:
        values = saved[f"pnl__{name}__{cost_bps:g}"] / premium
        axis.hist(np.clip(values, bins[0], bins[-1]), bins=bins, histtype="step", linewidth=1.6, label=STRATEGY_LABELS[name], color=STRATEGY_COLORS[name], density=True)
    axis.set_xlabel("terminal P&L / premium")
    axis.set_ylabel("density")
    axis.set_yscale("log")
    axis.set_title(f"P&L distribution, {simulator_name.upper()}, cost {cost_bps:g} bps (tails clipped to the axis range)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_frontier(metrics: pd.DataFrame, simulator_name: str, path: Path) -> None:
    subset = metrics[(metrics["simulator"] == simulator_name) & (metrics["strategy"] != "unhedged")]
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for name, group in subset.groupby("strategy"):
        group = group.sort_values("cost_bps")
        axis.plot(group["mean_total_cost_x_premium"], group["cvar95_x_premium"], "o-", label=STRATEGY_LABELS[name], color=STRATEGY_COLORS[name])
        for row in group.itertuples():
            axis.annotate(f"{row.cost_bps:g}", (row.mean_total_cost_x_premium, row.cvar95_x_premium), textcoords="offset points", xytext=(4, -10), fontsize=7, color=STRATEGY_COLORS[name])
    axis.set_xlabel("mean total transaction cost / premium")
    axis.set_ylabel("CVaR95 of loss / premium")
    axis.set_title(f"Cost versus tail risk, {simulator_name.upper()} (labels: cost in bps)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_cvar_versus_cost(metrics: pd.DataFrame, simulator_name: str, path: Path) -> None:
    subset = metrics[(metrics["simulator"] == simulator_name) & (metrics["strategy"] != "unhedged")]
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, group in subset.groupby("strategy"):
        group = group.sort_values("cost_bps")
        axis.plot(group["cost_bps"], group["cvar95_x_premium"], "o-", label=STRATEGY_LABELS[name], color=STRATEGY_COLORS[name])
    axis.set_xlabel("transaction cost (bps)")
    axis.set_ylabel("CVaR95 of loss / premium")
    axis.set_title(f"CVaR95 versus cost, {simulator_name.upper()}")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_sample_path(saved: dict, simulator_name: str, cost_bps: float, path_index: int, path: Path) -> None:
    times_days = np.arange(saved["sample_prices"].shape[1])
    figure, (price_axis, position_axis) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    price_axis.plot(times_days, saved["sample_prices"][path_index], color="black")
    price_axis.axhline(1.0, linestyle="--", color="gray", linewidth=0.8)
    price_axis.set_ylabel("S / K")
    price_axis.set_title(f"Sample test path {path_index}, {simulator_name.upper()}, cost {cost_bps:g} bps")
    for key in saved:
        if key.startswith("positions__") and key.endswith(f"__{cost_bps:g}"):
            name = key.split("__")[1]
            if name == "unhedged":
                continue
            position_axis.step(times_days[:-1], saved[key][path_index], where="post", label=STRATEGY_LABELS[name], color=STRATEGY_COLORS[name])
    position_axis.set_xlabel("trading day")
    position_axis.set_ylabel("hedge position h")
    position_axis.legend(fontsize=8)
    position_axis.grid(alpha=0.3)
    price_axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_training_curves(config: Config, path: Path) -> None:
    curves_directory = config.paths.resolve("metrics") / "training_curves"
    panels = [("mlp", "garch"), ("ntb", "garch"), ("ntb_tail", "garch"), ("ntb", "heston")]
    figure, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.costs.cost_grid)))
    for axis, (policy_name, simulator_name) in zip(axes, panels):
        for color, cost_bps in zip(colors, config.costs.cost_grid):
            curve_path = curves_directory / f"{policy_name}_{simulator_name}_{cost_bps:g}bps.csv"
            if not curve_path.exists():
                continue
            curve = pd.read_csv(curve_path)
            axis.plot(curve["epoch"], curve["val_cvar"], color=color, label=f"{cost_bps:g} bps")
        axis.set_title(f"{STRATEGY_LABELS[policy_name]}, {simulator_name.upper()}")
        axis.set_xlabel("epoch")
        axis.set_ylabel("validation objective (raw units)")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    seed_everything(config.seed)
    sigma0 = resolve_sigma0(config)
    premium = atm_call_premium(config.option.r, sigma0, config.option.maturity)
    band_widths = load_band_widths(config)
    metric_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    processed_directory = config.paths.resolve("processed")
    figures_directory = config.paths.resolve("figures")
    saved_by_simulator = {}
    for simulator_name in config.evaluation.simulators:
        rows, intervals, saved = evaluate_simulator(config, simulator_name, premium, band_widths)
        metric_rows.extend(rows)
        bootstrap_rows.extend(intervals)
        np.savez_compressed(processed_directory / f"evaluation_{simulator_name}.npz", **saved)
        saved_by_simulator[simulator_name] = saved
    metrics = add_premium_units(pd.DataFrame(metric_rows), premium)
    bootstrap = pd.DataFrame(bootstrap_rows)
    for column in ["difference", "ci_low", "ci_high"]:
        bootstrap[f"{column}_x_premium"] = bootstrap[column] / premium
    metrics_directory = config.paths.resolve("metrics")
    metrics.to_csv(metrics_directory / "test_metrics.csv", index=False)
    bootstrap.to_csv(metrics_directory / "bootstrap_cvar95_differences.csv", index=False)
    with open(metrics_directory / "test_metrics.json", "w") as handle:
        json.dump({"premium": premium, "sigma0": sigma0, "test_seed": config.simulation.test_seed, "n_test_paths": config.simulation.n_test_paths, "metrics": metrics.to_dict(orient="records"), "bootstrap": bootstrap.to_dict(orient="records")}, handle, indent=2)
    default_cost = config.costs.cost_bps
    for simulator_name, saved in saved_by_simulator.items():
        plot_histograms(saved, simulator_name, default_cost, premium, figures_directory / f"pnl_histogram_{simulator_name}_{default_cost:g}bps.png")
        plot_frontier(metrics, simulator_name, figures_directory / f"frontier_{simulator_name}.png")
        plot_cvar_versus_cost(metrics, simulator_name, figures_directory / f"cvar_vs_cost_{simulator_name}.png")
        plot_sample_path(saved, simulator_name, default_cost, 0, figures_directory / f"sample_path_positions_{simulator_name}_{default_cost:g}bps.png")
    plot_training_curves(config, figures_directory / "training_curves.png")
    view = metrics[["simulator", "cost_bps", "strategy", "mean_pnl_x_premium", "std_pnl_x_premium", "cvar95_x_premium", "cvar99_x_premium", "mean_total_cost_x_premium", "mean_turnover", "fraction_loss_above_5pct_premium"]]
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.4f}".format):
        print(view.to_string(index=False))
        print(bootstrap[["simulator", "cost_bps", "strategy", "versus", "difference_x_premium", "ci_low_x_premium", "ci_high_x_premium"]].to_string(index=False))


if __name__ == "__main__":
    main()
