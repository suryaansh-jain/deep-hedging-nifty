from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config, cost_rate_from_bps, load_config, load_fitted_params, seed_everything
from data_nifty import load_india_vix, load_nifty
from evaluate import DEEP_STRATEGIES, STRATEGY_LABELS, load_band_widths, paired_bootstrap_cvar_difference, strategies_for
from losses import exact_cvar_np
from real_data_test import RealWindows, build_windows, run_real_windows
from simulators import GarchParams

COMPARISON_BASELINES = ("bs_delta", "delta_band")
STRATEGY_COLORS = {"unhedged": "#9e9e9e", "bs_delta": "#1f77b4", "delta_band": "#2ca02c", "mlp": "#ff7f0e", "ntb": "#d62728"}


def summarize_windows(outputs: dict[str, np.ndarray]) -> dict[str, float]:
    pnl = outputs["pnl_x_premium"]
    losses = -pnl
    return {
        "mean_pnl_x_premium": float(pnl.mean()),
        "std_pnl_x_premium": float(pnl.std(ddof=1)),
        "cvar95_x_premium": exact_cvar_np(losses, 0.95),
        "cvar99_x_premium": exact_cvar_np(losses, 0.99),
        "worst_loss_x_premium": float(losses.max()),
        "mean_total_cost_x_premium": float(outputs["cost_x_premium"].mean()),
        "mean_turnover": float(outputs["turnover"].mean()),
        "fraction_loss_above_5pct_premium": float(np.mean(losses > 0.05)),
    }


def evaluate_windows(config: Config, windows: RealWindows, label: str, band_widths: dict, saved: dict) -> tuple[list[dict], list[dict]]:
    metric_rows, bootstrap_rows = [], []
    simulator_name = config.real_data.simulator_models
    for cost_bps in config.costs.cost_grid:
        strategies = strategies_for(config, simulator_name, cost_bps, band_widths)
        outputs = run_real_windows(strategies, windows, config.option.r, cost_rate_from_bps(cost_bps))
        for name, values in outputs.items():
            metric_rows.append({"windows": label, "n_windows": len(windows.premiums), "cost_bps": cost_bps, "strategy": name, **summarize_windows(values)})
            saved[f"{label}__pnl__{name}__{cost_bps:g}"] = values["pnl_x_premium"]
            saved[f"{label}__positions__{name}__{cost_bps:g}"] = values["positions"]
        for deep_name in DEEP_STRATEGIES:
            if deep_name not in outputs:
                continue
            for baseline in COMPARISON_BASELINES:
                interval = paired_bootstrap_cvar_difference(
                    -outputs[deep_name]["pnl_x_premium"],
                    -outputs[baseline]["pnl_x_premium"],
                    config.risk.alpha,
                    config.real_data.n_bootstrap,
                    config.real_data.bootstrap_seed,
                )
                bootstrap_rows.append({"windows": label, "n_windows": len(windows.premiums), "cost_bps": cost_bps, "strategy": deep_name, "versus": baseline, **interval})
    return metric_rows, bootstrap_rows


def plot_real_histogram(saved: dict, cost_bps: float, n_windows: int, path: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-6, 2, 41)
    for name in ["unhedged", "bs_delta", "delta_band", "mlp", "ntb"]:
        key = f"non_overlapping__pnl__{name}__{cost_bps:g}"
        if key in saved:
            axis.hist(np.clip(saved[key], bins[0], bins[-1]), bins=bins, histtype="step", linewidth=1.6, color=STRATEGY_COLORS[name], label=STRATEGY_LABELS[name])
    axis.set_xlabel("terminal P&L / premium")
    axis.set_ylabel("number of windows")
    axis.set_title(f"Real Nifty, non-overlapping 21-day windows from 2020 (n = {n_windows}), cost {cost_bps:g} bps")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    seed_everything(config.seed)
    params = GarchParams(**load_fitted_params(config)["garch"]["params"])
    prices, vix = load_nifty(config), load_india_vix(config)
    band_widths = load_band_widths(config)
    saved: dict[str, np.ndarray] = {}
    metric_rows, bootstrap_rows, window_info = [], [], {}
    for label, overlapping in [("non_overlapping", False), ("overlapping", True)]:
        windows = build_windows(prices, vix, params, config.data.test_start_date, config.option.maturity_days, config.option.trading_days_per_year, config.option.r, overlapping)
        saved[f"{label}__start_dates"] = np.array([date.strftime("%Y-%m-%d") for date in windows.start_dates])
        saved[f"{label}__prices"] = windows.prices
        saved[f"{label}__volatility"] = windows.volatility
        saved[f"{label}__implied_vol"] = windows.sigma0
        window_info[label] = {
            "n_windows": len(windows.premiums),
            "first_start": str(windows.start_dates[0].date()),
            "last_start": str(windows.start_dates[-1].date()),
            "independent": not overlapping,
        }
        rows, intervals = evaluate_windows(config, windows, label, band_widths, saved)
        metric_rows.extend(rows)
        bootstrap_rows.extend(intervals)
    metrics, bootstrap = pd.DataFrame(metric_rows), pd.DataFrame(bootstrap_rows)
    metrics_directory = config.paths.resolve("metrics")
    metrics.to_csv(metrics_directory / "real_data_metrics.csv", index=False)
    bootstrap.to_csv(metrics_directory / "real_data_bootstrap_cvar95_differences.csv", index=False)
    with open(metrics_directory / "real_data_metrics.json", "w") as handle:
        json.dump({"windows": window_info, "models_trained_on": config.real_data.simulator_models, "metrics": metrics.to_dict(orient="records"), "bootstrap": bootstrap.to_dict(orient="records")}, handle, indent=2)
    np.savez_compressed(config.paths.resolve("processed") / "real_data.npz", **saved)
    plot_real_histogram(saved, config.costs.cost_bps, window_info["non_overlapping"]["n_windows"], str(config.paths.resolve("figures") / f"real_data_pnl_histogram_{config.costs.cost_bps:g}bps.png"))
    print(json.dumps(window_info, indent=2))
    with pd.option_context("display.width", 220, "display.max_rows", 300, "display.float_format", "{:.3f}".format):
        print(metrics[metrics["windows"] == "non_overlapping"].drop(columns=["windows"]).to_string(index=False))
        print(bootstrap[["windows", "cost_bps", "strategy", "versus", "difference", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
