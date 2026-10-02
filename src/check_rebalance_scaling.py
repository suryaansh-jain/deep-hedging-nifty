from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baselines import DeltaHedgePolicy, run_policy
from black_scholes import call_price_np
from config import Config, load_config, seed_everything
from simulators import GBMSimulator

MEAN_TO_STD_LIMIT = 0.1


@dataclass
class ScalingVerdict:
    monotone: bool
    scaled_std_spread: float
    max_abs_mean_to_std: float

    def passed(self, tolerance: float) -> bool:
        return self.monotone and self.scaled_std_spread <= tolerance and self.max_abs_mean_to_std <= MEAN_TO_STD_LIMIT


def rebalance_scaling_table(config: Config) -> pd.DataFrame:
    settings = config.validation.rebalance
    maturity = config.option.maturity
    rate = config.option.r
    simulator = GBMSimulator(sigma=settings.sigma, mu=rate, maturity=maturity, n_steps=settings.fine_steps)
    paths = simulator.generate(settings.n_paths, settings.seed)
    prices, volatility = paths.to_torch(device="cpu", dtype=torch.float64)
    premium = float(call_price_np(np.array(1.0), 1.0, maturity, rate, settings.sigma))
    rows = []
    for n_rebalances in settings.rebalance_counts:
        if settings.fine_steps % n_rebalances != 0:
            raise ValueError(f"rebalance count {n_rebalances} must divide fine_steps {settings.fine_steps}")
        stride = settings.fine_steps // n_rebalances
        outcome = run_policy(
            DeltaHedgePolicy(),
            prices[:, ::stride],
            paths.times[::stride],
            volatility[:, ::stride],
            premium,
            1.0,
            rate,
            0.0,
        )
        pnl = outcome.pnl.numpy()
        rows.append({"n": n_rebalances, "std_pnl": pnl.std(ddof=1), "mean_pnl": pnl.mean()})
    table = pd.DataFrame(rows)
    table["std_times_sqrt_n"] = table["std_pnl"] * np.sqrt(table["n"])
    table["ratio_to_n1_std"] = table["std_pnl"] / table.loc[table["n"] == 1, "std_pnl"].iloc[0]
    table["mean_over_std"] = table["mean_pnl"] / table["std_pnl"]
    return table[["n", "std_pnl", "std_times_sqrt_n", "mean_pnl", "ratio_to_n1_std", "mean_over_std"]]


def judge(table: pd.DataFrame) -> ScalingVerdict:
    stds = table["std_pnl"].to_numpy()
    scaled = table["std_times_sqrt_n"].to_numpy()
    median_scaled = float(np.median(scaled))
    return ScalingVerdict(
        monotone=bool(np.all(np.diff(stds) < 0)),
        scaled_std_spread=float(np.max(np.abs(scaled / median_scaled - 1.0))),
        max_abs_mean_to_std=float(np.max(np.abs(table["mean_over_std"].to_numpy()))),
    )


def plot_scaling(table: pd.DataFrame, path: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 4.5))
    n_values = table["n"].to_numpy(dtype=float)
    stds = table["std_pnl"].to_numpy()
    axis.loglog(n_values, stds, "o-", label="BS delta hedge, std of P&L")
    reference = stds[0] * np.sqrt(n_values[0] / n_values)
    axis.loglog(n_values, reference, "--", color="gray", label="1/sqrt(n) reference")
    axis.set_xlabel("rebalances over 21 trading days (n)")
    axis.set_ylabel("std of terminal P&L (S0 = 1)")
    axis.set_title("Hedging error scaling under GBM, c = 0")
    axis.legend()
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    seed_everything(config.seed)
    output_directory = config.paths.resolve("validation")
    table = rebalance_scaling_table(config)
    table.to_csv(output_directory / "rebalance_scaling.csv", index=False)
    plot_scaling(table, str(output_directory / "rebalance_scaling.png"))
    verdict = judge(table)
    tolerance = config.validation.rebalance.tolerance
    passed = verdict.passed(tolerance)
    with pd.option_context("display.float_format", "{:.6f}".format, "display.width", 120):
        print(table.to_string(index=False))
    print(f"monotone decrease: {verdict.monotone}")
    print(f"max deviation of std*sqrt(n) from its median: {verdict.scaled_std_spread:.3f} (limit {tolerance})")
    print(f"max |mean P&L| / std: {verdict.max_abs_mean_to_std:.4f} (limit {MEAN_TO_STD_LIMIT})")
    print("CHECKPOINT A:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
