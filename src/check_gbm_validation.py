from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baselines import DeltaHedgePolicy, run_policy
from black_scholes import atm_call_premium, call_delta_np
from config import Config, load_config, seed_everything, select_device
from losses import exact_cvar_np
from policies import MLPPolicy
from simulators import GBMSimulator
from train import run_name, save_training_result, settings_from_config, train_policy

PLOT_STEPS = (1, 5, 10, 15, 20)
MONEYNESS_EDGES = (-np.inf, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, np.inf)
EXPIRY_BUCKETS = ((21, 15), (14, 8), (7, 4), (3, 2), (1, 1))


@dataclass
class GbmValidationOutcome:
    mad: float
    passed: bool
    report: str


def hedge_ratios(policy: torch.nn.Module, prices: torch.Tensor, volatility: torch.Tensor, times: np.ndarray, premium: float, rate: float) -> tuple[np.ndarray, np.ndarray]:
    policy.eval()
    outcome = run_policy(policy, prices, times, volatility, premium, 1.0, rate, 0.0)
    return outcome.positions.cpu().numpy(), outcome.pnl.cpu().numpy()


def delta_matrix(prices: np.ndarray, times: np.ndarray, rate: float, sigma: float) -> np.ndarray:
    time_to_maturity = times[-1] - times[:-1]
    return call_delta_np(prices[:, :-1], 1.0, time_to_maturity[None, :], rate, sigma)


def bucket_table(signed_difference: np.ndarray, prices: np.ndarray, n_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute_difference = np.abs(signed_difference)
    days_to_expiry = n_steps - np.arange(n_steps)
    expiry_rows = []
    for upper, lower in EXPIRY_BUCKETS:
        mask = (days_to_expiry <= upper) & (days_to_expiry >= lower)
        expiry_rows.append(
            {
                "days_to_expiry": f"{lower}-{upper}" if lower != upper else f"{lower}",
                "count": absolute_difference[:, mask].size,
                "mad": absolute_difference[:, mask].mean(),
                "mean_learned_minus_delta": signed_difference[:, mask].mean(),
            }
        )
    log_moneyness = np.log(prices[:, :-1])
    moneyness_rows = []
    for lower, upper in zip(MONEYNESS_EDGES[:-1], MONEYNESS_EDGES[1:]):
        mask = (log_moneyness >= lower) & (log_moneyness < upper)
        moneyness_rows.append(
            {
                "ln_S_over_K": f"[{lower:g}, {upper:g})",
                "count": int(mask.sum()),
                "mad": absolute_difference[mask].mean() if mask.any() else np.nan,
                "mean_learned_minus_delta": signed_difference[mask].mean() if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(expiry_rows), pd.DataFrame(moneyness_rows)


def plot_hedge_vs_delta(prices: np.ndarray, learned: np.ndarray, times: np.ndarray, rate: float, sigma: float, path: str) -> None:
    figure, axes = plt.subplots(1, len(PLOT_STEPS), figsize=(4 * len(PLOT_STEPS), 4), sharey=True)
    rng = np.random.default_rng(0)
    sample = rng.choice(prices.shape[0], size=min(3000, prices.shape[0]), replace=False)
    for axis, step in zip(axes, PLOT_STEPS):
        spots = prices[sample, step]
        axis.scatter(spots, learned[sample, step], s=3, alpha=0.3, label="learned MLP hedge")
        grid = np.linspace(spots.min(), spots.max(), 200)
        axis.plot(grid, call_delta_np(grid, 1.0, times[-1] - times[step], rate, sigma), color="black", linewidth=1.5, label="BS delta")
        axis.set_title(f"step {step} ({len(times) - 1 - step} days to expiry)")
        axis.set_xlabel("S / K")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("hedge ratio")
    axes[0].legend(loc="upper left")
    figure.suptitle("GBM, c = 0: learned hedge ratio versus Black-Scholes delta on test paths")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def format_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_gbm_validation(config: Config) -> GbmValidationOutcome:
    seed_everything(config.seed)
    device = select_device(config.device)
    settings_gbm = config.validation.gbm
    rate, sigma = config.option.r, config.gbm.sigma
    simulator = GBMSimulator(sigma, config.gbm.mu, config.option.maturity, config.option.maturity_days)
    premium = atm_call_premium(rate, sigma, config.option.maturity)
    validation = simulator.generate(config.simulation.n_val_paths, config.simulation.val_seed)
    policy = MLPPolicy(sigma, config.policy.hidden_size, config.policy.hidden_layers)
    settings = settings_from_config(config, premium, 0.0)
    result = train_policy(policy, simulator, validation, settings, device, log_prefix="[gbm mlp] ")
    name = run_name("mlp", "gbm", 0.0)
    metadata = {"policy": "mlp", "simulator": "gbm", "cost_bps": 0.0, "sigma0": sigma, "premium": premium, "config": config.source_path}
    save_training_result(result, name, config.paths.resolve("models"), config.paths.resolve("metrics") / "training_curves", metadata)

    test = simulator.generate(settings_gbm.n_test_paths, settings_gbm.test_seed)
    policy_cpu = result.policy.to("cpu")
    prices, volatility = test.to_torch(device="cpu")
    learned, learned_pnl = hedge_ratios(policy_cpu, prices, volatility, test.times, premium, rate)
    deltas = delta_matrix(test.prices, test.times, rate, sigma)
    prices_double, volatility_double = test.to_torch(device="cpu", dtype=torch.float64)
    delta_outcome = run_policy(DeltaHedgePolicy(), prices_double, test.times, volatility_double, premium, 1.0, rate, 0.0)
    delta_pnl = delta_outcome.pnl.numpy()
    signed_difference = learned - deltas
    absolute_difference = np.abs(signed_difference)
    mad = float(absolute_difference.mean())
    passed = mad < settings_gbm.target_mad
    expiry_table, moneyness_table = bucket_table(signed_difference, test.prices, test.prices.shape[1] - 1)
    alpha = config.risk.alpha
    pnl_table = pd.DataFrame(
        [
            {"strategy": "MLP policy", "mean_pnl": learned_pnl.mean(), "std_pnl": learned_pnl.std(ddof=1), "cvar95_loss": exact_cvar_np(-learned_pnl, alpha)},
            {"strategy": "BS delta", "mean_pnl": delta_pnl.mean(), "std_pnl": delta_pnl.std(ddof=1), "cvar95_loss": exact_cvar_np(-delta_pnl, alpha)},
        ]
    )
    for column in ["mean_pnl", "std_pnl", "cvar95_loss"]:
        pnl_table[f"{column}_x_premium"] = pnl_table[column] / premium
    validation_directory = config.paths.resolve("validation")
    plot_hedge_vs_delta(test.prices, learned, test.times, rate, sigma, str(validation_directory / "gbm_hedge_vs_delta.png"))
    expiry_table.to_csv(validation_directory / "gbm_mad_by_expiry.csv", index=False)
    moneyness_table.to_csv(validation_directory / "gbm_mad_by_moneyness.csv", index=False)
    pnl_table.to_csv(validation_directory / "gbm_pnl_comparison.csv", index=False)
    report = "\n".join(
        [
            "# Checkpoint B: GBM validation of the MLP policy",
            "",
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `src/check_gbm_validation.py` from `{config.source_path}`.",
            "",
            "## Setup",
            "",
            f"- Simulator: GBM, sigma = {sigma}, drift mu = {config.gbm.mu}, r = {rate}, 21 daily steps, S0 = K = 1.",
            f"- Short one ATM call, premium C0 = {premium:.6f} (BS price at sigma = {sigma}). Transaction cost c = 0.",
            f"- MLPPolicy trained with CVaR{int(alpha * 100)} (Rockafellar-Uryasev), {config.simulation.n_train_paths} fresh paths per epoch, batch {config.training.batch_size}, "
            f"Adam lr {config.training.learning_rate}, early stopping on {config.simulation.n_val_paths} fixed validation paths (seed {config.simulation.val_seed}).",
            f"- Best epoch {result.best_epoch} of {len(result.history)} run, training time {result.seconds:.0f} s on {device}.",
            f"- Test set: {settings_gbm.n_test_paths} fresh GBM paths (seed {settings_gbm.test_seed}), never used in training or model selection.",
            "",
            "## Hedge ratio versus Black-Scholes delta",
            "",
            f"Mean absolute difference over all {absolute_difference.size} (path, step) pairs: **{mad:.4f}** (target below {settings_gbm.target_mad}).",
            "",
            f"Median {np.median(absolute_difference):.4f}, 95th percentile {np.quantile(absolute_difference, 0.95):.4f}, max {absolute_difference.max():.4f}.",
            "",
            "### By time to expiry (trading days)",
            "",
            format_table(expiry_table),
            "",
            "### By moneyness ln(S/K)",
            "",
            format_table(moneyness_table),
            "",
            "Plot: `results/validation/gbm_hedge_vs_delta.png`.",
            "",
            "## P&L on the same test paths",
            "",
            format_table(pnl_table),
            "",
            "## Verdict",
            "",
            f"**{'PASS' if passed else 'FAIL'}**: mean absolute difference {mad:.4f} {'<' if passed else '>='} {settings_gbm.target_mad}.",
            "",
        ]
    )
    (validation_directory / "gbm_validation.md").write_text(report)
    return GbmValidationOutcome(mad=mad, passed=passed, report=report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    outcome = run_gbm_validation(load_config(arguments.config))
    print(outcome.report)
    print("CHECKPOINT B:", "PASS" if outcome.passed else "FAIL")
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    sys.exit(main())
