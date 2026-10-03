from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from baselines import DeltaBandPolicy, DeltaHedgePolicy, UnhedgedPolicy, run_policy
from config import Config
from costs import turnover
from hedging_env import HedgeOutcome
from losses import exact_cvar_np, tail_count
from policies import build_policy
from train import run_name

LOSS_THRESHOLD_FRACTION = 0.05
DEEP_STRATEGIES = ("mlp", "ntb")
STRATEGY_LABELS = {
    "unhedged": "Unhedged",
    "bs_delta": "BS delta",
    "delta_band": "Delta + no-trade band",
    "mlp": "Deep hedge (MLP)",
    "ntb": "Deep hedge (band network)",
}


@dataclass
class StrategyResult:
    strategy: str
    pnl: np.ndarray
    total_cost: np.ndarray
    turnover: np.ndarray
    positions: np.ndarray


def load_trained_policy(models_directory: Path, policy_name: str, simulator_name: str, cost_bps: float) -> nn.Module | None:
    path = models_directory / f"{run_name(policy_name, simulator_name, cost_bps)}.pt"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    metadata = checkpoint["metadata"]
    state_dict = checkpoint["state_dict"]
    first_weight = next(value for key, value in state_dict.items() if key.endswith("weight"))
    hidden_size = first_weight.shape[0]
    hidden_layers = sum(1 for key in state_dict if key.endswith("weight")) - 1
    policy = build_policy(policy_name, metadata["sigma0"], hidden_size, hidden_layers, bool(metadata.get("include_running_pnl", False)))
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def load_band_widths(config: Config) -> dict[str, dict[str, float]]:
    with open(config.paths.resolve("metrics") / "baseline_band_search.json") as handle:
        search = json.load(handle)["band_search"]
    return {simulator: {cost: values["best_half_width"] for cost, values in by_cost.items()} for simulator, by_cost in search.items()}


def cost_key(cost_bps: float) -> str:
    return str(int(cost_bps)) if float(cost_bps).is_integer() else str(cost_bps)


def strategies_for(config: Config, simulator_name: str, cost_bps: float, band_widths: dict[str, dict[str, float]]) -> dict[str, object]:
    strategies: dict[str, object] = {
        "unhedged": UnhedgedPolicy(),
        "bs_delta": DeltaHedgePolicy(),
        "delta_band": DeltaBandPolicy(band_widths[simulator_name][cost_key(cost_bps)]),
    }
    for policy_name in DEEP_STRATEGIES:
        policy = load_trained_policy(config.paths.resolve("models"), policy_name, simulator_name, cost_bps)
        if policy is not None:
            strategies[policy_name] = policy
    return strategies


def to_result(name: str, outcome: HedgeOutcome) -> StrategyResult:
    return StrategyResult(
        strategy=name,
        pnl=outcome.pnl.double().cpu().numpy(),
        total_cost=outcome.total_cost.double().cpu().numpy(),
        turnover=turnover(outcome.positions).double().cpu().numpy(),
        positions=outcome.positions.cpu().numpy(),
    )


def run_strategies(strategies: dict[str, object], prices: torch.Tensor, volatility: torch.Tensor, times: np.ndarray, premium: float | torch.Tensor, rate: float, cost_rate: float) -> dict[str, StrategyResult]:
    return {name: to_result(name, run_policy(policy, prices, times, volatility, premium, 1.0, rate, cost_rate)) for name, policy in strategies.items()}


def summarize(result: StrategyResult, premium: float) -> dict[str, float]:
    losses = -result.pnl
    return {
        "mean_pnl": float(result.pnl.mean()),
        "std_pnl": float(result.pnl.std(ddof=1)),
        "cvar95": exact_cvar_np(losses, 0.95),
        "cvar99": exact_cvar_np(losses, 0.99),
        "mean_total_cost": float(result.total_cost.mean()),
        "mean_turnover": float(result.turnover.mean()),
        "fraction_loss_above_5pct_premium": float(np.mean(losses > LOSS_THRESHOLD_FRACTION * premium)),
        "n_paths": int(result.pnl.shape[0]),
    }


def resampled_cvar(losses: np.ndarray, indices: np.ndarray, alpha: float) -> np.ndarray:
    count = tail_count(indices.shape[1], alpha)
    values = np.empty(indices.shape[0])
    for row, sample in enumerate(indices):
        resampled = losses[sample]
        values[row] = np.partition(resampled, -count)[-count:].mean()
    return values


def paired_bootstrap_cvar_difference(losses_a: np.ndarray, losses_b: np.ndarray, alpha: float, n_resamples: int, seed: int, chunk: int = 50) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    differences = []
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        indices = rng.integers(0, losses_a.shape[0], size=(size, losses_a.shape[0]))
        differences.append(resampled_cvar(losses_a, indices, alpha) - resampled_cvar(losses_b, indices, alpha))
    stacked = np.concatenate(differences)
    point = exact_cvar_np(losses_a, alpha) - exact_cvar_np(losses_b, alpha)
    return {
        "difference": float(point),
        "ci_low": float(np.quantile(stacked, 0.025)),
        "ci_high": float(np.quantile(stacked, 0.975)),
        "fraction_resamples_below_zero": float(np.mean(stacked < 0)),
        "n_resamples": int(n_resamples),
    }
