from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from black_scholes import call_delta_torch
from hedging_env import HedgeOutcome, HedgeState, run_hedge
from losses import exact_cvar


def bs_delta_for_state(state: HedgeState) -> torch.Tensor:
    return call_delta_torch(state.spot, state.strike, state.time_to_maturity, state.rate, state.volatility)


class UnhedgedPolicy:
    name = "unhedged"

    def __call__(self, state: HedgeState) -> torch.Tensor:
        return torch.zeros_like(state.spot)


class DeltaHedgePolicy:
    name = "bs_delta"

    def __call__(self, state: HedgeState) -> torch.Tensor:
        return bs_delta_for_state(state)


class DeltaBandPolicy:
    name = "delta_band"

    def __init__(self, half_width: float) -> None:
        self.half_width = half_width

    def __call__(self, state: HedgeState) -> torch.Tensor:
        delta = bs_delta_for_state(state)
        lower = torch.clamp(delta - self.half_width, 0.0, 1.0)
        upper = torch.clamp(delta + self.half_width, 0.0, 1.0)
        return torch.minimum(torch.maximum(state.h_prev, lower), upper)


@dataclass
class BandSearchResult:
    best_half_width: float
    cvar_by_half_width: dict[float, float]


def search_band_width(
    prices: torch.Tensor,
    times: np.ndarray,
    volatility: torch.Tensor,
    premium: float,
    strike: float,
    rate: float,
    cost_rate: float,
    alpha: float,
    band_grid: list[float],
) -> BandSearchResult:
    cvar_by_half_width: dict[float, float] = {}
    with torch.no_grad():
        for half_width in band_grid:
            outcome = run_hedge(prices, times, DeltaBandPolicy(half_width), premium, strike, rate, cost_rate, volatility)
            cvar_by_half_width[float(half_width)] = float(exact_cvar(outcome.loss, alpha))
    best = min(cvar_by_half_width, key=lambda width: cvar_by_half_width[width])
    return BandSearchResult(best_half_width=best, cvar_by_half_width=cvar_by_half_width)


def run_policy(
    policy: object,
    prices: torch.Tensor,
    times: np.ndarray,
    volatility: torch.Tensor,
    premium: float,
    strike: float,
    rate: float,
    cost_rate: float,
) -> HedgeOutcome:
    with torch.no_grad():
        return run_hedge(prices, times, policy, premium, strike, rate, cost_rate, volatility)
