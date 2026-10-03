from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from costs import proportional_cost


@dataclass
class HedgeState:
    step: int
    time: float
    time_to_maturity: float
    maturity: float
    spot: torch.Tensor
    h_prev: torch.Tensor
    volatility: torch.Tensor
    strike: float
    rate: float
    wealth: torch.Tensor
    premium: torch.Tensor


Policy = Callable[[HedgeState], torch.Tensor]


@dataclass
class HedgeOutcome:
    pnl: torch.Tensor
    positions: torch.Tensor
    costs: torch.Tensor
    payoff: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return -self.pnl

    @property
    def total_cost(self) -> torch.Tensor:
        return self.costs.sum(dim=1)


def as_time_tensor(times: np.ndarray | torch.Tensor) -> list[float]:
    values = times.tolist() if isinstance(times, torch.Tensor) else np.asarray(times, dtype=float).tolist()
    if any(later <= earlier for earlier, later in zip(values[:-1], values[1:])):
        raise ValueError("time grid must be strictly increasing")
    return values


def constant_volatility(prices: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.full_like(prices, sigma)


def run_hedge(
    prices: torch.Tensor,
    times: np.ndarray | torch.Tensor,
    policy: Policy,
    premium: float | torch.Tensor,
    strike: float,
    rate: float,
    cost_rate: float,
    volatility: torch.Tensor,
) -> HedgeOutcome:
    grid = as_time_tensor(times)
    n_steps = len(grid) - 1
    if prices.shape[1] != n_steps + 1 or volatility.shape != prices.shape:
        raise ValueError("prices and volatility must have one column per time grid point")
    maturity = grid[-1]
    n_paths = prices.shape[0]
    h_prev = torch.zeros(n_paths, dtype=prices.dtype, device=prices.device)
    premium_per_path = torch.as_tensor(premium, dtype=prices.dtype, device=prices.device).expand(n_paths).clone()
    cash = premium_per_path.clone()
    positions: list[torch.Tensor] = []
    costs: list[torch.Tensor] = []
    for step in range(n_steps):
        if step > 0:
            cash = cash * float(np.exp(rate * (grid[step] - grid[step - 1])))
        spot = prices[:, step]
        state = HedgeState(
            step=step,
            time=grid[step],
            time_to_maturity=maturity - grid[step],
            maturity=maturity,
            spot=spot,
            h_prev=h_prev,
            volatility=volatility[:, step],
            strike=strike,
            rate=rate,
            wealth=cash + h_prev * spot,
            premium=premium_per_path,
        )
        h_new = policy(state)
        trade = h_new - h_prev
        step_cost = proportional_cost(spot, trade, cost_rate)
        cash = cash - trade * spot - step_cost
        positions.append(h_new)
        costs.append(step_cost)
        h_prev = h_new
    cash = cash * float(np.exp(rate * (grid[n_steps] - grid[n_steps - 1])))
    terminal_spot = prices[:, n_steps]
    unwind_cost = proportional_cost(terminal_spot, h_prev, cost_rate)
    payoff = torch.clamp(terminal_spot - strike, min=0.0)
    pnl = cash + h_prev * terminal_spot - unwind_cost - payoff
    costs.append(unwind_cost)
    return HedgeOutcome(pnl=pnl, positions=torch.stack(positions, dim=1), costs=torch.stack(costs, dim=1), payoff=payoff)
