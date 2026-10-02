from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SimulatedPaths:
    prices: np.ndarray
    volatility: np.ndarray
    times: np.ndarray

    def to_torch(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        prices = torch.as_tensor(self.prices, dtype=dtype, device=device)
        volatility = torch.as_tensor(self.volatility, dtype=dtype, device=device)
        return prices, volatility


class GBMSimulator:
    name = "gbm"

    def __init__(self, sigma: float, mu: float, maturity: float, n_steps: int) -> None:
        self.sigma = sigma
        self.mu = mu
        self.maturity = maturity
        self.n_steps = n_steps

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.maturity, self.n_steps + 1)

    def generate(self, n_paths: int, seed: int) -> SimulatedPaths:
        rng = np.random.default_rng(seed)
        dt = self.maturity / self.n_steps
        shocks = rng.standard_normal((n_paths, self.n_steps))
        increments = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * shocks
        log_prices = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
        prices = np.exp(log_prices)
        return SimulatedPaths(prices=prices, volatility=np.full_like(prices, self.sigma), times=self.times)
