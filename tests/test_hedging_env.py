import numpy as np
import pytest
import torch

from baselines import DeltaBandPolicy, DeltaHedgePolicy, UnhedgedPolicy, run_policy, search_band_width
from black_scholes import call_price_np
from hedging_env import HedgeState, constant_volatility, run_hedge
from simulators import GBMSimulator


class FixedSchedulePolicy:
    def __init__(self, schedule: list[float]) -> None:
        self.schedule = schedule

    def __call__(self, state: HedgeState) -> torch.Tensor:
        return torch.full_like(state.spot, self.schedule[state.step])


def test_three_step_hand_computed_pnl() -> None:
    prices = torch.tensor([[1.0, 1.1, 0.9, 1.2]], dtype=torch.float64)
    times = np.array([0.0, 0.1, 0.2, 0.3])
    outcome = run_hedge(prices, times, FixedSchedulePolicy([0.5, 0.8, 0.3]), 0.1, 1.0, 0.05, 0.001, constant_volatility(prices, 0.2))
    assert abs(float(outcome.pnl[0]) - (-0.128759275322)) < 1e-10
    np.testing.assert_allclose(outcome.costs[0].numpy(), [0.0005, 0.00033, 0.00045, 0.00036], atol=1e-15)
    np.testing.assert_allclose(outcome.positions[0].numpy(), [0.5, 0.8, 0.3])
    assert abs(float(outcome.payoff[0]) - 0.2) < 1e-15


def test_unhedged_pnl_is_grown_premium_minus_payoff() -> None:
    prices = torch.tensor([[1.0, 1.05, 1.02], [1.0, 0.95, 0.9]], dtype=torch.float64)
    times = np.array([0.0, 0.5, 1.0])
    outcome = run_hedge(prices, times, UnhedgedPolicy(), 0.07, 1.0, 0.04, 0.01, constant_volatility(prices, 0.2))
    expected = 0.07 * np.exp(0.04) - np.maximum(prices[:, -1].numpy() - 1.0, 0.0)
    np.testing.assert_allclose(outcome.pnl.numpy(), expected, atol=1e-14)
    assert float(outcome.total_cost.abs().sum()) == 0.0


def test_full_hedge_from_start_has_no_market_exposure_without_costs() -> None:
    prices = torch.tensor([[1.0, 1.3, 0.7, 1.5]], dtype=torch.float64)
    times = np.array([0.0, 0.1, 0.2, 0.3])
    outcome = run_hedge(prices, times, FixedSchedulePolicy([1.0, 1.0, 1.0]), 0.0, 1.0, 0.0, 0.0, constant_volatility(prices, 0.2))
    assert abs(float(outcome.pnl[0]) - (0.5 - 0.5)) < 1e-14


def test_rejects_mismatched_grid() -> None:
    prices = torch.ones(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError):
        run_hedge(prices, np.array([0.0, 0.1, 0.2]), UnhedgedPolicy(), 0.0, 1.0, 0.0, 0.0, constant_volatility(prices, 0.2))


def test_zero_width_band_equals_delta_hedge() -> None:
    paths = GBMSimulator(0.18, 0.065, 21 / 252, 21).generate(2000, 3)
    prices, volatility = paths.to_torch(dtype=torch.float64)
    arguments = (prices, paths.times, volatility, 0.02, 1.0, 0.065, 0.0005)
    band = run_policy(DeltaBandPolicy(0.0), *arguments)
    delta = run_policy(DeltaHedgePolicy(), *arguments)
    np.testing.assert_allclose(band.pnl.numpy(), delta.pnl.numpy(), atol=1e-14)


def test_band_reduces_cost_and_search_picks_from_grid() -> None:
    paths = GBMSimulator(0.18, 0.065, 21 / 252, 21).generate(5000, 4)
    prices, volatility = paths.to_torch(dtype=torch.float64)
    premium = float(call_price_np(np.array(1.0), 1.0, 21 / 252, 0.065, 0.18))
    arguments = (prices, paths.times, volatility, premium, 1.0, 0.065, 0.005)
    narrow = run_policy(DeltaBandPolicy(0.0), *arguments)
    wide = run_policy(DeltaBandPolicy(0.1), *arguments)
    assert float(wide.total_cost.mean()) < float(narrow.total_cost.mean())
    grid = [0.0, 0.05, 0.1]
    result = search_band_width(prices, paths.times, volatility, premium, 1.0, 0.065, 0.005, 0.95, grid)
    assert result.best_half_width in grid
    assert set(result.cvar_by_half_width) == set(grid)


def test_delta_hedge_std_decreases_with_rebalancing() -> None:
    maturity, rate, sigma, fine_steps = 21 / 252, 0.065, 0.18, 84
    paths = GBMSimulator(sigma, rate, maturity, fine_steps).generate(20000, 11)
    prices, volatility = paths.to_torch(dtype=torch.float64)
    premium = float(call_price_np(np.array(1.0), 1.0, maturity, rate, sigma))
    stds = []
    for n_rebalances in [1, 2, 4, 7, 21, 84]:
        stride = fine_steps // n_rebalances
        outcome = run_policy(DeltaHedgePolicy(), prices[:, ::stride], paths.times[::stride], volatility[:, ::stride], premium, 1.0, rate, 0.0)
        stds.append(float(outcome.pnl.std()))
    assert all(later < earlier for earlier, later in zip(stds[:-1], stds[1:]))
