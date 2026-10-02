import numpy as np
import pytest

from simulators import (
    GarchParams,
    arch_reference_simulation,
    GarchSimulator,
    GBMSimulator,
    HestonParams,
    HestonSimulator,
    compare_with_arch,
    filter_garch_variance,
    recursion_matches_arch,
    simulate_garch_returns,
)

PARAMS = GarchParams(mu=0.04, omega=0.017, alpha=0.011, gamma=0.13, beta=0.913, nu=8.6)
HESTON = HestonParams(kappa=3.0, theta=0.0478, xi=0.4, rho=-0.7, v0=0.0478, mu=0.065)


def build_all() -> list:
    return [
        GBMSimulator(0.18, 0.065, 21 / 252, 21),
        GarchSimulator(PARAMS, 21, 252, 250),
        HestonSimulator(HESTON, 21, 252, 4),
    ]


@pytest.mark.parametrize("simulator", build_all(), ids=lambda simulator: simulator.name)
def test_shapes_and_normalization(simulator: object) -> None:
    paths = simulator.generate(500, 1)
    assert paths.prices.shape == (500, 22)
    assert paths.volatility.shape == (500, 22)
    assert paths.times.shape == (22,)
    assert np.allclose(paths.prices[:, 0], 1.0)
    assert abs(paths.times[-1] - 21 / 252) < 1e-12
    assert np.all(paths.prices > 0)
    assert np.all(paths.volatility >= 0)


@pytest.mark.parametrize("simulator", build_all(), ids=lambda simulator: simulator.name)
def test_seed_reproducibility(simulator: object) -> None:
    first = simulator.generate(300, 7)
    second = simulator.generate(300, 7)
    third = simulator.generate(300, 8)
    np.testing.assert_array_equal(first.prices, second.prices)
    np.testing.assert_array_equal(first.volatility, second.volatility)
    assert not np.allclose(first.prices, third.prices)


def test_gbm_moments() -> None:
    paths = GBMSimulator(0.2, 0.05, 1.0, 50).generate(200000, 3)
    log_terminal = np.log(paths.prices[:, -1])
    assert abs(log_terminal.std() - 0.2) < 0.002
    assert abs(paths.prices[:, -1].mean() - np.exp(0.05)) < 0.003


def test_garch_simulated_moments_match_fitted_model() -> None:
    returns, variances = simulate_garch_returns(PARAMS, 8000, 500, 500, np.random.default_rng(5))
    assert abs(returns.mean() - PARAMS.mu) < 0.01
    assert abs(returns.std() / np.sqrt(PARAMS.unconditional_variance) - 1.0) < 0.05
    _, reference_volatility = arch_reference_simulation(PARAMS, 4000000, 500, 6)
    assert abs(np.median(variances) / np.median(reference_volatility**2) - 1.0) < 0.05


def test_garch_custom_recursion_matches_arch_exactly() -> None:
    assert recursion_matches_arch(PARAMS, 2000, 3) < 1e-10


def test_garch_custom_simulation_matches_arch_distribution() -> None:
    comparison = compare_with_arch(PARAMS, 2000, 1000, 500, 11)
    custom, reference = comparison["custom"], comparison["arch"]
    assert abs(custom["std"] / reference["std"] - 1.0) < 0.1
    for key in ["quantile_0.01", "quantile_0.05", "quantile_0.95", "quantile_0.99"]:
        assert abs(custom[key] / reference[key] - 1.0) < 0.05
    for lag in [1, 5, 10]:
        assert abs(custom[f"acf_abs_lag{lag}"] - reference[f"acf_abs_lag{lag}"]) < 0.05
        assert custom[f"acf_abs_lag{lag}"] > 0.1


def test_garch_filter_reacts_asymmetrically() -> None:
    down = filter_garch_variance(PARAMS, np.array([PARAMS.mu - 2.0]))
    up = filter_garch_variance(PARAMS, np.array([PARAMS.mu + 2.0]))
    assert down[-1] > up[-1]


def test_heston_moments() -> None:
    simulator = HestonSimulator(HESTON, 21, 252, 4)
    paths = simulator.generate(100000, 9)
    assert abs(paths.prices[:, -1].mean() - np.exp(HESTON.mu * 21 / 252)) < 0.002
    assert abs(np.mean(paths.volatility**2) / HESTON.theta - 1.0) < 0.05
    log_returns = np.diff(np.log(paths.prices), axis=1)
    assert abs(log_returns.std() * np.sqrt(252) / np.sqrt(HESTON.theta) - 1.0) < 0.05
    leverage = np.corrcoef(log_returns[:, -1], paths.volatility[:, -1] - paths.volatility[:, -2])[0, 1]
    assert leverage < -0.5
