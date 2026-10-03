import numpy as np
import pandas as pd

from black_scholes import atm_call_premium
from real_data_test import build_windows, garch_volatility_series, window_starts
from simulators import GarchParams, annualize_percent_variance

PARAMS = GarchParams(mu=0.04, omega=0.017, alpha=0.011, gamma=0.13, beta=0.913, nu=8.6)


def synthetic_market(n_days: int = 300) -> tuple[pd.Series, pd.Series]:
    dates = pd.bdate_range("2019-06-03", periods=n_days)
    rng = np.random.default_rng(0)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days))), index=dates)
    vix = pd.Series(15.0 + rng.uniform(0, 5, n_days), index=dates)
    return prices, vix


def test_window_starts() -> None:
    assert window_starts(64, 21, overlapping=False) == [0, 21, 42]
    assert len(window_starts(64, 21, overlapping=True)) == 43


def test_garch_volatility_uses_only_past_returns() -> None:
    prices, _ = synthetic_market()
    volatility = garch_volatility_series(prices, PARAMS, 252)
    shocked = prices.copy()
    shocked.iloc[200:] *= 0.8
    shocked_volatility = garch_volatility_series(shocked, PARAMS, 252)
    np.testing.assert_allclose(volatility.iloc[:200].to_numpy(), shocked_volatility.iloc[:200].to_numpy())
    assert shocked_volatility.iloc[200] > volatility.iloc[200]
    assert abs(volatility.iloc[0] - annualize_percent_variance(np.array(PARAMS.unconditional_variance), 252)) < 1e-12


def test_build_windows_normalizes_and_prices_with_vix() -> None:
    prices, vix = synthetic_market()
    windows = build_windows(prices, vix, PARAMS, "2020-01-01", 21, 252, 0.065, overlapping=False)
    test_days = int((prices.index >= "2020-01-01").sum())
    assert windows.prices.shape == (len(window_starts(test_days, 21, False)), 22)
    np.testing.assert_allclose(windows.prices[:, 0], 1.0)
    first_start = windows.start_dates[0]
    assert first_start >= pd.Timestamp("2020-01-01")
    expected_sigma = vix.loc[first_start] / 100.0
    assert abs(windows.sigma0[0] - expected_sigma) < 1e-12
    assert abs(windows.premiums[0] - atm_call_premium(0.065, expected_sigma, 21 / 252)) < 1e-12
    end = prices.index.get_loc(first_start) + 21
    assert abs(windows.prices[0, -1] - prices.iloc[end] / prices.loc[first_start]) < 1e-12
