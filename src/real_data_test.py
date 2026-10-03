from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from black_scholes import atm_call_premium
from evaluate import run_strategies
from simulators import GarchParams, annualize_percent_variance, filter_garch_variance


@dataclass
class RealWindows:
    start_dates: list[pd.Timestamp]
    prices: np.ndarray
    volatility: np.ndarray
    premiums: np.ndarray
    sigma0: np.ndarray
    times: np.ndarray


def garch_volatility_series(prices: pd.Series, params: GarchParams, trading_days_per_year: int) -> pd.Series:
    returns_percent = np.log(prices).diff().dropna().to_numpy() * 100.0
    variances = filter_garch_variance(params, returns_percent)
    return pd.Series(annualize_percent_variance(variances, trading_days_per_year), index=prices.index)


def window_starts(n_prices: int, window_length: int, overlapping: bool) -> list[int]:
    stride = 1 if overlapping else window_length
    return list(range(0, n_prices - window_length, stride))


def build_windows(
    prices: pd.Series,
    vix: pd.Series,
    params: GarchParams,
    test_start_date: str,
    window_length: int,
    trading_days_per_year: int,
    rate: float,
    overlapping: bool,
) -> RealWindows:
    volatility = garch_volatility_series(prices, params, trading_days_per_year)
    test_prices = prices[prices.index >= pd.Timestamp(test_start_date)]
    test_volatility = volatility.loc[test_prices.index]
    vix_aligned = vix.reindex(test_prices.index, method="ffill")
    maturity = window_length / trading_days_per_year
    starts, price_rows, volatility_rows, premiums, sigmas = [], [], [], [], []
    for start in window_starts(len(test_prices), window_length, overlapping):
        implied_vol = vix_aligned.iloc[start]
        if not np.isfinite(implied_vol):
            continue
        window_prices = test_prices.iloc[start : start + window_length + 1].to_numpy()
        starts.append(test_prices.index[start])
        price_rows.append(window_prices / window_prices[0])
        volatility_rows.append(test_volatility.iloc[start : start + window_length + 1].to_numpy())
        sigma = float(implied_vol) / 100.0
        sigmas.append(sigma)
        premiums.append(atm_call_premium(rate, sigma, maturity))
    return RealWindows(
        start_dates=starts,
        prices=np.array(price_rows),
        volatility=np.array(volatility_rows),
        premiums=np.array(premiums),
        sigma0=np.array(sigmas),
        times=np.arange(window_length + 1) / trading_days_per_year,
    )


def run_real_windows(strategies: dict[str, object], windows: RealWindows, rate: float, cost_rate: float) -> dict[str, dict[str, np.ndarray]]:
    prices = torch.as_tensor(windows.prices, dtype=torch.float32)
    volatility = torch.as_tensor(windows.volatility, dtype=torch.float32)
    premiums = torch.as_tensor(windows.premiums, dtype=torch.float32)
    results = run_strategies(strategies, prices, volatility, windows.times, premiums, rate, cost_rate)
    return {
        name: {
            "pnl_x_premium": result.pnl / windows.premiums,
            "cost_x_premium": result.total_cost / windows.premiums,
            "turnover": result.turnover,
            "positions": result.positions,
        }
        for name, result in results.items()
    }
