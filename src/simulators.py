from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import Config


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


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    @property
    def fourth_moment_factor(self) -> float:
        standardized_fourth_moment = 3.0 * (self.nu - 2.0) / (self.nu - 4.0)
        squared_shock_terms = self.alpha**2 + self.alpha * self.gamma + 0.5 * self.gamma**2
        return squared_shock_terms * standardized_fourth_moment + 2.0 * self.beta * (self.alpha + 0.5 * self.gamma) + self.beta**2

    def unconditional_annual_vol(self, trading_days_per_year: int) -> float:
        return float(np.sqrt(self.unconditional_variance * trading_days_per_year) / 100.0)

    def as_arch_vector(self) -> np.ndarray:
        return np.array([self.mu, self.omega, self.alpha, self.gamma, self.beta, self.nu])


def fit_gjr_garch(returns_percent: np.ndarray) -> GarchParams:
    from arch import arch_model

    model = arch_model(returns_percent, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
    fitted = model.fit(disp="off")
    values = fitted.params
    return GarchParams(
        mu=float(values["mu"]),
        omega=float(values["omega"]),
        alpha=float(values["alpha[1]"]),
        gamma=float(values["gamma[1]"]),
        beta=float(values["beta[1]"]),
        nu=float(values["nu"]),
    )


def standardized_t(rng: np.random.Generator, nu: float, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_t(nu, size=shape) * np.sqrt((nu - 2.0) / nu)


def next_garch_variance(params: GarchParams, variance: np.ndarray, shock: np.ndarray) -> np.ndarray:
    leverage = params.alpha + params.gamma * (shock < 0.0)
    return params.omega + leverage * shock * shock + params.beta * variance


def simulate_garch_returns(params: GarchParams, n_paths: int, n_steps: int, burn_in: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    variance = np.full(n_paths, params.unconditional_variance)
    for _ in range(burn_in):
        shock = np.sqrt(variance) * standardized_t(rng, params.nu, (n_paths,))
        variance = next_garch_variance(params, variance, shock)
    returns = np.empty((n_paths, n_steps))
    variances = np.empty((n_paths, n_steps + 1))
    for step in range(n_steps):
        variances[:, step] = variance
        shock = np.sqrt(variance) * standardized_t(rng, params.nu, (n_paths,))
        returns[:, step] = params.mu + shock
        variance = next_garch_variance(params, variance, shock)
    variances[:, n_steps] = variance
    return returns, variances


def filter_garch_variance(params: GarchParams, returns_percent: np.ndarray, initial_variance: float | None = None) -> np.ndarray:
    variances = np.empty(len(returns_percent) + 1)
    variances[0] = params.unconditional_variance if initial_variance is None else initial_variance
    for index, value in enumerate(returns_percent):
        variances[index + 1] = next_garch_variance(params, np.array(variances[index]), np.array(value - params.mu))
    return variances


def annualize_percent_variance(variance: np.ndarray, trading_days_per_year: int) -> np.ndarray:
    return np.sqrt(variance * trading_days_per_year) / 100.0


class GarchSimulator:
    name = "garch"

    def __init__(self, params: GarchParams, maturity_days: int, trading_days_per_year: int, burn_in: int) -> None:
        self.params = params
        self.maturity_days = maturity_days
        self.trading_days_per_year = trading_days_per_year
        self.burn_in = burn_in

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.maturity_days + 1) / self.trading_days_per_year

    def generate(self, n_paths: int, seed: int) -> SimulatedPaths:
        rng = np.random.default_rng(seed)
        returns, variances = simulate_garch_returns(self.params, n_paths, self.maturity_days, self.burn_in, rng)
        log_prices = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(returns / 100.0, axis=1)], axis=1)
        volatility = annualize_percent_variance(variances, self.trading_days_per_year)
        return SimulatedPaths(prices=np.exp(log_prices), volatility=volatility, times=self.times)


@dataclass
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    mu: float

    @property
    def satisfies_feller(self) -> bool:
        return 2.0 * self.kappa * self.theta >= self.xi**2


class HestonSimulator:
    name = "heston"

    def __init__(self, params: HestonParams, maturity_days: int, trading_days_per_year: int, substeps_per_day: int) -> None:
        self.params = params
        self.maturity_days = maturity_days
        self.trading_days_per_year = trading_days_per_year
        self.substeps_per_day = substeps_per_day

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.maturity_days + 1) / self.trading_days_per_year

    def generate(self, n_paths: int, seed: int) -> SimulatedPaths:
        rng = np.random.default_rng(seed)
        params = self.params
        dt = 1.0 / (self.trading_days_per_year * self.substeps_per_day)
        sqrt_dt = np.sqrt(dt)
        orthogonal_weight = np.sqrt(1.0 - params.rho**2)
        log_price = np.zeros(n_paths)
        variance = np.full(n_paths, params.v0)
        log_prices = np.empty((n_paths, self.maturity_days + 1))
        variances = np.empty((n_paths, self.maturity_days + 1))
        log_prices[:, 0] = log_price
        variances[:, 0] = variance
        for day in range(self.maturity_days):
            for _ in range(self.substeps_per_day):
                first = rng.standard_normal(n_paths)
                second = params.rho * first + orthogonal_weight * rng.standard_normal(n_paths)
                positive_variance = np.maximum(variance, 0.0)
                log_price = log_price + (params.mu - 0.5 * positive_variance) * dt + np.sqrt(positive_variance) * sqrt_dt * first
                variance = variance + params.kappa * (params.theta - positive_variance) * dt + params.xi * np.sqrt(positive_variance) * sqrt_dt * second
            log_prices[:, day + 1] = log_price
            variances[:, day + 1] = variance
        volatility = np.sqrt(np.maximum(variances, 0.0))
        return SimulatedPaths(prices=np.exp(log_prices), volatility=volatility, times=self.times)


ACF_LAGS = (1, 5, 10, 20)
QUANTILES = (0.01, 0.05, 0.95, 0.99)


def lagged_correlation(values: np.ndarray, lag: int) -> float:
    centered = np.atleast_2d(values) - np.mean(values)
    numerator = np.mean(centered[:, lag:] * centered[:, :-lag])
    return float(numerator / np.mean(centered * centered))


def return_moments(returns: np.ndarray) -> dict[str, float]:
    flat = np.asarray(returns).ravel()
    centered = flat - flat.mean()
    variance = np.mean(centered**2)
    summary = {
        "mean": float(flat.mean()),
        "std": float(np.sqrt(variance)),
        "excess_kurtosis": float(np.mean(centered**4) / variance**2 - 3.0),
    }
    for quantile in QUANTILES:
        summary[f"quantile_{quantile:g}"] = float(np.quantile(flat, quantile))
    absolute = np.abs(np.atleast_2d(returns) - flat.mean())
    for lag in ACF_LAGS:
        summary[f"acf_sq_lag{lag}"] = lagged_correlation(absolute**2, lag)
        summary[f"acf_abs_lag{lag}"] = lagged_correlation(absolute, lag)
    return summary


def arch_reference_simulation(params: GarchParams, n_obs: int, burn_in: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from arch.univariate import GARCH, ConstantMean, StudentsT

    model = ConstantMean(None, volatility=GARCH(p=1, o=1, q=1), distribution=StudentsT(seed=seed))
    simulated = model.simulate(params.as_arch_vector(), n_obs, burn=burn_in)
    return simulated["data"].to_numpy(), simulated["volatility"].to_numpy()


def recursion_matches_arch(params: GarchParams, n_obs: int, seed: int) -> float:
    returns, volatility = arch_reference_simulation(params, n_obs, 500, seed)
    filtered = filter_garch_variance(params, returns, initial_variance=volatility[0] ** 2)
    return float(np.max(np.abs(np.sqrt(filtered[:-1]) / volatility - 1.0)))


def compare_with_arch(params: GarchParams, n_paths: int, path_length: int, burn_in: int, seed: int) -> dict[str, dict[str, float]]:
    custom_returns, _ = simulate_garch_returns(params, n_paths, path_length, burn_in, np.random.default_rng(seed))
    reference_returns, _ = arch_reference_simulation(params, n_paths * path_length, burn_in, seed + 1)
    return {"custom": return_moments(custom_returns), "arch": return_moments(reference_returns)}


def build_simulator(name: str, config: Config, fitted: dict) -> GBMSimulator | GarchSimulator | HestonSimulator:
    option = config.option
    if name == "gbm":
        return GBMSimulator(config.gbm.sigma, config.gbm.mu, option.maturity, option.maturity_days)
    if name == "garch":
        return GarchSimulator(GarchParams(**fitted["garch"]["params"]), option.maturity_days, option.trading_days_per_year, config.garch.burn_in)
    if name == "heston":
        return HestonSimulator(HestonParams(**fitted["heston"]), option.maturity_days, option.trading_days_per_year, config.heston.substeps_per_day)
    raise ValueError(f"unknown simulator {name}")
