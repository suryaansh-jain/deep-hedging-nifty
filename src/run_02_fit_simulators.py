from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import fitted_params_path, load_config, seed_everything
from data_nifty import load_nifty, log_returns, split_before
from simulators import (
    GarchSimulator,
    HestonParams,
    HestonSimulator,
    compare_with_arch,
    recursion_matches_arch,
    fit_gjr_garch,
    return_moments,
)

ARCH_CHECK_PATHS = 2000
ARCH_CHECK_LENGTH = 2000
RECURSION_CHECK_LENGTH = 5000


def resolve_heston(config_value: float | str, default: float) -> float:
    return default if config_value == "auto" else float(config_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    seed_everything(config.seed)
    days_per_year = config.option.trading_days_per_year
    training_returns = split_before(log_returns(load_nifty(config)), config.data.test_start_date)
    returns_percent = training_returns.to_numpy() * 100.0
    garch = fit_gjr_garch(returns_percent)
    realized_variance = float(np.var(training_returns.to_numpy(), ddof=1) * days_per_year)
    heston = HestonParams(
        kappa=config.heston.kappa,
        theta=resolve_heston(config.heston.theta, realized_variance),
        xi=config.heston.xi,
        rho=config.heston.rho,
        v0=resolve_heston(config.heston.v0, realized_variance),
        mu=config.heston.mu,
    )
    fitted = {
        "fit_sample": {
            "start": str(training_returns.index[0].date()),
            "end": str(training_returns.index[-1].date()),
            "n_returns": int(len(training_returns)),
        },
        "garch": {
            "params": garch.__dict__,
            "persistence": garch.persistence,
            "fourth_moment_factor": garch.fourth_moment_factor,
            "unconditional_daily_variance_percent2": garch.unconditional_variance,
            "unconditional_annual_vol": garch.unconditional_annual_vol(days_per_year),
        },
        "heston": heston.__dict__,
        "heston_feller_satisfied": heston.satisfies_feller,
        "realized_annual_vol_pre_test": float(np.sqrt(realized_variance)),
    }
    with open(fitted_params_path(config), "w") as handle:
        json.dump(fitted, handle, indent=2)
    print(json.dumps(fitted, indent=2))

    comparison = compare_with_arch(garch, ARCH_CHECK_PATHS, ARCH_CHECK_LENGTH, config.garch.burn_in, config.seed)
    comparison["historical_pre_test"] = return_moments(returns_percent)
    implied_std = float(np.sqrt(garch.unconditional_variance))
    table = pd.DataFrame(comparison)
    table.loc["std", "fitted_model"] = implied_std
    table.loc["mean", "fitted_model"] = garch.mu
    validation_directory = config.paths.resolve("validation")
    table.to_csv(validation_directory / "garch_simulation_check.csv")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))
    recursion_error = recursion_matches_arch(garch, RECURSION_CHECK_LENGTH, config.seed)
    print(f"max relative difference between custom variance recursion and arch volatility path: {recursion_error:.2e}")
    print(
        f"fourth moment factor E[(alpha + gamma 1[z<0]) z^2 + beta]^2 = {garch.fourth_moment_factor:.4f}; "
        "above 1 means the unconditional kurtosis is infinite, so sample kurtosis and squared-return ACFs are not stable and are reported only"
    )

    garch_paths = GarchSimulator(garch, config.option.maturity_days, days_per_year, config.garch.burn_in).generate(20000, config.seed)
    heston_paths = HestonSimulator(heston, config.option.maturity_days, days_per_year, config.heston.substeps_per_day).generate(20000, config.seed)
    for name, paths in [("garch", garch_paths), ("heston", heston_paths)]:
        daily_log_returns = np.diff(np.log(paths.prices), axis=1)
        print(
            f"{name}: annualized std of daily returns {daily_log_returns.std() * np.sqrt(days_per_year):.4f}, "
            f"terminal price 1%/50%/99% {np.percentile(paths.prices[:, -1], [1, 50, 99]).round(4).tolist()}, "
            f"vol feature 5%/50%/95% {np.percentile(paths.volatility[:, 0], [5, 50, 95]).round(4).tolist()}"
        )


if __name__ == "__main__":
    main()
