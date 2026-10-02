from __future__ import annotations

import argparse
import json

import torch

from baselines import DeltaHedgePolicy, UnhedgedPolicy, run_policy, search_band_width
from black_scholes import atm_call_premium
from config import cost_rate_from_bps, load_config, load_fitted_params, resolve_sigma0, seed_everything
from losses import exact_cvar
from simulators import build_simulator

BASELINE_SIMULATORS = ("garch", "heston")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    seed_everything(config.seed)
    fitted = load_fitted_params(config)
    sigma0 = resolve_sigma0(config)
    rate = config.option.r
    alpha = config.risk.alpha
    premium = atm_call_premium(rate, sigma0, config.option.maturity)
    print(f"sigma0 = {sigma0:.4f}, premium C0 = {premium:.6f} (fraction of S0)")
    summary: dict[str, dict] = {"sigma0": sigma0, "premium": premium, "band_search": {}}
    for simulator_name in BASELINE_SIMULATORS:
        simulator = build_simulator(simulator_name, config, fitted)
        validation = simulator.generate(config.simulation.n_val_paths, config.simulation.val_seed)
        prices, volatility = validation.to_torch(dtype=torch.float64)
        summary["band_search"][simulator_name] = {}
        for cost_bps in config.costs.cost_grid:
            cost_rate = cost_rate_from_bps(cost_bps)
            search = search_band_width(prices, validation.times, volatility, premium, 1.0, rate, cost_rate, alpha, config.baselines.band_grid)
            arguments_for_run = (prices, validation.times, volatility, premium, 1.0, rate, cost_rate)
            delta_cvar = float(exact_cvar(run_policy(DeltaHedgePolicy(), *arguments_for_run).loss, alpha))
            unhedged_cvar = float(exact_cvar(run_policy(UnhedgedPolicy(), *arguments_for_run).loss, alpha))
            best_cvar = search.cvar_by_half_width[search.best_half_width]
            summary["band_search"][simulator_name][str(cost_bps)] = {
                "best_half_width": search.best_half_width,
                "validation_cvar_by_half_width": {str(width): value for width, value in search.cvar_by_half_width.items()},
                "validation_cvar_band": best_cvar,
                "validation_cvar_bs_delta": delta_cvar,
                "validation_cvar_unhedged": unhedged_cvar,
            }
            print(
                f"{simulator_name:7s} cost {cost_bps:>4} bps: best band {search.best_half_width:.3f}, "
                f"val CVaR95 band {best_cvar / premium:.3f}, delta {delta_cvar / premium:.3f}, unhedged {unhedged_cvar / premium:.3f} (x premium)"
            )
    output_path = config.paths.resolve("metrics") / "baseline_band_search.json"
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
