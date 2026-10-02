from __future__ import annotations

import argparse
import json

from black_scholes import atm_call_premium
from config import Config, cost_rate_from_bps, load_config, load_fitted_params, resolve_sigma0, seed_everything, select_device
from policies import build_policy
from simulators import build_simulator
from train import run_name, save_training_result, settings_from_config, train_policy

MAIN_SIMULATOR = "garch"
ROBUSTNESS_SIMULATOR = "heston"


def train_grid(config: Config, simulator_name: str, policy_names: list[str], cost_grid: list[float], summary: dict) -> None:
    fitted = load_fitted_params(config)
    sigma0 = resolve_sigma0(config)
    premium = atm_call_premium(config.option.r, sigma0, config.option.maturity)
    simulator = build_simulator(simulator_name, config, fitted)
    validation = simulator.generate(config.simulation.n_val_paths, config.simulation.val_seed)
    device = select_device(config.device)
    for policy_name in policy_names:
        for cost_bps in cost_grid:
            seed_everything(config.seed)
            name = run_name(policy_name, simulator_name, cost_bps)
            policy = build_policy(policy_name, sigma0, config.policy.hidden_size, config.policy.hidden_layers)
            settings = settings_from_config(config, premium, cost_rate_from_bps(cost_bps))
            result = train_policy(policy, simulator, validation, settings, device, log_prefix=f"[{name}] ")
            metadata = {
                "policy": policy_name,
                "simulator": simulator_name,
                "cost_bps": float(cost_bps),
                "sigma0": sigma0,
                "premium": premium,
                "config": config.source_path,
            }
            save_training_result(result, name, config.paths.resolve("models"), config.paths.resolve("metrics") / "training_curves", metadata)
            summary[name] = {**metadata, "best_epoch": result.best_epoch, "epochs_run": len(result.history), "best_val_cvar": result.best_val_cvar, "best_val_cvar_x_premium": result.best_val_cvar / premium, "seconds": result.seconds}
            print(f"[{name}] done: best val CVaR95 {result.best_val_cvar / premium:.4f} x premium at epoch {result.best_epoch} in {result.seconds:.0f} s", flush=True)


def choose_robustness_policy(config: Config, summary: dict) -> str:
    if config.training.heston_policy != "auto":
        return config.training.heston_policy
    default_cost = config.costs.cost_bps
    candidates = {name: summary[run_name(name, MAIN_SIMULATOR, default_cost)]["best_val_cvar"] for name in config.training.policies}
    return min(candidates, key=lambda name: candidates[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    summary: dict[str, dict] = {}
    train_grid(config, MAIN_SIMULATOR, list(config.training.policies), list(config.costs.cost_grid), summary)
    robustness_policy = choose_robustness_policy(config, summary)
    print(f"robustness check on {ROBUSTNESS_SIMULATOR} with policy {robustness_policy}", flush=True)
    train_grid(config, ROBUSTNESS_SIMULATOR, [robustness_policy], list(config.costs.cost_grid), summary)
    output = {"robustness_policy": robustness_policy, "runs": summary}
    with open(config.paths.resolve("metrics") / "training_summary.json", "w") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
