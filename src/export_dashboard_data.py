from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

HISTOGRAM_EDGES = np.linspace(-5.0, 2.0, 141)
DASHBOARD_DIRECTORY_NAME = "dashboard"


def histogram_rows(evaluation: dict, simulator_name: str, premium: float) -> list[dict]:
    rows = []
    centers = 0.5 * (HISTOGRAM_EDGES[:-1] + HISTOGRAM_EDGES[1:])
    for key in evaluation.files:
        if not key.startswith("pnl__"):
            continue
        _, strategy, cost = key.split("__")
        values = np.clip(evaluation[key] / premium, HISTOGRAM_EDGES[0] + 1e-9, HISTOGRAM_EDGES[-1] - 1e-9)
        counts, _ = np.histogram(values, bins=HISTOGRAM_EDGES)
        density = counts / counts.sum()
        rows.extend({"simulator": simulator_name, "cost_bps": float(cost), "strategy": strategy, "bin_center": center, "probability": share} for center, share in zip(centers, density))
    return rows


def sample_path_arrays(evaluation: dict, simulator_name: str) -> dict[str, np.ndarray]:
    arrays = {f"{simulator_name}__prices": evaluation["sample_prices"], f"{simulator_name}__volatility": evaluation["sample_volatility"]}
    for key in evaluation.files:
        if key.startswith("positions__"):
            _, strategy, cost = key.split("__")
            arrays[f"{simulator_name}__positions__{strategy}__{cost}"] = evaluation[key]
    return arrays


def real_window_rows(real: dict, label: str) -> list[dict]:
    rows = []
    dates = real[f"{label}__start_dates"]
    for key in real.files:
        if key.startswith(f"{label}__pnl__"):
            _, _, strategy, cost = key.split("__")
            rows.extend({"windows": label, "start_date": date, "cost_bps": float(cost), "strategy": strategy, "pnl_x_premium": float(value)} for date, value in zip(dates, real[key]))
    return rows


def export_dashboard_data(config: Config) -> Path:
    processed = config.paths.resolve("processed")
    metrics_directory = config.paths.resolve("metrics")
    output = processed / DASHBOARD_DIRECTORY_NAME
    output.mkdir(parents=True, exist_ok=True)
    with open(metrics_directory / "test_metrics.json") as handle:
        test_summary = json.load(handle)
    premium = test_summary["premium"]
    histogram_records: list[dict] = []
    path_arrays: dict[str, np.ndarray] = {}
    for simulator_name in config.evaluation.simulators:
        evaluation = np.load(processed / f"evaluation_{simulator_name}.npz")
        histogram_records.extend(histogram_rows(evaluation, simulator_name, premium))
        path_arrays.update(sample_path_arrays(evaluation, simulator_name))
        path_arrays["times"] = evaluation["times"]
    pd.DataFrame(histogram_records).to_csv(output / "pnl_histograms.csv", index=False)
    np.savez_compressed(output / "sample_paths.npz", **path_arrays)
    for name in ["test_metrics.csv", "bootstrap_cvar95_differences.csv", "real_data_metrics.csv", "real_data_bootstrap_cvar95_differences.csv"]:
        shutil.copyfile(metrics_directory / name, output / name)
    real = np.load(processed / "real_data.npz")
    pd.DataFrame(real_window_rows(real, "non_overlapping") + real_window_rows(real, "overlapping")).to_csv(output / "real_data_windows.csv", index=False)
    with open(metrics_directory / "real_data_metrics.json") as handle:
        real_summary = json.load(handle)
    manifest = {
        "premium": premium,
        "sigma0": test_summary["sigma0"],
        "n_test_paths": test_summary["n_test_paths"],
        "test_seed": test_summary["test_seed"],
        "cost_grid": list(config.costs.cost_grid),
        "default_cost_bps": config.costs.cost_bps,
        "simulators": list(config.evaluation.simulators),
        "real_data_windows": real_summary["windows"],
        "maturity_days": config.option.maturity_days,
        "rate": config.option.r,
    }
    with open(output / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return output
