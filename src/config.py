from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class PathsConfig:
    raw: str
    processed: str
    models: str
    metrics: str
    figures: str
    validation: str

    def resolve(self, name: str) -> Path:
        directory = PROJECT_ROOT / getattr(self, name)
        directory.mkdir(parents=True, exist_ok=True)
        return directory


@dataclass
class DataConfig:
    nifty_ticker: str
    vix_ticker: str
    start_date: str
    end_date: str | None
    test_start_date: str


@dataclass
class OptionConfig:
    maturity_days: int
    trading_days_per_year: int
    r: float
    sigma0: float | str

    @property
    def dt(self) -> float:
        return 1.0 / self.trading_days_per_year

    @property
    def maturity(self) -> float:
        return self.maturity_days / self.trading_days_per_year


@dataclass
class CostsConfig:
    cost_bps: float
    cost_grid: list[float]


@dataclass
class RiskConfig:
    alpha: float


@dataclass
class SimulationConfig:
    n_train_paths: int
    n_val_paths: int
    n_test_paths: int
    val_seed: int
    test_seed: int
    train_seed_base: int


@dataclass
class GbmConfig:
    sigma: float
    mu: float


@dataclass
class GarchConfig:
    burn_in: int


@dataclass
class HestonConfig:
    kappa: float
    xi: float
    rho: float
    theta: float | str
    v0: float | str
    substeps_per_day: int
    mu: float


@dataclass
class BaselinesConfig:
    band_grid: list[float]


@dataclass
class PolicyConfig:
    hidden_size: int
    hidden_layers: int
    include_running_pnl: bool


@dataclass
class TrainingConfig:
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    patience: int
    lr_factor: float
    lr_patience: int
    min_lr: float
    policies: list[str]
    heston_policy: str


@dataclass
class RebalanceValidationConfig:
    n_paths: int
    fine_steps: int
    rebalance_counts: list[int]
    sigma: float
    seed: int
    tolerance: float


@dataclass
class GbmValidationConfig:
    n_test_paths: int
    test_seed: int
    target_mad: float


@dataclass
class ValidationConfig:
    rebalance: RebalanceValidationConfig
    gbm: GbmValidationConfig


@dataclass
class EvaluationConfig:
    simulators: list[str]
    n_bootstrap: int
    bootstrap_seed: int
    sample_paths_saved: int


@dataclass
class RealDataConfig:
    simulator_models: str
    n_bootstrap: int
    bootstrap_seed: int


@dataclass
class Config:
    seed: int
    device: str
    paths: PathsConfig
    data: DataConfig
    option: OptionConfig
    costs: CostsConfig
    risk: RiskConfig
    simulation: SimulationConfig
    gbm: GbmConfig
    garch: GarchConfig
    heston: HestonConfig
    baselines: BaselinesConfig
    policy: PolicyConfig
    training: TrainingConfig
    validation: ValidationConfig
    evaluation: EvaluationConfig
    real_data: RealDataConfig
    source_path: str = field(default="")


def build_dataclass(cls: type, values: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        nested_type = item.type if isinstance(item.type, type) else globals().get(str(item.type))
        if nested_type is not None and is_dataclass(nested_type) and isinstance(value, dict):
            kwargs[item.name] = build_dataclass(nested_type, value)
        else:
            kwargs[item.name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path) as handle:
        values = yaml.safe_load(handle)
    config = build_dataclass(Config, values)
    config.source_path = str(config_path)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fitted_params_path(config: Config) -> Path:
    return config.paths.resolve("processed") / "simulator_params.json"


def load_fitted_params(config: Config) -> dict[str, Any]:
    path = fitted_params_path(config)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/run_02_fit_simulators.py first.")
    with open(path) as handle:
        return json.load(handle)


def resolve_sigma0(config: Config) -> float:
    if isinstance(config.option.sigma0, (int, float)):
        return float(config.option.sigma0)
    return float(load_fitted_params(config)["garch"]["unconditional_annual_vol"])


def time_grid(maturity: float, n_steps: int) -> np.ndarray:
    return np.linspace(0.0, maturity, n_steps + 1)


def cost_rate_from_bps(cost_bps: float) -> float:
    return cost_bps / 10000.0
