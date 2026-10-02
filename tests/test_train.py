import numpy as np
import torch

from black_scholes import atm_call_premium
from policies import build_policy
from simulators import GBMSimulator
from train import TrainingSettings, train_policy


def small_settings(premium: float) -> TrainingSettings:
    return TrainingSettings(
        alpha=0.95,
        premium=premium,
        rate=0.065,
        cost_rate=0.0005,
        n_train_paths=2000,
        batch_size=500,
        max_epochs=6,
        learning_rate=0.005,
        weight_decay=0.0,
        grad_clip=1.0,
        patience=10,
        lr_factor=0.5,
        lr_patience=3,
        min_lr=1e-5,
        train_seed_base=500,
    )


def train_once(policy_name: str) -> tuple[float, list[float]]:
    torch.manual_seed(0)
    simulator = GBMSimulator(0.2, 0.065, 21 / 252, 21)
    validation = simulator.generate(2000, 1)
    premium = atm_call_premium(0.065, 0.2, 21 / 252)
    policy = build_policy(policy_name, 0.2, 16, 2)
    result = train_policy(policy, simulator, validation, small_settings(premium), torch.device("cpu"))
    return result.best_val_cvar, result.history["val_cvar"].tolist()


def test_training_is_reproducible_and_improves() -> None:
    first_best, first_curve = train_once("mlp")
    second_best, second_curve = train_once("mlp")
    assert first_best == second_best
    np.testing.assert_allclose(first_curve, second_curve)
    assert first_best == min(first_curve)
    assert min(first_curve) < first_curve[0]


def test_band_policy_trains() -> None:
    best, curve = train_once("ntb")
    assert np.isfinite(best)
    assert best <= curve[0]
