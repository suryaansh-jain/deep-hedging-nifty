from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn

from hedging_env import run_hedge
from losses import CVaRLoss, exact_cvar, value_at_risk_np
from config import Config
from simulators import SimulatedPaths


class PathGenerator(Protocol):
    def generate(self, n_paths: int, seed: int) -> SimulatedPaths: ...


@dataclass
class TrainingSettings:
    alpha: float
    premium: float
    rate: float
    cost_rate: float
    n_train_paths: int
    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    patience: int
    lr_factor: float
    lr_patience: int
    min_lr: float
    train_seed_base: int


@dataclass
class TrainingResult:
    policy: nn.Module
    threshold: float
    history: pd.DataFrame
    best_epoch: int
    best_val_cvar: float
    seconds: float


def hedge_losses(policy: nn.Module, tensors: tuple[torch.Tensor, torch.Tensor], times: np.ndarray, settings: TrainingSettings) -> torch.Tensor:
    prices, volatility = tensors
    outcome = run_hedge(prices, times, policy, settings.premium, 1.0, settings.rate, settings.cost_rate, volatility)
    return outcome.loss


def evaluate_cvar(policy: nn.Module, prices: torch.Tensor, volatility: torch.Tensor, times: np.ndarray, settings: TrainingSettings) -> tuple[float, float]:
    policy.eval()
    with torch.no_grad():
        losses = hedge_losses(policy, (prices, volatility), times, settings)
    policy.train()
    return float(exact_cvar(losses, settings.alpha)), float(-losses.mean())


def train_policy(
    policy: nn.Module,
    simulator: PathGenerator,
    validation: SimulatedPaths,
    settings: TrainingSettings,
    device: torch.device,
    log_prefix: str = "",
) -> TrainingResult:
    started = time.time()
    policy = policy.to(device)
    times = validation.times
    val_prices, val_volatility = validation.to_torch(device=device)
    with torch.no_grad():
        initial_losses = hedge_losses(policy, (val_prices, val_volatility), times, settings)
    cvar_loss = CVaRLoss(settings.alpha, value_at_risk_np(initial_losses.cpu().numpy(), settings.alpha)).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(cvar_loss.parameters()), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=settings.lr_factor, patience=settings.lr_patience, min_lr=settings.min_lr
    )
    best_val_cvar, best_epoch = float("inf"), -1
    best_state = copy.deepcopy(policy.state_dict())
    best_threshold = cvar_loss.threshold.item()
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    for epoch in range(settings.max_epochs):
        paths = simulator.generate(settings.n_train_paths, settings.train_seed_base + epoch)
        prices, volatility = paths.to_torch(device=device)
        permutation = torch.randperm(prices.shape[0], device=device)
        batch_losses = []
        for start in range(0, prices.shape[0], settings.batch_size):
            index = permutation[start : start + settings.batch_size]
            losses = hedge_losses(policy, (prices[index], volatility[index]), paths.times, settings)
            objective = cvar_loss(losses)
            optimizer.zero_grad()
            objective.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(cvar_loss.parameters()), settings.grad_clip)
            optimizer.step()
            batch_losses.append(float(objective.detach()))
        val_cvar, val_mean_pnl = evaluate_cvar(policy, val_prices, val_volatility, times, settings)
        scheduler.step(val_cvar)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_ru_cvar": float(np.mean(batch_losses)),
                "val_cvar": val_cvar,
                "val_mean_pnl": val_mean_pnl,
                "threshold": cvar_loss.threshold.item(),
                "learning_rate": learning_rate,
            }
        )
        if val_cvar < best_val_cvar - 1e-7:
            best_val_cvar, best_epoch = val_cvar, epoch
            best_state = copy.deepcopy(policy.state_dict())
            best_threshold = cvar_loss.threshold.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % 10 == 0 or epochs_without_improvement == 0:
            print(
                f"{log_prefix}epoch {epoch:3d} train {history[-1]['train_ru_cvar'] / settings.premium:.4f} "
                f"val {val_cvar / settings.premium:.4f} best {best_val_cvar / settings.premium:.4f} lr {learning_rate:.1e}",
                flush=True,
            )
        if epochs_without_improvement >= settings.patience:
            break
    policy.load_state_dict(best_state)
    policy.eval()
    return TrainingResult(
        policy=policy,
        threshold=best_threshold,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_val_cvar=best_val_cvar,
        seconds=time.time() - started,
    )


def settings_from_config(config: Config, premium: float, cost_rate: float) -> TrainingSettings:
    training = config.training
    return TrainingSettings(
        alpha=config.risk.alpha,
        premium=premium,
        rate=config.option.r,
        cost_rate=cost_rate,
        n_train_paths=config.simulation.n_train_paths,
        batch_size=training.batch_size,
        max_epochs=training.max_epochs,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        grad_clip=training.grad_clip,
        patience=training.patience,
        lr_factor=training.lr_factor,
        lr_patience=training.lr_patience,
        min_lr=training.min_lr,
        train_seed_base=config.simulation.train_seed_base,
    )


def run_name(policy_name: str, simulator_name: str, cost_bps: float) -> str:
    return f"{policy_name}_{simulator_name}_{cost_bps:g}bps"


def save_training_result(result: TrainingResult, name: str, models_directory: Path, curves_directory: Path, metadata: dict) -> None:
    models_directory.mkdir(parents=True, exist_ok=True)
    curves_directory.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: value.detach().cpu() for key, value in result.policy.state_dict().items()}
    torch.save({"state_dict": cpu_state, "threshold": result.threshold, "metadata": metadata}, models_directory / f"{name}.pt")
    result.history.to_csv(curves_directory / f"{name}.csv", index=False)
    summary = {**metadata, "best_epoch": result.best_epoch, "best_val_cvar": result.best_val_cvar, "seconds": result.seconds}
    with open(models_directory / f"{name}.json", "w") as handle:
        json.dump(summary, handle, indent=2)
