from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def tail_count(n_samples: int, alpha: float) -> int:
    return max(1, int(math.ceil(n_samples * (1.0 - alpha) - 1e-9)))


def exact_cvar(losses: torch.Tensor, alpha: float) -> torch.Tensor:
    count = tail_count(losses.shape[0], alpha)
    worst, _ = torch.topk(losses, count)
    return worst.mean()


def exact_cvar_np(losses: np.ndarray, alpha: float) -> float:
    count = tail_count(losses.shape[0], alpha)
    return float(np.sort(losses)[-count:].mean())


def value_at_risk_np(losses: np.ndarray, alpha: float) -> float:
    count = tail_count(losses.shape[0], alpha)
    return float(np.sort(losses)[-count])


def rockafellar_uryasev_cvar(losses: torch.Tensor, threshold: torch.Tensor, alpha: float) -> torch.Tensor:
    return threshold + torch.relu(losses - threshold).mean() / (1.0 - alpha)


class CVaRLoss(nn.Module):
    def __init__(self, alpha: float, initial_threshold: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.threshold = nn.Parameter(torch.tensor(float(initial_threshold)))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        return rockafellar_uryasev_cvar(losses, self.threshold.to(losses.dtype), self.alpha)
