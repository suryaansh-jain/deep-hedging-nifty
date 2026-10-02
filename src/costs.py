from __future__ import annotations

import torch


def proportional_cost(spot: torch.Tensor, trade: torch.Tensor, cost_rate: float) -> torch.Tensor:
    return cost_rate * spot * torch.abs(trade)


def turnover(positions: torch.Tensor) -> torch.Tensor:
    previous = torch.cat([torch.zeros_like(positions[:, :1]), positions[:, :-1]], dim=1)
    return torch.abs(positions - previous).sum(dim=1) + torch.abs(positions[:, -1])
