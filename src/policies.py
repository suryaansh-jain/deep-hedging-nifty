from __future__ import annotations

import math

import torch
from torch import nn

from baselines import bs_delta_for_state
from hedging_env import HedgeState


def log_moneyness_scaled(state: HedgeState, sigma0: float) -> torch.Tensor:
    return torch.log(state.spot / state.strike) / (sigma0 * math.sqrt(state.maturity))


def time_fraction(state: HedgeState) -> torch.Tensor:
    return torch.full_like(state.spot, state.time / state.maturity)


def build_mlp(input_size: int, output_size: int, hidden_size: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_size
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(width, hidden_size), nn.ReLU()])
        width = hidden_size
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


class MLPPolicy(nn.Module):
    name = "mlp"
    feature_names = ("t_over_T", "log_moneyness_scaled", "h_prev", "sigma_scaled", "bs_delta")

    def __init__(self, sigma0: float, hidden_size: int = 64, hidden_layers: int = 3) -> None:
        super().__init__()
        self.sigma0 = sigma0
        self.network = build_mlp(len(self.feature_names), 1, hidden_size, hidden_layers)

    def features(self, state: HedgeState) -> torch.Tensor:
        return torch.stack(
            [
                time_fraction(state),
                log_moneyness_scaled(state, self.sigma0),
                state.h_prev,
                state.volatility / self.sigma0,
                bs_delta_for_state(state),
            ],
            dim=1,
        )

    def forward(self, state: HedgeState) -> torch.Tensor:
        return torch.sigmoid(self.network(self.features(state)).squeeze(1))


class NoTransactionBandPolicy(nn.Module):
    name = "ntb"
    feature_names = ("t_over_T", "log_moneyness_scaled", "sigma_scaled", "bs_delta")
    initial_band_logit = -3.0

    def __init__(self, sigma0: float, hidden_size: int = 64, hidden_layers: int = 3) -> None:
        super().__init__()
        self.sigma0 = sigma0
        self.network = build_mlp(len(self.feature_names), 3, hidden_size, hidden_layers)
        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            final_layer.bias.copy_(torch.tensor([0.0, self.initial_band_logit, self.initial_band_logit]))

    def features(self, state: HedgeState) -> tuple[torch.Tensor, torch.Tensor]:
        delta = bs_delta_for_state(state)
        stacked = torch.stack(
            [time_fraction(state), log_moneyness_scaled(state, self.sigma0), state.volatility / self.sigma0, delta],
            dim=1,
        )
        return stacked, delta

    def band(self, state: HedgeState) -> tuple[torch.Tensor, torch.Tensor]:
        stacked, delta = self.features(state)
        output = self.network(stacked)
        reference = torch.clamp(delta + output[:, 0], 0.0, 1.0)
        lower = torch.clamp(reference - nn.functional.softplus(output[:, 1]), 0.0, 1.0)
        upper = torch.clamp(reference + nn.functional.softplus(output[:, 2]), 0.0, 1.0)
        return lower, upper

    def forward(self, state: HedgeState) -> torch.Tensor:
        lower, upper = self.band(state)
        return torch.minimum(torch.maximum(state.h_prev, lower), upper)


POLICY_CLASSES: dict[str, type[nn.Module]] = {"mlp": MLPPolicy, "ntb": NoTransactionBandPolicy}


def build_policy(name: str, sigma0: float, hidden_size: int, hidden_layers: int) -> nn.Module:
    return POLICY_CLASSES[name](sigma0, hidden_size, hidden_layers)
