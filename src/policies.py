from __future__ import annotations

import math

import torch
from torch import nn

from baselines import bs_delta_for_state
from black_scholes import call_price_torch
from hedging_env import HedgeState

FEATURE_CLIP = 5.0
LOG_VOLATILITY_CLIP = 2.0


def log_moneyness_scaled(state: HedgeState, sigma0: float) -> torch.Tensor:
    scaled = torch.log(state.spot / state.strike) / (sigma0 * math.sqrt(state.maturity))
    return torch.clamp(scaled, -FEATURE_CLIP, FEATURE_CLIP)


def volatility_scaled(state: HedgeState, sigma0: float) -> torch.Tensor:
    return torch.clamp(torch.log(torch.clamp(state.volatility, min=1e-6) / sigma0), -LOG_VOLATILITY_CLIP, LOG_VOLATILITY_CLIP)


def time_fraction(state: HedgeState) -> torch.Tensor:
    return torch.full_like(state.spot, state.time / state.maturity)


def running_pnl_scaled(state: HedgeState) -> torch.Tensor:
    option_value = call_price_torch(state.spot, state.strike, state.time_to_maturity, state.rate, state.volatility)
    return torch.clamp((state.wealth - option_value) / state.premium, -FEATURE_CLIP, FEATURE_CLIP)


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
    base_feature_names = ("t_over_T", "log_moneyness_scaled", "h_prev", "log_sigma_ratio", "bs_delta")

    def __init__(self, sigma0: float, hidden_size: int = 64, hidden_layers: int = 3, include_running_pnl: bool = False) -> None:
        super().__init__()
        self.sigma0 = sigma0
        self.include_running_pnl = include_running_pnl
        self.feature_names = self.base_feature_names + (("running_pnl_scaled",) if include_running_pnl else ())
        self.network = build_mlp(len(self.feature_names), 1, hidden_size, hidden_layers)

    def features(self, state: HedgeState) -> torch.Tensor:
        columns = [
            time_fraction(state),
            log_moneyness_scaled(state, self.sigma0),
            state.h_prev,
            volatility_scaled(state, self.sigma0),
            bs_delta_for_state(state),
        ]
        if self.include_running_pnl:
            columns.append(running_pnl_scaled(state))
        return torch.stack(columns, dim=1)

    def forward(self, state: HedgeState) -> torch.Tensor:
        return torch.sigmoid(self.network(self.features(state)).squeeze(1))


class NoTransactionBandPolicy(nn.Module):
    name = "ntb"
    base_feature_names = ("t_over_T", "log_moneyness_scaled", "log_sigma_ratio", "bs_delta")
    initial_band_logit = -3.0

    def __init__(self, sigma0: float, hidden_size: int = 64, hidden_layers: int = 3, include_running_pnl: bool = False) -> None:
        super().__init__()
        self.sigma0 = sigma0
        self.include_running_pnl = include_running_pnl
        self.feature_names = self.base_feature_names + (("running_pnl_scaled",) if include_running_pnl else ())
        self.network = build_mlp(len(self.feature_names), 3, hidden_size, hidden_layers)
        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            final_layer.bias.copy_(torch.tensor([0.0, self.initial_band_logit, self.initial_band_logit]))

    def features(self, state: HedgeState) -> tuple[torch.Tensor, torch.Tensor]:
        delta = bs_delta_for_state(state)
        columns = [time_fraction(state), log_moneyness_scaled(state, self.sigma0), volatility_scaled(state, self.sigma0), delta]
        if self.include_running_pnl:
            columns.append(running_pnl_scaled(state))
        return torch.stack(columns, dim=1), delta

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


def build_policy(name: str, sigma0: float, hidden_size: int, hidden_layers: int, include_running_pnl: bool = False) -> nn.Module:
    return POLICY_CLASSES[name](sigma0, hidden_size, hidden_layers, include_running_pnl)
