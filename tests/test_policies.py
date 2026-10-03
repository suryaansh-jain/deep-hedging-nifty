import numpy as np
import torch

from hedging_env import HedgeState, run_hedge
from policies import MLPPolicy, NoTransactionBandPolicy, build_policy
from simulators import GBMSimulator


def make_state(n_paths: int = 64, h_prev: torch.Tensor | None = None) -> HedgeState:
    generator = torch.Generator().manual_seed(0)
    spot = torch.exp(0.05 * torch.randn(n_paths, generator=generator))
    return HedgeState(
        step=3,
        time=3 / 252,
        time_to_maturity=18 / 252,
        maturity=21 / 252,
        spot=spot,
        h_prev=torch.rand(n_paths, generator=generator) if h_prev is None else h_prev,
        volatility=torch.full((n_paths,), 0.2),
        strike=1.0,
        rate=0.065,
        wealth=torch.full((n_paths,), 0.02),
        premium=torch.full((n_paths,), 0.02),
    )


def test_mlp_output_in_unit_interval_and_uses_h_prev() -> None:
    torch.manual_seed(0)
    policy = MLPPolicy(0.2)
    state = make_state()
    output = policy(state)
    assert output.shape == (64,)
    assert torch.all((output >= 0) & (output <= 1))
    assert policy.features(state).shape == (64, 5)
    changed = make_state(h_prev=torch.ones(64))
    assert not torch.allclose(policy(changed), policy(make_state(h_prev=torch.zeros(64))))


def test_band_policy_stays_inside_band_and_keeps_h_prev_inside() -> None:
    torch.manual_seed(1)
    policy = NoTransactionBandPolicy(0.2)
    for layer in policy.network:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, std=0.3)
    state = make_state()
    lower, upper = policy.band(state)
    output = policy(state)
    assert torch.all(lower <= upper)
    assert torch.all((output >= lower - 1e-7) & (output <= upper + 1e-7))
    inside = (state.h_prev >= lower) & (state.h_prev <= upper)
    assert torch.equal(output[inside], state.h_prev[inside])
    assert torch.all((output >= 0) & (output <= 1))


def test_band_policy_starts_near_delta_with_narrow_band() -> None:
    policy = NoTransactionBandPolicy(0.2)
    state = make_state(h_prev=torch.zeros(64))
    lower, upper = policy.band(state)
    width = upper - lower
    assert float(width.detach().max()) < 0.11


def test_gradients_flow_through_recurrent_rollout() -> None:
    paths = GBMSimulator(0.2, 0.065, 21 / 252, 21).generate(256, 2)
    prices, volatility = paths.to_torch()
    for name in ["mlp", "ntb"]:
        torch.manual_seed(0)
        policy = build_policy(name, 0.2, 16, 2)
        outcome = run_hedge(prices, paths.times, policy, 0.02, 1.0, 0.065, 0.001, volatility)
        outcome.loss.mean().backward()
        gradient_norm = sum(float(parameter.grad.norm()) for parameter in policy.parameters() if parameter.grad is not None)
        assert np.isfinite(gradient_norm) and gradient_norm > 0


def test_running_pnl_feature_is_optional_and_tracks_wealth() -> None:
    torch.manual_seed(0)
    plain = MLPPolicy(0.2)
    extended = MLPPolicy(0.2, include_running_pnl=True)
    state = make_state()
    assert plain.features(state).shape == (64, 5)
    features = extended.features(state)
    assert features.shape == (64, 6)
    richer = make_state()
    richer.wealth = richer.wealth + 0.01
    unclipped = (features[:, 5].abs() < 4.0) & (extended.features(richer)[:, 5].abs() < 4.0)
    shift = extended.features(richer)[:, 5] - features[:, 5]
    assert unclipped.sum() > 50
    assert torch.allclose(shift[unclipped], torch.full((int(unclipped.sum()),), 0.5))
    band = NoTransactionBandPolicy(0.2, include_running_pnl=True)
    assert band.features(state)[0].shape == (64, 5)


def test_initial_wealth_equals_premium() -> None:
    paths = GBMSimulator(0.2, 0.065, 21 / 252, 21).generate(8, 2)
    prices, volatility = paths.to_torch()
    seen: list[torch.Tensor] = []

    def recorder(state: HedgeState) -> torch.Tensor:
        seen.append(state.wealth.clone())
        return torch.full_like(state.spot, 0.5)

    run_hedge(prices, paths.times, recorder, 0.02, 1.0, 0.065, 0.0, volatility)
    assert torch.allclose(seen[0], torch.full((8,), 0.02))
    expected_second = (0.02 - 0.5) * np.exp(0.065 / 252) + 0.5 * prices[:, 1]
    assert torch.allclose(seen[1], expected_second.float(), atol=1e-6)


def test_features_are_bounded_in_extreme_states() -> None:
    policy = MLPPolicy(0.2, include_running_pnl=True)
    state = make_state()
    state.spot = torch.full_like(state.spot, 0.05)
    state.volatility = torch.full_like(state.spot, 5.0)
    state.wealth = torch.full_like(state.spot, -1.0)
    features = policy.features(state)
    assert torch.all(features.abs() <= 5.0)
