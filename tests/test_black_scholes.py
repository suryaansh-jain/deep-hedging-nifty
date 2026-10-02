import numpy as np
import pytest
import torch

from black_scholes import (
    call_delta_np,
    call_delta_torch,
    call_gamma_np,
    call_gamma_torch,
    call_price_np,
    call_price_torch,
    call_vega_np,
    call_vega_torch,
    put_price_np,
)


def test_reference_call_price() -> None:
    price = call_price_np(np.array(100.0), 100.0, 1.0, 0.05, 0.2)
    assert abs(float(price) - 10.4506) < 1e-3


def test_torch_matches_numpy() -> None:
    spots = np.linspace(0.8, 1.2, 9)
    taus = np.linspace(0.01, 0.5, 9)
    sigmas = np.linspace(0.1, 0.4, 9)
    spot_t = torch.tensor(spots, dtype=torch.float64)
    tau_t = torch.tensor(taus, dtype=torch.float64)
    sigma_t = torch.tensor(sigmas, dtype=torch.float64)
    pairs = [
        (call_price_np, call_price_torch),
        (call_delta_np, call_delta_torch),
        (call_gamma_np, call_gamma_torch),
        (call_vega_np, call_vega_torch),
    ]
    for numpy_function, torch_function in pairs:
        expected = numpy_function(spots, 1.0, taus, 0.065, sigmas)
        actual = torch_function(spot_t, 1.0, tau_t, 0.065, sigma_t).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("spot,strike,tau,rate,sigma", [(100.0, 100.0, 1.0, 0.05, 0.2), (1.0, 1.1, 0.1, 0.065, 0.18), (1.2, 1.0, 21 / 252, 0.0, 0.3)])
def test_put_call_parity(spot: float, strike: float, tau: float, rate: float, sigma: float) -> None:
    call = call_price_np(np.array(spot), strike, tau, rate, sigma)
    put = put_price_np(np.array(spot), strike, tau, rate, sigma)
    assert abs(float(call - put) - (spot - strike * np.exp(-rate * tau))) < 1e-12


def test_delta_gamma_vega_match_finite_differences() -> None:
    spots = np.linspace(0.85, 1.15, 13)
    tau, rate, sigma, bump = 21 / 252, 0.065, 0.18, 1e-4
    up = call_price_np(spots + bump, 1.0, tau, rate, sigma)
    down = call_price_np(spots - bump, 1.0, tau, rate, sigma)
    center = call_price_np(spots, 1.0, tau, rate, sigma)
    np.testing.assert_allclose(call_delta_np(spots, 1.0, tau, rate, sigma), (up - down) / (2 * bump), atol=1e-6)
    np.testing.assert_allclose(call_gamma_np(spots, 1.0, tau, rate, sigma), (up - 2 * center + down) / bump**2, rtol=1e-3, atol=1e-3)
    vol_up = call_price_np(spots, 1.0, tau, rate, sigma + bump)
    vol_down = call_price_np(spots, 1.0, tau, rate, sigma - bump)
    np.testing.assert_allclose(call_vega_np(spots, 1.0, tau, rate, sigma), (vol_up - vol_down) / (2 * bump), atol=1e-7)


def test_torch_delta_matches_autograd() -> None:
    spot = torch.linspace(0.9, 1.1, 7, dtype=torch.float64, requires_grad=True)
    price = call_price_torch(spot, 1.0, 0.05, 0.065, 0.2)
    (gradient,) = torch.autograd.grad(price.sum(), spot)
    np.testing.assert_allclose(gradient.numpy(), call_delta_torch(spot.detach(), 1.0, 0.05, 0.065, 0.2).numpy(), atol=1e-10)
