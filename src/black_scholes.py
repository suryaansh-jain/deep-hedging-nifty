from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm

MIN_TIME = 1e-12
MIN_VOL = 1e-8


def d1_d2_np(spot: np.ndarray, strike: np.ndarray | float, tau: np.ndarray | float, rate: float, sigma: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    tau_safe = np.maximum(tau, MIN_TIME)
    vol_sqrt_tau = np.maximum(sigma, MIN_VOL) * np.sqrt(tau_safe)
    d1 = (np.log(spot / strike) + (rate + 0.5 * np.square(sigma)) * tau_safe) / vol_sqrt_tau
    return d1, d1 - vol_sqrt_tau


def call_price_np(spot: np.ndarray, strike: np.ndarray | float, tau: np.ndarray | float, rate: float, sigma: np.ndarray | float) -> np.ndarray:
    d1, d2 = d1_d2_np(spot, strike, tau, rate, sigma)
    price = spot * norm.cdf(d1) - strike * np.exp(-rate * np.asarray(tau)) * norm.cdf(d2)
    return np.where(np.asarray(tau) <= 0, np.maximum(spot - strike, 0.0), price)


def put_price_np(spot: np.ndarray, strike: np.ndarray | float, tau: np.ndarray | float, rate: float, sigma: np.ndarray | float) -> np.ndarray:
    return call_price_np(spot, strike, tau, rate, sigma) - spot + strike * np.exp(-rate * np.asarray(tau))


def call_delta_np(spot: np.ndarray, strike: np.ndarray | float, tau: np.ndarray | float, rate: float, sigma: np.ndarray | float) -> np.ndarray:
    d1, _ = d1_d2_np(spot, strike, tau, rate, sigma)
    return norm.cdf(d1)


def call_gamma_np(spot: np.ndarray, strike: np.ndarray | float, tau: np.ndarray | float, rate: float, sigma: np.ndarray | float) -> np.ndarray:
    d1, _ = d1_d2_np(spot, strike, tau, rate, sigma)
    return norm.pdf(d1) / (spot * np.maximum(sigma, MIN_VOL) * np.sqrt(np.maximum(tau, MIN_TIME)))


def call_vega_np(spot: np.ndarray, strike: np.ndarray | float, tau: np.ndarray | float, rate: float, sigma: np.ndarray | float) -> np.ndarray:
    d1, _ = d1_d2_np(spot, strike, tau, rate, sigma)
    return spot * norm.pdf(d1) * np.sqrt(np.maximum(tau, MIN_TIME))


def normal_cdf_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def normal_pdf_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def as_tensor_like(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def d1_d2_torch(spot: torch.Tensor, strike: torch.Tensor | float, tau: torch.Tensor | float, rate: float, sigma: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    tau_safe = torch.clamp(as_tensor_like(tau, spot), min=MIN_TIME)
    sigma_safe = torch.clamp(as_tensor_like(sigma, spot), min=MIN_VOL)
    vol_sqrt_tau = sigma_safe * torch.sqrt(tau_safe)
    d1 = (torch.log(spot / as_tensor_like(strike, spot)) + (rate + 0.5 * sigma_safe * sigma_safe) * tau_safe) / vol_sqrt_tau
    return d1, d1 - vol_sqrt_tau


def call_price_torch(spot: torch.Tensor, strike: torch.Tensor | float, tau: torch.Tensor | float, rate: float, sigma: torch.Tensor | float) -> torch.Tensor:
    d1, d2 = d1_d2_torch(spot, strike, tau, rate, sigma)
    tau_tensor = as_tensor_like(tau, spot)
    strike_tensor = as_tensor_like(strike, spot)
    return spot * normal_cdf_torch(d1) - strike_tensor * torch.exp(-rate * tau_tensor) * normal_cdf_torch(d2)


def call_delta_torch(spot: torch.Tensor, strike: torch.Tensor | float, tau: torch.Tensor | float, rate: float, sigma: torch.Tensor | float) -> torch.Tensor:
    d1, _ = d1_d2_torch(spot, strike, tau, rate, sigma)
    return normal_cdf_torch(d1)


def call_gamma_torch(spot: torch.Tensor, strike: torch.Tensor | float, tau: torch.Tensor | float, rate: float, sigma: torch.Tensor | float) -> torch.Tensor:
    d1, _ = d1_d2_torch(spot, strike, tau, rate, sigma)
    tau_safe = torch.clamp(as_tensor_like(tau, spot), min=MIN_TIME)
    sigma_safe = torch.clamp(as_tensor_like(sigma, spot), min=MIN_VOL)
    return normal_pdf_torch(d1) / (spot * sigma_safe * torch.sqrt(tau_safe))


def call_vega_torch(spot: torch.Tensor, strike: torch.Tensor | float, tau: torch.Tensor | float, rate: float, sigma: torch.Tensor | float) -> torch.Tensor:
    d1, _ = d1_d2_torch(spot, strike, tau, rate, sigma)
    tau_safe = torch.clamp(as_tensor_like(tau, spot), min=MIN_TIME)
    return spot * normal_pdf_torch(d1) * torch.sqrt(tau_safe)


def atm_call_premium(rate: float, sigma: float, maturity: float) -> float:
    return float(call_price_np(np.array(1.0), 1.0, maturity, rate, sigma))
