import numpy as np
import torch

from losses import CVaRLoss, exact_cvar, exact_cvar_np, rockafellar_uryasev_cvar, tail_count, value_at_risk_np


def test_tail_count() -> None:
    assert tail_count(100, 0.95) == 5
    assert tail_count(20000, 0.95) == 1000
    assert tail_count(10, 0.99) == 1


def test_exact_cvar_simple() -> None:
    losses = torch.arange(1, 101, dtype=torch.float64)
    assert float(exact_cvar(losses, 0.95)) == float(np.mean([96, 97, 98, 99, 100]))
    assert exact_cvar_np(losses.numpy(), 0.95) == 98.0


def test_rockafellar_uryasev_matches_exact_at_var() -> None:
    rng = np.random.default_rng(0)
    losses = rng.standard_t(4, size=20000)
    threshold = value_at_risk_np(losses, 0.95)
    tensor = torch.tensor(losses)
    ru_value = float(rockafellar_uryasev_cvar(tensor, torch.tensor(threshold, dtype=torch.float64), 0.95))
    assert abs(ru_value - exact_cvar_np(losses, 0.95)) < 1e-10


def test_rockafellar_uryasev_minimized_by_learning_threshold() -> None:
    rng = np.random.default_rng(1)
    losses = torch.tensor(rng.normal(0.0, 1.0, size=50000))
    loss_module = CVaRLoss(0.95)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=0.05)
    for _ in range(600):
        optimizer.zero_grad()
        value = loss_module(losses)
        value.backward()
        optimizer.step()
    exact = float(exact_cvar(losses, 0.95))
    with torch.no_grad():
        learned = float(loss_module(losses))
    assert abs(learned - exact) < 2e-3
    assert learned >= exact - 1e-9


def test_tail_weighted_loss_matches_exact_at_var_thresholds() -> None:
    from losses import TailWeightedCVaRLoss, exact_tail_weighted_cvar

    losses = torch.tensor(np.random.default_rng(3).standard_t(4, size=20000))
    module = TailWeightedCVaRLoss(0.95, 0.99, 0.5, value_at_risk_np(losses.numpy(), 0.95), value_at_risk_np(losses.numpy(), 0.99))
    with torch.no_grad():
        value = float(module(losses))
    assert abs(value - float(exact_tail_weighted_cvar(losses, 0.95, 0.99, 0.5))) < 1e-9
    assert len(list(module.parameters())) == 2
