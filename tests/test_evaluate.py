import numpy as np

from evaluate import StrategyResult, paired_bootstrap_cvar_difference, resampled_cvar, summarize
from losses import exact_cvar_np


def test_summarize_reports_expected_fields() -> None:
    pnl = np.array([0.01, -0.02, 0.005, -0.001, 0.0])
    result = StrategyResult("x", pnl, np.full(5, 0.001), np.full(5, 2.0), np.zeros((5, 3)))
    summary = summarize(result, premium=0.02)
    assert abs(summary["mean_pnl"] - pnl.mean()) < 1e-15
    assert summary["cvar95"] == 0.02
    assert summary["fraction_loss_above_5pct_premium"] == 0.2
    assert summary["mean_turnover"] == 2.0
    assert summary["n_paths"] == 5


def test_resampled_cvar_identity_indices_matches_exact() -> None:
    losses = np.random.default_rng(0).normal(size=1000)
    indices = np.tile(np.arange(1000), (3, 1))
    np.testing.assert_allclose(resampled_cvar(losses, indices, 0.95), exact_cvar_np(losses, 0.95))


def test_paired_bootstrap_detects_shift_and_brackets_point() -> None:
    rng = np.random.default_rng(1)
    base = rng.standard_t(5, size=20000)
    better = base - 0.3
    interval = paired_bootstrap_cvar_difference(better, base, 0.95, 200, 7)
    assert abs(interval["difference"] + 0.3) < 1e-9
    assert interval["ci_low"] <= interval["difference"] <= interval["ci_high"]
    assert interval["ci_high"] < 0
    same = paired_bootstrap_cvar_difference(base, base, 0.95, 50, 7)
    assert same["ci_low"] == same["ci_high"] == 0.0
