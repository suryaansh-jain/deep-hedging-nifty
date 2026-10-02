# Checkpoint B: GBM validation of the MLP policy

Generated 2026-10-02 19:24 UTC by `src/check_gbm_validation.py` from `/Users/suryaansh/projects/deep-hedging-nifty/config/default.yaml`.

## Setup

- Simulator: GBM, sigma = 0.18, drift mu = 0.065, r = 0.065, 21 daily steps, S0 = K = 1.
- Short one ATM call, premium C0 = 0.023485 (BS price at sigma = 0.18). Transaction cost c = 0.
- MLPPolicy trained with CVaR95 (Rockafellar-Uryasev), 50000 fresh paths per epoch, batch 5000, Adam lr 0.001, early stopping on 20000 fixed validation paths (seed 20001).
- Best epoch 157 of 183 run, training time 74 s on cpu.
- Test set: 20000 fresh GBM paths (seed 40001), never used in training or model selection.

## Hedge ratio versus Black-Scholes delta

Mean absolute difference over all 420000 (path, step) pairs: **0.0306** (target below 0.05).

Median 0.0288, 95th percentile 0.0653, max 0.1244.

### By time to expiry (trading days)

| days_to_expiry | count | mad | mean_learned_minus_delta |
|---|---|---|---|
| 15-21 | 140000 | 0.0198 | -0.0029 |
| 8-14 | 140000 | 0.0331 | 0.0005 |
| 4-7 | 80000 | 0.0354 | 0.0007 |
| 2-3 | 40000 | 0.0395 | -0.0026 |
| 1 | 20000 | 0.0503 | -0.0035 |

### By moneyness ln(S/K)

| ln_S_over_K | count | mad | mean_learned_minus_delta |
|---|---|---|---|
| [-inf, -0.06) | 19302 | 0.0139 | 0.0139 |
| [-0.06, -0.03) | 49480 | 0.0462 | 0.0462 |
| [-0.03, -0.01) | 73053 | 0.0414 | 0.0414 |
| [-0.01, 0.01) | 122888 | 0.0120 | -0.0021 |
| [0.01, 0.03) | 76320 | 0.0419 | -0.0419 |
| [0.03, 0.06) | 55386 | 0.0419 | -0.0419 |
| [0.06, inf) | 23571 | 0.0108 | -0.0108 |

Plot: `results/validation/gbm_hedge_vs_delta.png`.

## P&L on the same test paths

| strategy | mean_pnl | std_pnl | cvar95_loss | mean_pnl_x_premium | std_pnl_x_premium | cvar95_loss_x_premium |
|---|---|---|---|---|---|---|
| MLP policy | -0.0000 | 0.0043 | 0.0081 | -0.0017 | 0.1840 | 0.3466 |
| BS delta | -0.0000 | 0.0039 | 0.0092 | -0.0013 | 0.1643 | 0.3900 |

## Verdict

**PASS**: mean absolute difference 0.0306 < 0.05.
