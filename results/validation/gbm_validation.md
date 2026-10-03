# Checkpoint B: GBM validation of the MLP policy

Generated 2026-10-03 04:41 UTC by `src/check_gbm_validation.py` from `/Users/suryaansh/projects/deep-hedging-nifty/config/default.yaml`.

## Setup

- Simulator: GBM, sigma = 0.18, drift mu = 0.065, r = 0.065, 21 daily steps, S0 = K = 1.
- Short one ATM call, premium C0 = 0.023485 (BS price at sigma = 0.18). Transaction cost c = 0.
- MLPPolicy trained with CVaR95 (Rockafellar-Uryasev), 50000 fresh paths per epoch, batch 25000, Adam lr 0.001, early stopping on 20000 fixed validation paths (seed 20001).
- Best epoch 259 of 285 run, training time 146 s on cpu.
- Test set: 20000 fresh GBM paths (seed 40001), never used in training or model selection.

## Hedge ratio versus Black-Scholes delta

Mean absolute difference over all 420000 (path, step) pairs: **0.0458** (target below 0.05).

Median 0.0349, 95th percentile 0.1296, max 0.3047.

### By time to expiry (trading days)

| days_to_expiry | count | mad | mean_learned_minus_delta |
|---|---|---|---|
| 15-21 | 140000 | 0.0218 | -0.0046 |
| 8-14 | 140000 | 0.0463 | -0.0015 |
| 4-7 | 80000 | 0.0616 | -0.0009 |
| 2-3 | 40000 | 0.0748 | 0.0001 |
| 1 | 20000 | 0.0884 | 0.0009 |

### By moneyness ln(S/K)

| ln_S_over_K | count | mad | mean_learned_minus_delta |
|---|---|---|---|
| [-inf, -0.06) | 19302 | 0.0223 | 0.0223 |
| [-0.06, -0.03) | 49480 | 0.0735 | 0.0730 |
| [-0.03, -0.01) | 73053 | 0.0601 | 0.0591 |
| [-0.01, 0.01) | 122888 | 0.0168 | -0.0054 |
| [0.01, 0.03) | 76320 | 0.0625 | -0.0616 |
| [0.03, 0.06) | 55386 | 0.0645 | -0.0641 |
| [0.06, inf) | 23571 | 0.0150 | -0.0149 |

Plot: `results/validation/gbm_hedge_vs_delta.png`.

## P&L on the same test paths

| strategy | mean_pnl | std_pnl | cvar95_loss | mean_pnl_x_premium | std_pnl_x_premium | cvar95_loss_x_premium |
|---|---|---|---|---|---|---|
| MLP policy | -0.0000 | 0.0051 | 0.0076 | -0.0019 | 0.2150 | 0.3240 |
| BS delta | -0.0000 | 0.0039 | 0.0092 | -0.0013 | 0.1643 | 0.3900 |

## Verdict

**PASS**: mean absolute difference 0.0458 < 0.05.
