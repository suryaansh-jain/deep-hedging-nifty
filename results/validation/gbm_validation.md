# Checkpoint B: GBM validation of the MLP policy

Generated 2026-10-02 19:50 UTC by `src/check_gbm_validation.py` from `/Users/suryaansh/projects/deep-hedging-nifty/config/default.yaml`.

## Setup

- Simulator: GBM, sigma = 0.18, drift mu = 0.065, r = 0.065, 21 daily steps, S0 = K = 1.
- Short one ATM call, premium C0 = 0.023485 (BS price at sigma = 0.18). Transaction cost c = 0.
- MLPPolicy trained with CVaR95 (Rockafellar-Uryasev), 50000 fresh paths per epoch, batch 25000, Adam lr 0.001, early stopping on 20000 fixed validation paths (seed 20001).
- Best epoch 321 of 347 run, training time 181 s on cpu.
- Test set: 20000 fresh GBM paths (seed 40001), never used in training or model selection.

## Hedge ratio versus Black-Scholes delta

Mean absolute difference over all 420000 (path, step) pairs: **0.0451** (target below 0.05).

Median 0.0334, 95th percentile 0.1288, max 0.3389.

### By time to expiry (trading days)

| days_to_expiry | count | mad | mean_learned_minus_delta |
|---|---|---|---|
| 15-21 | 140000 | 0.0211 | -0.0037 |
| 8-14 | 140000 | 0.0449 | -0.0013 |
| 4-7 | 80000 | 0.0623 | -0.0010 |
| 2-3 | 40000 | 0.0746 | -0.0032 |
| 1 | 20000 | 0.0858 | -0.0056 |

### By moneyness ln(S/K)

| ln_S_over_K | count | mad | mean_learned_minus_delta |
|---|---|---|---|
| [-inf, -0.06) | 19302 | 0.0202 | 0.0201 |
| [-0.06, -0.03) | 49480 | 0.0735 | 0.0728 |
| [-0.03, -0.01) | 73053 | 0.0570 | 0.0557 |
| [-0.01, 0.01) | 122888 | 0.0170 | -0.0043 |
| [0.01, 0.03) | 76320 | 0.0609 | -0.0598 |
| [0.03, 0.06) | 55386 | 0.0652 | -0.0648 |
| [0.06, inf) | 23571 | 0.0170 | -0.0169 |

Plot: `results/validation/gbm_hedge_vs_delta.png`.

## P&L on the same test paths

| strategy | mean_pnl | std_pnl | cvar95_loss | mean_pnl_x_premium | std_pnl_x_premium | cvar95_loss_x_premium |
|---|---|---|---|---|---|---|
| MLP policy | -0.0000 | 0.0050 | 0.0076 | -0.0020 | 0.2135 | 0.3219 |
| BS delta | -0.0000 | 0.0039 | 0.0092 | -0.0013 | 0.1643 | 0.3900 |

## Verdict

**PASS**: mean absolute difference 0.0451 < 0.05.
