# Build plan: assumptions and deviations

This file records the decisions taken where the spec leaves room, so every later result can be traced back to a stated choice.

## Environment
- Python 3.13 in a project-local `.venv` (spec asks for 3.10+). Dependencies pinned loosely in `requirements.txt`.
- Device selection is `device: auto` in config: CUDA, then MPS, then CPU. Any value can be forced from config. Validation scripts default to CPU float64 where exactness matters (accounting checks); training runs in float32.

## Problem setup
- Prices are normalized so S0 = 1 and K = 1 everywhere inside models. Reporting converts to percent of premium or percent of notional.
- Time grid is an explicit array of times in years. The daily grid is `k / 252` for k = 0..21. The hedging environment accepts any increasing grid ending at maturity T = 21/252.
- Accounting (per path):
  - cash_0 = C0 - h_0 S_0 - c S_0 |h_0 - 0|
  - cash_{k} = cash_{k-1} e^{r dt_k} - (h_k - h_{k-1}) S_k - c S_k |h_k - h_{k-1}| for k = 1..N-1
  - cash_N = cash_{N-1} e^{r dt_N}
  - P&L = cash_N + h_{N-1} S_N - c S_N |h_{N-1}| - max(S_N - K, 0)
  - Loss L = -P&L. P&L is a time-T amount and is not discounted.
- Exact CVaR uses the mean of the worst ceil(n (1 - alpha)) losses.

## Simulators
- GBM: drift `mu` from config. Validation experiments use mu = r so that the delta-hedge P&L has mean close to zero.
- GJR-GARCH(1,1)-t is fitted with `arch` on daily percentage log returns of Nifty with a constant mean, using only data before 2020-01-01. Simulation runs under the fitted (real-world) drift. Each path starts after a burn-in of 250 simulated days from the unconditional variance, so starting volatility varies across paths like it does in history.
- Conditional volatility fed to policies is the one-step-ahead GARCH volatility, annualized with sqrt(252).
- Heston: full-truncation Euler with 4 sub-steps per trading day, theta and v0 set to the pre-2020 Nifty realized variance, kappa = 3, xi = 0.4, rho = -0.7, drift = r. These are illustrative and not calibrated to options. The volatility feature for Heston is sqrt(v_t) (the simulator's own state).
- sigma0 (used for the premium C0) is `auto` in config, resolved to the fitted GARCH unconditional annualized volatility saved by script 02. Phase 1 checks use explicit sigmas and do not depend on it.

## Policies and training
- Feature scaling: ln(S/K) is divided by sigma0 sqrt(T); sigma_t is divided by sigma0; t/T, h_prev and bs_delta are passed as is. bs_delta uses the per-step conditional volatility and remaining time to maturity.
- No-transaction band: reference = clamp(bs_delta + correction, 0, 1); band = [reference - softplus(a), reference + softplus(b)]; h = clamp(h_prev, lower, upper), then clamped to [0, 1].
- An epoch is 50,000 fresh paths, processed in minibatches (default 5,000). Validation set and test set are fixed by seed and shared by every strategy so comparisons are paired.
- The Rockafellar-Uryasev w is a learnable scalar trained jointly with the network.

## Baselines
- BS delta uses the same per-step conditional volatility the policies see (constant sigma under GBM).
- The no-trade-band delta hedge uses a symmetric band of half-width b around BS delta; b is chosen per cost level by grid search on the validation paths (minimum exact CVaR95).

## Checkpoint A
- 84 fine steps over 21 days; rebalance counts {1, 2, 4, 7, 21, 84} all divide 84, so each coarser grid is an exact subsample of the same paths.

## Deviations from the spec
- None intended. Anything that turns out to need a deviation will be added here with the reason.
