# Build plan: assumptions and deviations

This file records the decisions taken where the spec leaves room, so every later result can be traced back to a stated choice.

## Environment
- Python 3.13 in a project-local `.venv` (spec asks for 3.10+). Dependencies pinned loosely in `requirements.txt`.
- Device selection is `device: auto` in config: CUDA if available, otherwise CPU. Any device, including `mps`, can be forced from config. Validation scripts use CPU float64 where exactness matters (accounting checks); training runs in float32.

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
- Feature scaling: ln(S/K) is divided by sigma0 sqrt(T) and clipped to [-5, 5]; the volatility input is ln(sigma_t / sigma0) clipped to [-2, 2]; t/T, h_prev and bs_delta are passed as is. bs_delta uses the per-step conditional volatility and remaining time to maturity.
- Both networks also receive the running mark-to-market P&L, (cash + h_prev S - BS call value) / premium, clipped to [-5, 5]. This is an addition to the spec's MLP input list, agreed with the user after Checkpoint B, because CVaR-optimal hedges depend on how far a path already is into the loss tail.
- No-transaction band: reference = clamp(bs_delta + correction, 0, 1); band = [reference - softplus(a), reference + softplus(b)]; h = clamp(h_prev, lower, upper), then clamped to [0, 1].
- An epoch is 50,000 fresh paths, processed in minibatches of 25,000 (raised from 5,000 after a validation comparison at 5 bps; with 5,000 only about 250 tail paths drive each CVaR gradient). Validation set and test set are fixed by seed and shared by every strategy so comparisons are paired.
- The Rockafellar-Uryasev w is a learnable scalar trained jointly with the network.

## Baselines
- BS delta uses the same per-step conditional volatility the policies see (constant sigma under GBM).
- The no-trade-band delta hedge uses a symmetric band of half-width b around BS delta; b is chosen per cost level by grid search on the validation paths (minimum exact CVaR95).

## Checkpoint A
- 84 fine steps over 21 days; rebalance counts {1, 2, 4, 7, 21, 84} all divide 84, so each coarser grid is an exact subsample of the same paths.

## Deviations from the spec
- `device: auto` skips MPS. Measured on this Mac (M-series, torch 2.14): one 50,000-path training epoch takes about 0.7 s on CPU against 1.4 s on MPS for the MLP and 0.6 s against 0.9 s for the band network, because the 21-step recurrent loop is dominated by kernel launch overhead. MPS is still selectable with `device: mps`.
- The GARCH check against `arch` compares mean, std, quantiles and ACFs of absolute and squared returns, plus an exact check that the custom variance recursion reproduces `arch`'s own simulated volatility path. Kurtosis is reported but not used as a pass criterion: for the fitted parameters E[(alpha + gamma 1[z<0]) z^2 + beta]^2 is about 1.015, so the unconditional fourth moment is infinite and sample kurtosis does not converge.
- `losses.py` and the GBM generator were created during Phase 1 because the baselines and the Checkpoint A experiment need them.
- Input clipping was added after the first Phase 4 evaluation. Inspection showed that in rare extreme GARCH states (simulated volatility above roughly 70 percent, about 0.7 percent of states) the unclipped networks extrapolated to a full hedge h = 1 while the index collapsed, producing worst-case losses of 24 to 32 premiums against 11 to 14 for BS delta. The flaw was first seen on the test set and then confirmed on the validation set, which is what the fix was judged on. The fix is a generic input bound, not a tuned parameter.
- A tail-aware band network (`ntb_tail`) trained on CVaR95 + 0.5 x CVaR99 was added after Phase 4 at the user's request, alongside (not replacing) the CVaR95 models the spec defines. The weight 0.5 was fixed before training. Early stopping for this variant uses the same combined objective on the validation set.
