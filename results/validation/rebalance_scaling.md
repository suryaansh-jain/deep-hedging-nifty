# Checkpoint A: rebalance scaling of the BS delta hedge

GBM, 50000 paths, sigma = 0.18, mu = r = 0.065, 84 fine steps over 21 trading days, c = 0, ATM call, seed 777. The same paths are used for every n.

| n | std P&L | std x sqrt(n) | mean P&L | std / std(n=1) | mean / std |
|---|---|---|---|---|---|
| 1 | 0.015658 | 0.015658 | -0.000067 | 1.0000 | -0.0043 |
| 2 | 0.011422 | 0.016153 | 0.000017 | 0.7295 | 0.0015 |
| 4 | 0.008345 | 0.016691 | 0.000027 | 0.5330 | 0.0033 |
| 7 | 0.006438 | 0.017034 | 0.000020 | 0.4112 | 0.0031 |
| 21 | 0.003819 | 0.017499 | -0.000008 | 0.2439 | -0.0022 |
| 84 | 0.001953 | 0.017903 | -0.000011 | 0.1248 | -0.0056 |

- Std decreases monotonically in n: True
- Max deviation of std x sqrt(n) from its median: 0.071 (limit 0.25)
- Max |mean P&L| / std: 0.0056 (limit 0.1)

**PASS**

Plot: `results/validation/rebalance_scaling.png`.
