from pathlib import Path

import numpy as np

from export_dashboard_data import histogram_rows, real_window_rows, sample_path_arrays


def test_histograms_sum_to_one_and_paths_are_keyed(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.npz"
    rng = np.random.default_rng(0)
    np.savez(path, pnl__bs_delta__5=rng.normal(0, 0.01, 1000), positions__bs_delta__5=np.zeros((3, 21)), sample_prices=np.ones((3, 22)), sample_volatility=np.ones((3, 22)), times=np.arange(22))
    archive = np.load(path)
    rows = histogram_rows(archive, "garch", 0.02)
    assert abs(sum(row["probability"] for row in rows) - 1.0) < 1e-12
    assert {row["strategy"] for row in rows} == {"bs_delta"}
    arrays = sample_path_arrays(archive, "garch")
    assert set(arrays) == {"garch__prices", "garch__volatility", "garch__positions__bs_delta__5"}


def test_real_window_rows(tmp_path: Path) -> None:
    path = tmp_path / "real.npz"
    np.savez(path, non_overlapping__start_dates=np.array(["2020-01-01", "2020-02-01"]), non_overlapping__pnl__ntb__5=np.array([0.1, -0.2]))
    rows = real_window_rows(np.load(path), "non_overlapping")
    assert rows == [
        {"windows": "non_overlapping", "start_date": "2020-01-01", "cost_bps": 5.0, "strategy": "ntb", "pnl_x_premium": 0.1},
        {"windows": "non_overlapping", "start_date": "2020-02-01", "cost_bps": 5.0, "strategy": "ntb", "pnl_x_premium": -0.2},
    ]
