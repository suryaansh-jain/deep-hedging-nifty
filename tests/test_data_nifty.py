from pathlib import Path

import pandas as pd
import pytest

import data_nifty


def test_load_close_series_sorts_and_drops_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "nsei.csv"
    pd.DataFrame({"Date": ["2020-01-03", "2020-01-02", "2020-01-02", "2020-01-06"], "Close": [101.0, 100.0, 100.0, 0.0]}).to_csv(path, index=False)
    series = data_nifty.load_close_series(path)
    assert list(series.values) == [100.0, 101.0]
    assert series.index.is_monotonic_increasing


def test_fetch_is_idempotent_when_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached = data_nifty.raw_csv_path(tmp_path, "^NSEI")
    pd.DataFrame({"Date": ["2020-01-02"], "Close": [100.0]}).to_csv(cached, index=False)

    def fail_download(*arguments: object) -> pd.DataFrame:
        raise AssertionError("download should not be called when cache exists")

    monkeypatch.setattr(data_nifty, "download_close_series", fail_download)
    assert data_nifty.fetch_series("^NSEI", tmp_path, "2007-01-01", None) == cached


def test_failed_download_explains_manual_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_download(*arguments: object) -> pd.DataFrame:
        raise ConnectionError("offline")

    monkeypatch.setattr(data_nifty, "download_close_series", broken_download)
    with pytest.raises(data_nifty.DataDownloadError, match="Place a CSV manually"):
        data_nifty.fetch_series("^NSEI", tmp_path, "2007-01-01", None)


def test_log_returns_and_split() -> None:
    series = pd.Series([100.0, 110.0, 99.0], index=pd.to_datetime(["2019-12-30", "2019-12-31", "2020-01-02"]))
    returns = data_nifty.log_returns(series)
    assert len(returns) == 2
    assert len(data_nifty.split_before(series, "2020-01-01")) == 2
    assert len(data_nifty.split_from(series, "2020-01-01")) == 1
