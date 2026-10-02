from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

MANUAL_CSV_HELP = (
    "Could not download {ticker} with yfinance. Place a CSV manually at {path} with columns "
    "'Date' (YYYY-MM-DD) and 'Close' (daily closing level), one row per trading day, then rerun."
)


class DataDownloadError(RuntimeError):
    pass


def raw_csv_path(raw_directory: Path, ticker: str) -> Path:
    safe_name = ticker.replace("^", "").lower()
    return raw_directory / f"{safe_name}.csv"


def download_close_series(ticker: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    import yfinance

    history = yfinance.download(
        ticker, start=start_date, end=end_date, auto_adjust=False, progress=False, threads=False
    )
    if history is None or history.empty:
        raise ValueError(f"empty download for {ticker}")
    close = history["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    frame = pd.DataFrame({"Date": pd.to_datetime(close.index).tz_localize(None), "Close": close.to_numpy()})
    return frame.dropna().reset_index(drop=True)


def fetch_series(ticker: str, raw_directory: Path, start_date: str, end_date: str | None, force: bool = False) -> Path:
    path = raw_csv_path(raw_directory, ticker)
    if path.exists() and not force:
        return path
    try:
        frame = download_close_series(ticker, start_date, end_date)
    except Exception as error:
        raise DataDownloadError(MANUAL_CSV_HELP.format(ticker=ticker, path=path) + f" Cause: {error}") from error
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def load_close_series(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(MANUAL_CSV_HELP.format(ticker=path.stem, path=path))
    frame = pd.read_csv(path, parse_dates=["Date"])
    series = frame.set_index("Date")["Close"].astype(float)
    series = series[series > 0].sort_index()
    return series[~series.index.duplicated(keep="first")]


def load_nifty(config: Config) -> pd.Series:
    return load_close_series(raw_csv_path(config.paths.resolve("raw"), config.data.nifty_ticker))


def load_india_vix(config: Config) -> pd.Series:
    return load_close_series(raw_csv_path(config.paths.resolve("raw"), config.data.vix_ticker))


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def split_before(series: pd.Series, cutoff_date: str) -> pd.Series:
    return series[series.index < pd.Timestamp(cutoff_date)]


def split_from(series: pd.Series, cutoff_date: str) -> pd.Series:
    return series[series.index >= pd.Timestamp(cutoff_date)]
