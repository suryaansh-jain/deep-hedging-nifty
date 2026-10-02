from __future__ import annotations

import argparse

from config import load_config
from data_nifty import fetch_series, load_close_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    raw_directory = config.paths.resolve("raw")
    for ticker in [config.data.nifty_ticker, config.data.vix_ticker]:
        path = fetch_series(ticker, raw_directory, config.data.start_date, config.data.end_date, force=arguments.force)
        series = load_close_series(path)
        print(f"{ticker}: {len(series)} rows from {series.index[0].date()} to {series.index[-1].date()} -> {path}")


if __name__ == "__main__":
    main()
