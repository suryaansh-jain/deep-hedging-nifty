from __future__ import annotations

import argparse

from config import load_config
from export_dashboard_data import export_dashboard_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()
    output = export_dashboard_data(load_config(arguments.config))
    for path in sorted(output.iterdir()):
        print(f"{path.stat().st_size / 1024:8.1f} KB  {path}")


if __name__ == "__main__":
    main()
