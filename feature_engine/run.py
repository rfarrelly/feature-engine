from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable
import re
import pandas as pd
import sys
from rs import compute_historical_ppi
from residuals import compute_residuals


@dataclass(frozen=True)
class SeasonFile:
    league: str
    season: str
    path: Path


# Matches exactly 4 consecutive digits (e.g., 2425, 2526)
_SEASON_RE = re.compile(r"(?P<season>\d{4})")


def _extract_season(filename: str) -> Optional[str]:
    """
    Extract season in 4-digit format.
    Example:
        Bundesliga_2425.csv -> 2425
        Serie-A_2526.csv -> 2526
    """
    match = _SEASON_RE.search(filename)
    return match.group("season") if match else None


def list_season_csvs(root_dir: str | Path) -> List[SeasonFile]:
    """
    Recursively scans directory structure like:

    DATA/
        FBDUK/
            Bundesliga/
                Bundesliga_2425.csv
                Bundesliga_2526.csv

    Returns list of SeasonFile objects.
    """
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    results: List[SeasonFile] = []

    for path in root.rglob("*.csv"):
        if not path.is_file():
            continue

        season = _extract_season(path.name)
        if not season:
            continue

        league = path.parent.name

        results.append(SeasonFile(league=league, season=season, path=path))

    results.sort(key=lambda x: (x.league.lower(), x.season))
    return results


def group_by_league(files: Iterable[SeasonFile]) -> dict[str, List[SeasonFile]]:
    grouped: dict[str, List[SeasonFile]] = {}
    for f in files:
        grouped.setdefault(f.league, []).append(f)
    # keep seasons sorted within each league
    for league in grouped:
        grouped[league].sort(key=lambda x: x.season)
    return grouped


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python data_file_index.py <root_dir>")

    root_dir = sys.argv[1]
    files = list_season_csvs(root_dir)

    print(f"Found {len(files)} CSV files under {Path(root_dir).resolve()}\n")
    for f in files:
        print(f"{f.league:25} {f.season:9} {f.path}")

    ppi_results = []
    for f in files:
        print(f"Processing {f.league} - {f.season}")
        ppi_df = compute_historical_ppi(str(f.path))
        ppi_results.append(ppi_df)

    all_historical_ppi = pd.concat(ppi_results)
    all_historical_ppi.to_csv("OUTPUT/historical_ppi.csv", index=False)
