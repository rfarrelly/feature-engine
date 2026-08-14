from pathlib import Path

import pandas as pd

from residuals import build_residual_dataset
from analysis import (
    add_rolling_features,
    evaluate_thresholds,
    add_confidence_intervals,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


DATA_DIR = Path("/Users/ryanfarrelly/Desktop/collector/DATA/football-data")

OUTPUT_DIR = Path("residuals")


# ---------------------------------------------------------------------
# Residual dataset handling
# ---------------------------------------------------------------------


def load_all_residuals(directory="residuals"):
    """
    Load all previously generated team-match residual datasets.
    """

    files = sorted(Path(directory).glob("*/*.csv"))

    if not files:
        raise FileNotFoundError(f"No residual CSV files found in {directory}")

    frames = [pd.read_csv(file) for file in files]

    return pd.concat(
        frames,
        ignore_index=True,
    )


def process_all_leagues():
    """
    Build the canonical residual dataset for every
    league/season CSV in DATA_DIR.
    """

    for csv_file in sorted(DATA_DIR.glob("*/*.csv")):

        league = csv_file.parent.name
        season = csv_file.stem

        print(f"Processing {league} — {season}")

        result = build_residual_dataset(
            csv_file,
            residual_definition="points",
        )

        output_dir = OUTPUT_DIR / league

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = output_dir / f"{season}.csv"

        result.to_csv(
            output_path,
            index=False,
        )


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------


def analyse_dataset(df):
    """
    Add team-history signals and evaluate
    the next-match relationship.
    """

    df = add_rolling_features(
        df,
        windows=(3, 5),
    )

    # ---------------------------------------------------------------
    # Overall
    # ---------------------------------------------------------------

    overall = evaluate_thresholds(
        df,
        z_column="RollingZ_3",
    )

    overall = add_confidence_intervals(
        overall,
        df,
        z_column="RollingZ_3",
    )

    # ---------------------------------------------------------------
    # Home
    # ---------------------------------------------------------------

    home = df[df["Venue"] == "home"].copy()

    home_results = evaluate_thresholds(
        home,
        z_column="RollingZ_3",
    )

    home_results = add_confidence_intervals(
        home_results,
        home,
        z_column="RollingZ_3",
    )

    # ---------------------------------------------------------------
    # Away
    # ---------------------------------------------------------------

    away = df[df["Venue"] == "away"].copy()

    away_results = evaluate_thresholds(
        away,
        z_column="RollingZ_3",
    )

    away_results = add_confidence_intervals(
        away_results,
        away,
        z_column="RollingZ_3",
    )

    return (
        df,
        overall,
        home_results,
        away_results,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


if __name__ == "__main__":

    # Uncomment this when the raw datasets need rebuilding.
    #
    # process_all_leagues()

    df = load_all_residuals(directory=OUTPUT_DIR)

    (
        df,
        overall,
        home_results,
        away_results,
    ) = analyse_dataset(df)

    print("\nOVERALL")
    print(overall.to_string(index=False))

    print("\nHOME")
    print(home_results.to_string(index=False))

    print("\nAWAY")
    print(away_results.to_string(index=False))
