from pathlib import Path

import pandas as pd

from residuals import build_residual_dataset

from analysis import (
    add_rolling_features,
    evaluate_thresholds,
    add_confidence_intervals,
    identify_episodes,
    summarize_episodes,
    measure_episode_outcomes,
    summarize_episode_performance,
)

DATA_DIR = Path("/Users/ryanfarrelly/Desktop/collector/DATA/football-data")

OUTPUT_DIR = Path("residuals")


# ---------------------------------------------------------------------
# Build residual datasets
# ---------------------------------------------------------------------


def load_all_residuals(directory="residuals"):
    files = Path(directory).glob("*/*.csv")

    frames = [pd.read_csv(file) for file in files]

    if not frames:
        raise FileNotFoundError(f"No residual CSV files found in {directory}")

    return pd.concat(
        frames,
        ignore_index=True,
    )


def process_all_leagues():
    for csv_file in DATA_DIR.glob("*/*.csv"):

        league = csv_file.parent.name
        season = csv_file.stem

        print(f"Processing {league} — {season}")

        result = build_residual_dataset(csv_file)

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
# Load data
# ---------------------------------------------------------------------


df = load_all_residuals(directory="residuals")


# ---------------------------------------------------------------------
# Rolling team history
# ---------------------------------------------------------------------


df = add_rolling_features(
    df,
    windows=(3, 5),
)


# ---------------------------------------------------------------------
# Existing pooled threshold analysis
# ---------------------------------------------------------------------


results = evaluate_thresholds(
    df,
    z_column="ResidualZ_3",
)

results = add_confidence_intervals(
    results,
    df,
    z_column="ResidualZ_3",
)


print("\nOVERALL")
print(results)


# ---------------------------------------------------------------------
# Home / away threshold analysis
# ---------------------------------------------------------------------


home = df[df["Venue"] == "home"].copy()

away = df[df["Venue"] == "away"].copy()


home_results = evaluate_thresholds(
    home,
    z_column="ResidualZ_3",
)

home_results = add_confidence_intervals(
    home_results,
    home,
    z_column="ResidualZ_3",
)


away_results = evaluate_thresholds(
    away,
    z_column="ResidualZ_3",
)

away_results = add_confidence_intervals(
    away_results,
    away,
    z_column="ResidualZ_3",
)


print("\nHOME")
print(home_results)

print("\nAWAY")
print(away_results)


# ---------------------------------------------------------------------
# Episode detection
# ---------------------------------------------------------------------


df = identify_episodes(
    df,
    z_column="ResidualZ_3",
    positive_threshold=1.25,
    negative_threshold=-1.25,
)


# ---------------------------------------------------------------------
# Episode summaries
# ---------------------------------------------------------------------


episodes = summarize_episodes(
    df,
    z_column="ResidualZ_3",
)


# ---------------------------------------------------------------------
# Forward episode outcomes
# ---------------------------------------------------------------------


episode_outcomes = measure_episode_outcomes(
    df,
    episodes,
    horizons=(1, 2, 3, 5),
)


# ---------------------------------------------------------------------
# Aggregate episode performance
# ---------------------------------------------------------------------


episode_results = summarize_episode_performance(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
)


print("\nEPISODE PERFORMANCE")
print(episode_results)


# ---------------------------------------------------------------------
# Save episode-level data
# ---------------------------------------------------------------------


episodes.to_csv(
    "episode_summary.csv",
    index=False,
)

episode_outcomes.to_csv(
    "episode_outcomes.csv",
    index=False,
)
breakpoint()
