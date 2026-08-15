from pathlib import Path

import pandas as pd

from residuals import build_residual_dataset

from analysis import (
    add_rolling_features,
    identify_episodes,
    summarize_episodes,
    measure_episode_outcomes,
    summarize_episode_performance,
    summarize_forward_residuals,
    compare_episode_signals,
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
# Load existing residual datasets
# ---------------------------------------------------------------------


df = load_all_residuals(directory="residuals")


# ---------------------------------------------------------------------
# Build signal
# ---------------------------------------------------------------------


df = add_rolling_features(
    df,
    windows=(3,),
)


# ---------------------------------------------------------------------
# Identify extreme episodes
# ---------------------------------------------------------------------


df = identify_episodes(
    df,
    z_column="ResidualZ_3",
    positive_threshold=1.25,
    negative_threshold=-1.25,
)


# ---------------------------------------------------------------------
# Summarize episodes
# ---------------------------------------------------------------------


episodes = summarize_episodes(
    df,
    z_column="ResidualZ_3",
)


print("\nEPISODE SUMMARY")
print(episodes.head())


# ---------------------------------------------------------------------
# Measure forward outcomes
# ---------------------------------------------------------------------


episode_outcomes = measure_episode_outcomes(
    df,
    episodes,
    horizons=(1, 2, 3, 5),
)


print("\nEPISODE OUTCOMES")
print(episode_outcomes.head())


# ---------------------------------------------------------------------
# Clustered episode performance
# ---------------------------------------------------------------------


episode_results = summarize_episode_performance(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
    n_bootstrap=5000,
)


print("\nEPISODE PERFORMANCE")
print(episode_results)


# ---------------------------------------------------------------------
# Exact forward residuals
# ---------------------------------------------------------------------


forward_results = summarize_forward_residuals(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
)


print("\nEPISODE FORWARD RESIDUALS")
print(forward_results)


# ---------------------------------------------------------------------
# Direct positive-vs-negative comparison
# ---------------------------------------------------------------------


comparison_results = compare_episode_signals(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
    n_bootstrap=5000,
)


print("\nPOSITIVE VS NEGATIVE")
print(comparison_results)


# ---------------------------------------------------------------------
# Episode diagnostics
# ---------------------------------------------------------------------


print("\nEPISODE LENGTH")

print(episodes.groupby("EpisodeSignal")["Length"].describe())


print("\nEPISODES PER TEAM-SEASON")

episode_counts = episodes.groupby(["League", "Season", "Team"]).size()

print(episode_counts.describe())


# ---------------------------------------------------------------------
# Save analysis datasets
# ---------------------------------------------------------------------


episodes.to_csv(
    "episode_summary.csv",
    index=False,
)

episode_outcomes.to_csv(
    "episode_outcomes.csv",
    index=False,
)

episode_results.to_csv(
    "episode_performance.csv",
    index=False,
)

forward_results.to_csv(
    "episode_forward_residuals.csv",
    index=False,
)

comparison_results.to_csv(
    "episode_signal_comparison.csv",
    index=False,
)
