from pathlib import Path

import pandas as pd

from analysis import (
    add_rolling_features,
    identify_episodes,
    summarize_episodes,
    measure_episode_outcomes,
    summarize_episode_performance,
    summarize_forward_residuals,
    compare_episode_signals,
)

RESIDUAL_DIR = Path("residuals")

HORIZONS = (1, 2, 3, 5)

Z_COLUMN = "ResidualZ_3"

POSITIVE_THRESHOLD = 1.25
NEGATIVE_THRESHOLD = -1.25

N_BOOTSTRAP = 5000


def load_all_residuals(directory=RESIDUAL_DIR):
    """
    Load all previously generated team-match residual datasets.
    """

    files = Path(directory).glob("*/*.csv")

    frames = [pd.read_csv(file) for file in files]

    if not frames:
        raise FileNotFoundError(f"No residual CSV files found in {directory}")

    return pd.concat(
        frames,
        ignore_index=True,
    )


def run_analysis():

    # ---------------------------------------------------------------
    # Load residual data
    # ---------------------------------------------------------------

    df = load_all_residuals()

    # ---------------------------------------------------------------
    # Build rolling signal
    # ---------------------------------------------------------------

    df = add_rolling_features(
        df,
        windows=(3,),
    )

    # ---------------------------------------------------------------
    # Identify extreme episodes
    # ---------------------------------------------------------------

    df = identify_episodes(
        df,
        z_column=Z_COLUMN,
        positive_threshold=POSITIVE_THRESHOLD,
        negative_threshold=NEGATIVE_THRESHOLD,
    )

    # ---------------------------------------------------------------
    # Episode-level dataset
    # ---------------------------------------------------------------

    episodes = summarize_episodes(
        df,
        z_column=Z_COLUMN,
    )

    # ---------------------------------------------------------------
    # Forward outcomes
    # ---------------------------------------------------------------

    episode_outcomes = measure_episode_outcomes(
        df,
        episodes,
        horizons=HORIZONS,
    )

    # ---------------------------------------------------------------
    # Episode performance
    # ---------------------------------------------------------------

    episode_performance = summarize_episode_performance(
        episode_outcomes,
        horizons=HORIZONS,
        n_bootstrap=N_BOOTSTRAP,
    )

    # ---------------------------------------------------------------
    # Individual forward residuals
    # ---------------------------------------------------------------

    forward_residuals = summarize_forward_residuals(
        episode_outcomes,
        horizons=HORIZONS,
    )

    # ---------------------------------------------------------------
    # Positive vs negative comparison
    # ---------------------------------------------------------------

    signal_comparison = compare_episode_signals(
        episode_outcomes,
        horizons=HORIZONS,
        n_bootstrap=N_BOOTSTRAP,
    )

    return {
        "episodes": episodes,
        "episode_outcomes": episode_outcomes,
        "episode_performance": episode_performance,
        "forward_residuals": forward_residuals,
        "signal_comparison": signal_comparison,
    }


if __name__ == "__main__":

    results = run_analysis()

    print("\nEPISODE PERFORMANCE")
    print(results["episode_performance"])

    print("\nPOSITIVE VS NEGATIVE")
    print(results["signal_comparison"])

    print("\nEPISODE FORWARD RESIDUALS")
    print(results["forward_residuals"])

    results["episodes"].to_csv(
        "episode_summary.csv",
        index=False,
    )

    results["episode_outcomes"].to_csv(
        "episode_outcomes.csv",
        index=False,
    )

    results["episode_performance"].to_csv(
        "episode_performance.csv",
        index=False,
    )

    results["forward_residuals"].to_csv(
        "episode_forward_residuals.csv",
        index=False,
    )

    results["signal_comparison"].to_csv(
        "episode_signal_comparison.csv",
        index=False,
    )
