# workflow.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import (
    add_rolling_residuals,
    identify_residual_runs,
    build_runs,
    measure_run_outcomes,
    summarize_run_response,
    summarize_prospective_market_movement,
    build_prospective_market_movement,
    aggregate_team_seasons,
    run_parameter_grid,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESIDUAL_DIR = Path("residuals")

ROLLING_WINDOWS = (
    3,
    5,
    8,
)

RUN_WINDOWS = (
    3,
    5,
    8,
)

RUN_THRESHOLDS = (
    0.25,
    0.50,
    0.75,
    1.00,
)

HORIZONS = (
    1,
    2,
    3,
    5,
)

MIN_RUNS = 3

N_BOOTSTRAP = 5000

SEED = 42


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_all_residuals(
    directory: Path = RESIDUAL_DIR,
) -> pd.DataFrame:
    """Load all generated residual datasets."""

    files = sorted(Path(directory).glob("*/*.csv"))

    if not files:
        raise FileNotFoundError(f"No residual CSV files found in {directory}")

    frames = [pd.read_csv(file) for file in files]

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df["Date"] = pd.to_datetime(df["Date"])

    return df


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------


def run_analysis() -> dict[str, pd.DataFrame]:
    """Run the complete analysis."""

    df = load_all_residuals()

    # ---------------------------------------------------------------
    # Validate generated dataset
    # ---------------------------------------------------------------

    required_columns = [
        "Residual",
        "PreCloseWinProb",
        "CloseWinProb",
        "PreCloseDrawProb",
        "CloseDrawProb",
        "PreCloseLossProb",
        "CloseLossProb",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Residual dataset is missing required columns: "
            f"{missing}\n\n"
            "Run generate_residual_dataset.py first."
        )

    # ---------------------------------------------------------------
    # Rolling residual history
    # ---------------------------------------------------------------

    df = add_rolling_residuals(
        df,
        windows=ROLLING_WINDOWS,
    )

    # ---------------------------------------------------------------
    # Baseline run
    # ---------------------------------------------------------------

    signalled = identify_residual_runs(
        df,
        window=5,
        threshold=0.50,
    )

    runs = build_runs(
        signalled,
        window=5,
    )

    # ---------------------------------------------------------------
    # Future matches
    # ---------------------------------------------------------------

    outcomes = measure_run_outcomes(
        signalled,
        runs,
        horizons=HORIZONS,
    )

    # ---------------------------------------------------------------
    # Residual response
    # ---------------------------------------------------------------

    response = summarize_run_response(
        outcomes,
        horizons=HORIZONS,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    # ---------------------------------------------------------------
    # Prospective market movement
    # ---------------------------------------------------------------

    prospective_market_movement = build_prospective_market_movement(
        outcomes,
        horizons=HORIZONS,
    )

    market_movement_response = summarize_prospective_market_movement(
        outcomes,
        horizons=HORIZONS,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    # ---------------------------------------------------------------
    # Team-season responses
    # ---------------------------------------------------------------

    team_seasons = aggregate_team_seasons(
        outcomes,
        horizons=HORIZONS,
        min_runs=MIN_RUNS,
    )

    # ---------------------------------------------------------------
    # Run-definition sensitivity
    # ---------------------------------------------------------------

    sensitivity = run_parameter_grid(
        df,
        windows=RUN_WINDOWS,
        thresholds=RUN_THRESHOLDS,
        horizons=HORIZONS,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    return {
        "runs": runs,
        "outcomes": outcomes,
        "response": response,
        "prospective_market_movement": (prospective_market_movement),
        "market_movement_response": (market_movement_response),
        "team_seasons": team_seasons,
        "sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------


def save_results(
    results: dict[str, pd.DataFrame],
) -> None:
    """Save analysis outputs."""

    outputs = {
        "runs": "residual_runs.csv",
        "outcomes": "residual_run_outcomes.csv",
        "response": "residual_run_response.csv",
        "prospective_market_movement": ("prospective_market_movement.csv"),
        "market_movement_response": ("market_movement_response.csv"),
        "team_seasons": ("team_season_responses.csv"),
        "sensitivity": ("parameter_sensitivity.csv"),
    }

    for key, filename in outputs.items():

        results[key].to_csv(
            filename,
            index=False,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


if __name__ == "__main__":

    results = run_analysis()

    print("\nOVERALL RUN RESPONSE")
    print(results["response"].to_string(index=False))

    print("\nPROSPECTIVE MARKET MOVEMENT")
    print(results["market_movement_response"].to_string(index=False))

    print("\nTOP TEAM-SEASON RESPONSES")
    print(
        results["team_seasons"]
        .sort_values(
            "MeanResidual",
            ascending=False,
        )
        .head(50)
        .to_string(index=False)
    )

    print("\nPARAMETER SENSITIVITY")
    print(results["sensitivity"].to_string(index=False))

    print("\nRUN COUNTS")
    print(results["runs"].groupby("RunSignal").size())

    save_results(results)
