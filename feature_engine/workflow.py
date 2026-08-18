# workflow.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import (
    add_rolling_residuals,
    aggregate_team_seasons,
    build_runs,
    identify_residual_runs,
    measure_run_outcomes,
    run_parameter_grid,
    summarize_market_movement_response,
    summarize_run_response,
)

RESIDUAL_DIR = Path("residuals")

ROLLING_WINDOWS = (3, 5, 8)
RUN_WINDOWS = (3, 5, 8)
RUN_THRESHOLDS = (0.25, 0.50, 0.75, 1.00)
HORIZONS = (1, 2, 3, 5)

MIN_RUNS = 3
N_BOOTSTRAP = 5000
SEED = 42


def load_all_residuals(
    directory: Path = RESIDUAL_DIR,
) -> pd.DataFrame:
    """Load all generated residual CSV files."""

    files = sorted(directory.glob("*/*.csv"))

    if not files:
        raise FileNotFoundError(f"No residual CSV files found in {directory}")

    frames = [pd.read_csv(file) for file in files]

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def run_analysis() -> dict[str, pd.DataFrame]:
    """Run the complete residual-run analysis."""

    df = load_all_residuals()

    df = add_rolling_residuals(
        df,
        windows=ROLLING_WINDOWS,
    )

    signalled = identify_residual_runs(
        df,
        window=5,
        threshold=0.50,
    )

    runs = build_runs(
        signalled,
        window=5,
    )

    outcomes = measure_run_outcomes(
        signalled,
        runs,
        horizons=HORIZONS,
    )

    response = summarize_run_response(
        outcomes,
        horizons=HORIZONS,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    market_movement_response = summarize_market_movement_response(
        outcomes,
        horizons=HORIZONS,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    team_seasons = aggregate_team_seasons(
        outcomes,
        horizons=HORIZONS,
        min_runs=MIN_RUNS,
    )

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
        "market_movement_response": market_movement_response,
        "team_seasons": team_seasons,
        "sensitivity": sensitivity,
    }


def save_results(
    results: dict[str, pd.DataFrame],
) -> None:
    """Save analysis outputs."""

    outputs = {
        "runs": "residual_runs.csv",
        "outcomes": "residual_run_outcomes.csv",
        "response": "residual_run_response.csv",
        "market_movement_response": "market_movement_response.csv",
        "team_seasons": "team_season_responses.csv",
        "sensitivity": "parameter_sensitivity.csv",
    }

    for key, filename in outputs.items():
        results[key].to_csv(
            filename,
            index=False,
        )


if __name__ == "__main__":
    results = run_analysis()

    print("\nOVERALL RUN RESPONSE")
    print(results["response"].to_string(index=False))

    print("\nMARKET MOVEMENT RESPONSE")
    print(results["market_movement_response"].to_string(index=False))

    print("\nTOP TEAM-SEASON RESPONSES")
    print(
        results["team_seasons"]
        .sort_values("MeanResidual", ascending=False)
        .head(50)
        .to_string(index=False)
    )

    print("\nPARAMETER SENSITIVITY")
    print(results["sensitivity"].to_string(index=False))

    print("\nRUN COUNTS")
    print(results["runs"].groupby("RunSignal").size())

    save_results(results)
