# analysis.py

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Rolling residual history
# ---------------------------------------------------------------------


def add_rolling_residuals(
    df: pd.DataFrame,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """
    Add rolling residual means using ONLY matches already completed.

    This is deliberately shifted by one match to prevent the current
    match from influencing the signal used to enter the next match.
    """

    df = df.copy()

    df = df.sort_values(["League", "Season", "Team", "Match"]).reset_index(drop=True)

    group = df.groupby(
        ["League", "Season", "Team"],
        sort=False,
    )["Residual"]

    for window in windows:
        df[f"RollingResidual_{window}"] = (
            group.rolling(window)
            .mean()
            .shift(1)
            .reset_index(level=[0, 1, 2], drop=True)
        )

    return df


# ---------------------------------------------------------------------
# Signal identification
# ---------------------------------------------------------------------


def identify_residual_runs(
    df: pd.DataFrame,
    window: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Identify teams whose previous residual history is sufficiently
    positive or negative to create a signal for the CURRENT match.

    The signal is therefore available before the current match.
    """

    df = df.copy()

    column = f"RollingResidual_{window}"

    if column not in df.columns:
        raise ValueError(f"{column} not found. " "Run add_rolling_residuals first.")

    df["RunSignal"] = pd.NA

    positive = df[column] >= threshold
    negative = df[column] <= -threshold

    df.loc[positive, "RunSignal"] = "positive"
    df.loc[negative, "RunSignal"] = "negative"

    return df


# ---------------------------------------------------------------------
# Run construction
# ---------------------------------------------------------------------


def build_runs(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Build signal runs.

    A run is a consecutive sequence of matches for the same
    team-season carrying the same signal.
    """

    df = df.copy()

    df["SignalChange"] = df["RunSignal"] != df.groupby(["League", "Season", "Team"])[
        "RunSignal"
    ].shift(1)

    df["RunID"] = df.groupby(["League", "Season", "Team"])["SignalChange"].cumsum()

    runs = (
        df[df["RunSignal"].notna()]
        .groupby(
            [
                "League",
                "Season",
                "Team",
                "RunSignal",
                "RunID",
            ],
            as_index=False,
        )
        .agg(
            StartMatch=("Match", "min"),
            EndMatch=("Match", "max"),
            RunLength=("Match", "size"),
            StartDate=("Date", "min"),
            EndDate=("Date", "max"),
        )
    )

    return runs


# ---------------------------------------------------------------------
# Future match outcomes
# ---------------------------------------------------------------------


def measure_run_outcomes(
    signalled: pd.DataFrame,
    runs: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """
    Attach future match outcomes to every signal.

    For a signal at match N, MatchAhead=1 refers to N+1,
    MatchAhead=2 to N+2, etc.
    """

    df = signalled.copy()

    key_columns = [
        "League",
        "Season",
        "Team",
    ]

    history_columns = [
        "Date",
        "League",
        "Season",
        "Team",
        "Opponent",
        "Venue",
        "Match",
        "GoalsFor",
        "GoalsAgainst",
        "GoalDifference",
        "ActualPoints",
        "Residual",
        "PreCloseWinProb",
        "PreCloseDrawProb",
        "PreCloseLossProb",
        "CloseWinProb",
        "CloseDrawProb",
        "CloseLossProb",
    ]

    history = df[history_columns].sort_values(key_columns + ["Match"]).copy()

    signal_rows = df[df["RunSignal"].notna()][
        key_columns
        + [
            "Match",
            "RunSignal",
            "RollingResidual_3",
            "RollingResidual_5",
            "RollingResidual_8",
        ]
    ].copy()

    signal_rows = signal_rows.rename(columns={"Match": "SignalMatch"})

    outputs = []

    for horizon in horizons:

        future = history.copy()

        future["SignalMatch"] = future["Match"] - horizon

        future = future.rename(
            columns={
                column: f"Future_{column}"
                for column in history_columns
                if column not in key_columns
            }
        )

        merged = signal_rows.merge(
            future,
            on=key_columns + ["SignalMatch"],
            how="left",
        )

        merged["MatchAhead"] = horizon

        outputs.append(merged)

    return pd.concat(
        outputs,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------


def bootstrap_mean_ci(
    values: pd.Series,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for a mean."""

    values = pd.Series(values).dropna().to_numpy()

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return values[0], values[0]

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


# ---------------------------------------------------------------------
# Overall response
# ---------------------------------------------------------------------


def summarize_run_response(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarise future residual response."""

    rows = []

    for signal in ["positive", "negative"]:

        for horizon in horizons:

            subset = outcomes[
                (outcomes["RunSignal"] == signal)
                & (outcomes["MatchAhead"] == horizon)
                & outcomes["Future_Residual"].notna()
            ]

            values = subset["Future_Residual"]

            lower, upper = bootstrap_mean_ci(
                values,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            rows.append(
                {
                    "RunSignal": signal,
                    "MatchAhead": horizon,
                    "N": len(values),
                    "TeamSeasons": (
                        subset[["League", "Season", "Team"]].drop_duplicates().shape[0]
                    ),
                    "MeanResidual": values.mean(),
                    "MedianResidual": values.median(),
                    "Positive_%": ((values > 0).mean() if len(values) else np.nan),
                    "CI_Lower": lower,
                    "CI_Upper": upper,
                    "CI_Excludes_Zero": (lower > 0 or upper < 0),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Prospective market movement
# ---------------------------------------------------------------------


def build_prospective_market_movement(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """
    Construct the event-level prospective market dataset.

    Crucially, market movement belongs to the FUTURE match.

    Example:

        Signal after Match 10
        Future MatchAhead=1 = Match 11

        Future_PreCloseWinProb
        Future_CloseWinProb

    are therefore the odds available before Match 11 and at its close.

    No result from Match 11 is used to construct the movement itself.
    """

    df = outcomes.copy()

    df = df[df["MatchAhead"].isin(horizons)].copy()

    df = df[
        df["Future_PreCloseWinProb"].notna() & df["Future_CloseWinProb"].notna()
    ].copy()

    # ---------------------------------------------------------------
    # Probability movement
    # ---------------------------------------------------------------

    df["WinProbMove"] = df["Future_CloseWinProb"] - df["Future_PreCloseWinProb"]

    df["DrawProbMove"] = df["Future_CloseDrawProb"] - df["Future_PreCloseDrawProb"]

    df["LossProbMove"] = df["Future_CloseLossProb"] - df["Future_PreCloseLossProb"]

    # ---------------------------------------------------------------
    # Absolute movement
    # ---------------------------------------------------------------

    df["AbsWinProbMove"] = df["WinProbMove"].abs()
    df["AbsDrawProbMove"] = df["DrawProbMove"].abs()
    df["AbsLossProbMove"] = df["LossProbMove"].abs()

    df["TotalMarketMovement"] = (
        df["AbsWinProbMove"] + df["AbsDrawProbMove"] + df["AbsLossProbMove"]
    )

    # ---------------------------------------------------------------
    # Direction relative to signal
    #
    # Positive signal:
    #   Did the market subsequently become MORE positive?
    #
    # Negative signal:
    #   Did the market subsequently become LESS negative?
    # ---------------------------------------------------------------

    df["SignalAlignedWinMove"] = np.where(
        df["RunSignal"] == "positive",
        df["WinProbMove"],
        -df["WinProbMove"],
    )

    df["MarketMovedTowardSignal"] = df["SignalAlignedWinMove"] > 0

    # ---------------------------------------------------------------
    # Keep useful columns only
    # ---------------------------------------------------------------

    columns = [
        "League",
        "Season",
        "Team",
        "Opponent",
        "Venue",
        "SignalMatch",
        "MatchAhead",
        "RunSignal",
        "Future_Match",
        "Future_Date",
        "Future_PreCloseWinProb",
        "Future_CloseWinProb",
        "WinProbMove",
        "Future_PreCloseDrawProb",
        "Future_CloseDrawProb",
        "DrawProbMove",
        "Future_PreCloseLossProb",
        "Future_CloseLossProb",
        "LossProbMove",
        "AbsWinProbMove",
        "AbsDrawProbMove",
        "AbsLossProbMove",
        "TotalMarketMovement",
        "SignalAlignedWinMove",
        "MarketMovedTowardSignal",
        # Outcome is retained for later value analysis,
        # but was NOT used to calculate movement.
        "Future_ActualPoints",
        "Future_Residual",
        "Future_GoalDifference",
    ]

    columns = [column for column in columns if column in df.columns]

    return (
        df[columns]
        .sort_values(
            [
                "League",
                "Season",
                "Team",
                "SignalMatch",
                "MatchAhead",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_prospective_market_movement(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarise prospective market movement."""

    movement = build_prospective_market_movement(
        outcomes,
        horizons,
    )

    rows = []

    for signal in ["positive", "negative"]:

        for horizon in horizons:

            subset = movement[
                (movement["RunSignal"] == signal) & (movement["MatchAhead"] == horizon)
            ].copy()

            values = subset["WinProbMove"]

            lower, upper = bootstrap_mean_ci(
                values,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            rows.append(
                {
                    "RunSignal": signal,
                    "MatchAhead": horizon,
                    "N": len(subset),
                    "TeamSeasons": (
                        subset[["League", "Season", "Team"]].drop_duplicates().shape[0]
                    ),
                    "MeanWinProbMove": values.mean(),
                    "MedianWinProbMove": values.median(),
                    "MeanAbsWinProbMove": (values.abs().mean()),
                    "MeanTotalMarketMovement": (subset["TotalMarketMovement"].mean()),
                    "PctMovedTowardSignal": (subset["MarketMovedTowardSignal"].mean()),
                    "CI_Lower": lower,
                    "CI_Upper": upper,
                    "CI_Excludes_Zero": (lower > 0 or upper < 0),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Team-season analysis
# ---------------------------------------------------------------------


def aggregate_team_seasons(
    outcomes: pd.DataFrame,
    min_runs: int = 3,
) -> pd.DataFrame:
    """Aggregate future residual response by team-season."""

    valid = outcomes[
        outcomes["Future_Residual"].notna() & outcomes["RunSignal"].notna()
    ].copy()

    grouped = valid.groupby(
        [
            "League",
            "Season",
            "Team",
            "RunSignal",
        ],
        as_index=False,
    ).agg(
        Runs=("SignalMatch", "nunique"),
        MeanResidual=("Future_Residual", "mean"),
        MedianResidual=("Future_Residual", "median"),
        PositiveRate=(
            "Future_Residual",
            lambda x: (x > 0).mean(),
        ),
    )

    return (
        grouped[grouped["Runs"] >= min_runs]
        .sort_values(
            "MeanResidual",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------


def run_parameter_grid(
    df: pd.DataFrame,
    windows: tuple[int, ...],
    thresholds: tuple[float, ...],
    horizons: tuple[int, ...],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Test whether the residual effect depends materially on the
    signal definition.

    The output is intentionally compact.
    """

    rows = []

    for window in windows:

        column = f"RollingResidual_{window}"

        if column not in df.columns:
            continue

        for threshold in thresholds:

            signalled = identify_residual_runs(
                df,
                window=window,
                threshold=threshold,
            )

            runs = build_runs(
                signalled,
                window=window,
            )

            if runs.empty:
                continue

            outcomes = measure_run_outcomes(
                signalled,
                runs,
                horizons=horizons,
            )

            for signal in ["positive", "negative"]:

                for horizon in horizons:

                    subset = outcomes[
                        (outcomes["RunSignal"] == signal)
                        & (outcomes["MatchAhead"] == horizon)
                        & outcomes["Future_Residual"].notna()
                    ]

                    values = subset["Future_Residual"]

                    if values.empty:
                        continue

                    lower, upper = bootstrap_mean_ci(
                        values,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )

                    rows.append(
                        {
                            "Window": window,
                            "Threshold": threshold,
                            "RunSignal": signal,
                            "MatchAhead": horizon,
                            "N": len(values),
                            "MeanResidual": values.mean(),
                            "CI_Lower": lower,
                            "CI_Upper": upper,
                            "CI_Excludes_Zero": (lower > 0 or upper < 0),
                        }
                    )

    return pd.DataFrame(rows)
