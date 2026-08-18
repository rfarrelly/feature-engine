# analysis.py

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Rolling residuals
# ---------------------------------------------------------------------


def add_rolling_residuals(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 5, 8),
) -> pd.DataFrame:
    """
    Add trailing residual statistics.

    The current match is excluded.

    History resets for each League / Season / Team.
    """

    df = df.copy()

    group_cols = [
        "League",
        "Season",
        "Team",
    ]

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(group_cols + ["Date", "Match"]).reset_index(drop=True)

    grouped = df.groupby(
        group_cols,
        sort=False,
    )["Residual"]

    for window in windows:

        prior = grouped.shift(1)

        rolling = prior.groupby(
            [
                df["League"],
                df["Season"],
                df["Team"],
            ],
            sort=False,
        ).rolling(
            window=window,
            min_periods=window,
        )

        df[f"ResidualMean_{window}"] = (
            rolling.mean()
            .reset_index(
                level=group_cols,
                drop=True,
            )
            .to_numpy()
        )

        df[f"ResidualStd_{window}"] = (
            rolling.std(ddof=1)
            .reset_index(
                level=group_cols,
                drop=True,
            )
            .to_numpy()
        )

        df[f"ResidualSum_{window}"] = (
            rolling.sum()
            .reset_index(
                level=group_cols,
                drop=True,
            )
            .to_numpy()
        )

    return df


# ---------------------------------------------------------------------
# Residual runs
# ---------------------------------------------------------------------


def identify_residual_runs(
    df: pd.DataFrame,
    window: int = 5,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Identify positive and negative residual runs.

    The signal for match t uses only residuals from matches before t.
    """

    df = df.copy()

    signal_column = f"ResidualMean_{window}"

    if signal_column not in df.columns:
        raise ValueError(
            f"{signal_column} not found. " "Run add_rolling_residuals() first."
        )

    df["RunSignal"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    df.loc[
        df[signal_column] >= threshold,
        "RunSignal",
    ] = "positive"

    df.loc[
        df[signal_column] <= -threshold,
        "RunSignal",
    ] = "negative"

    group_cols = [
        "League",
        "Season",
        "Team",
    ]

    previous = df.groupby(
        group_cols,
        sort=False,
    )[
        "RunSignal"
    ].shift(1)

    new_run = df["RunSignal"].notna() & (previous.isna() | df["RunSignal"].ne(previous))

    run_number = (
        new_run.astype(int)
        .groupby(
            [
                df["League"],
                df["Season"],
                df["Team"],
            ],
            sort=False,
        )
        .cumsum()
    )

    df["RunID"] = np.where(
        df["RunSignal"].notna(),
        run_number,
        np.nan,
    )

    return df


# ---------------------------------------------------------------------
# Run construction
# ---------------------------------------------------------------------


def build_runs(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Create one row per residual run."""

    signal = f"ResidualMean_{window}"

    required = [
        "League",
        "Season",
        "Team",
        "Match",
        "RunID",
        "RunSignal",
        signal,
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[df["RunID"].notna()].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "League",
                "Season",
                "Team",
                "RunID",
                "RunSignal",
                "StartMatch",
                "EndMatch",
                "Length",
                "EntrySignal",
                "MeanSignal",
                "PeakSignal",
            ]
        )

    group_cols = [
        "League",
        "Season",
        "Team",
        "RunID",
    ]

    grouped = data.groupby(
        group_cols,
        sort=False,
    )

    result = grouped.agg(
        RunSignal=("RunSignal", "first"),
        StartMatch=("Match", "min"),
        EndMatch=("Match", "max"),
        Length=("Match", "size"),
        EntrySignal=(signal, "first"),
        MeanSignal=(signal, "mean"),
    ).reset_index()

    peak = (
        grouped[signal]
        .apply(lambda x: x.loc[x.abs().idxmax()] if x.notna().any() else np.nan)
        .rename("PeakSignal")
        .reset_index()
    )

    return result.merge(
        peak,
        on=group_cols,
        how="left",
    )


# ---------------------------------------------------------------------
# Forward outcomes
# ---------------------------------------------------------------------


def measure_run_outcomes(
    df: pd.DataFrame,
    runs: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """
    Attach information from matches after each run.

    MatchAhead=1 is the first match after the run.

    The resulting columns describe the future match and include:

        Residual
        PreCloseWinProb
        CloseWinProb
        WinProbMove
        PreCloseDrawProb
        CloseDrawProb
        DrawProbMove
        PreCloseLossProb
        CloseLossProb
        LossProbMove

    These are measured only after the run has ended.
    """

    if runs.empty:
        return runs.copy()

    group_cols = [
        "League",
        "Season",
        "Team",
    ]

    columns = [
        *group_cols,
        "Match",
        "Residual",
        "PreCloseWinProb",
        "CloseWinProb",
        "PreCloseDrawProb",
        "CloseDrawProb",
        "PreCloseLossProb",
        "CloseLossProb",
    ]

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Residual dataset is missing required columns: "
            f"{missing}. "
            "Regenerate the residual dataset with "
            "generate_residual_dataset.py."
        )

    history = df[columns].sort_values(group_cols + ["Match"]).copy()

    history["WinProbMove"] = history["CloseWinProb"] - history["PreCloseWinProb"]

    history["DrawProbMove"] = history["CloseDrawProb"] - history["PreCloseDrawProb"]

    history["LossProbMove"] = history["CloseLossProb"] - history["PreCloseLossProb"]

    history = history.set_index(group_cols + ["Match"])

    result = runs.copy()

    for horizon in horizons:

        rows = []

        for row in result.itertuples(index=False):

            key = (
                row.League,
                row.Season,
                row.Team,
                int(row.EndMatch) + horizon,
            )

            try:
                match = history.loc[key]

            except KeyError:
                rows.append(
                    {
                        "Residual": np.nan,
                        "PreCloseWinProb": np.nan,
                        "CloseWinProb": np.nan,
                        "WinProbMove": np.nan,
                        "PreCloseDrawProb": np.nan,
                        "CloseDrawProb": np.nan,
                        "DrawProbMove": np.nan,
                        "PreCloseLossProb": np.nan,
                        "CloseLossProb": np.nan,
                        "LossProbMove": np.nan,
                    }
                )
                continue

            rows.append(
                {
                    "Residual": match["Residual"],
                    "PreCloseWinProb": match["PreCloseWinProb"],
                    "CloseWinProb": match["CloseWinProb"],
                    "WinProbMove": match["WinProbMove"],
                    "PreCloseDrawProb": match["PreCloseDrawProb"],
                    "CloseDrawProb": match["CloseDrawProb"],
                    "DrawProbMove": match["DrawProbMove"],
                    "PreCloseLossProb": match["PreCloseLossProb"],
                    "CloseLossProb": match["CloseLossProb"],
                    "LossProbMove": match["LossProbMove"],
                }
            )

        prefix = f"MatchAhead_{horizon}_"

        values = pd.DataFrame(
            rows,
            index=result.index,
        )

        values = values.rename(
            columns={column: f"{prefix}{column}" for column in values.columns}
        )

        result = pd.concat(
            [result, values],
            axis=1,
        )

    return result


# ---------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------


def bootstrap_cluster_mean_ci(
    data: pd.DataFrame,
    value_column: str,
    cluster_columns: list[str],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a mean after aggregating within clusters."""

    cluster_values = (
        data.groupby(cluster_columns)[value_column].mean().dropna().to_numpy()
    )

    if len(cluster_values) == 0:
        return np.nan, np.nan

    if len(cluster_values) == 1:
        value = float(cluster_values[0])
        return value, value

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        cluster_values,
        size=(
            n_bootstrap,
            len(cluster_values),
        ),
        replace=True,
    )

    means = samples.mean(axis=1)

    return tuple(
        np.percentile(
            means,
            [2.5, 97.5],
        )
    )


# ---------------------------------------------------------------------
# Run response
# ---------------------------------------------------------------------


def summarize_run_response(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate post-run residual response."""

    rows = []

    for signal in [
        "positive",
        "negative",
    ]:

        subset = outcomes[outcomes["RunSignal"] == signal]

        for horizon in horizons:

            column = f"MatchAhead_{horizon}_Residual"

            valid = subset.dropna(subset=[column])

            if valid.empty:
                continue

            team_seasons = valid.groupby(
                [
                    "League",
                    "Season",
                    "Team",
                ],
                sort=False,
            )[column].mean()

            ci_lower, ci_upper = bootstrap_cluster_mean_ci(
                valid,
                column,
                [
                    "League",
                    "Season",
                    "Team",
                ],
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            rows.append(
                {
                    "RunSignal": signal,
                    "MatchAhead": horizon,
                    "N": len(valid),
                    "TeamSeasons": len(team_seasons),
                    "MeanResidual": valid[column].mean(),
                    "TeamSeasonMean": team_seasons.mean(),
                    "MedianResidual": valid[column].median(),
                    "Positive_%": (valid[column] > 0).mean(),
                    "CI_Lower": ci_lower,
                    "CI_Upper": ci_upper,
                    "CI_Excludes_Zero": (ci_lower > 0 or ci_upper < 0),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Prospective market movement
# ---------------------------------------------------------------------


def summarize_prospective_market_movement(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Test whether residual runs predict subsequent market movement.

    WinProbMove is:

        closing win probability
        minus
        pre-closing win probability

    Positive values therefore mean that the market became more
    optimistic about the team's chance of winning.

    This is prospective from the perspective of the run: the run is
    formed entirely from earlier matches, while market movement belongs
    to a future match.
    """

    rows = []

    cluster_cols = [
        "League",
        "Season",
        "Team",
    ]

    for signal in [
        "positive",
        "negative",
    ]:

        subset = outcomes[outcomes["RunSignal"] == signal]

        for horizon in horizons:

            movement_column = f"MatchAhead_{horizon}_WinProbMove"

            residual_column = f"MatchAhead_{horizon}_Residual"

            valid = subset.dropna(
                subset=[
                    movement_column,
                    residual_column,
                ]
            ).copy()

            if valid.empty:
                continue

            cluster_means = valid.groupby(
                cluster_cols,
                sort=False,
            )[movement_column].mean()

            ci_lower, ci_upper = bootstrap_cluster_mean_ci(
                valid,
                movement_column,
                cluster_cols,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            correlation = valid[movement_column].corr(valid[residual_column])

            rows.append(
                {
                    "RunSignal": signal,
                    "MatchAhead": horizon,
                    "N": len(valid),
                    "TeamSeasons": len(cluster_means),
                    "MeanWinProbMove": valid[movement_column].mean(),
                    "TeamSeasonMeanWinProbMove": (cluster_means.mean()),
                    "MeanAbsWinProbMove": valid[movement_column].abs().mean(),
                    "CorrelationWithResidual": (correlation),
                    "CI_Lower": ci_lower,
                    "CI_Upper": ci_upper,
                    "CI_Excludes_Zero": (ci_lower > 0 or ci_upper < 0),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Prospective market movement dataset
# ---------------------------------------------------------------------


def build_prospective_market_movement(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """
    Create a match-level dataset containing future market movement.

    This is the main dataset to use when looking for characteristics
    associated with teams whose future odds subsequently move.
    """

    rows = []

    base_columns = [
        "League",
        "Season",
        "Team",
        "RunID",
        "RunSignal",
        "StartMatch",
        "EndMatch",
        "Length",
        "EntrySignal",
        "MeanSignal",
        "PeakSignal",
    ]

    for horizon in horizons:

        columns = {
            "PreCloseWinProb": (f"MatchAhead_{horizon}_" "PreCloseWinProb"),
            "CloseWinProb": (f"MatchAhead_{horizon}_" "CloseWinProb"),
            "WinProbMove": (f"MatchAhead_{horizon}_" "WinProbMove"),
            "Residual": (f"MatchAhead_{horizon}_" "Residual"),
        }

        available = (
            outcomes[base_columns + list(columns.values())]
            .dropna(subset=list(columns.values()))
            .copy()
        )

        if available.empty:
            continue

        available = available.rename(
            columns={value: key for key, value in columns.items()}
        )

        available["MatchAhead"] = horizon

        rows.append(available)

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .sort_values(
            [
                "MatchAhead",
                "RunSignal",
                "League",
                "Season",
                "Team",
                "EndMatch",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Team-season aggregation
# ---------------------------------------------------------------------


def aggregate_team_seasons(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
    min_runs: int = 3,
) -> pd.DataFrame:
    """
    Aggregate future residuals to team-season level.

    A team-season must have at least min_runs observations.
    """

    rows = []

    group_cols = [
        "League",
        "Season",
        "Team",
        "RunSignal",
    ]

    for horizon in horizons:

        column = f"MatchAhead_{horizon}_Residual"

        valid = outcomes.dropna(subset=[column])

        if valid.empty:
            continue

        grouped = (
            valid.groupby(
                group_cols,
                sort=False,
            )[column]
            .agg(
                Runs="count",
                MeanResidual="mean",
                MedianResidual="median",
                PositiveRate=lambda x: (x > 0).mean(),
            )
            .reset_index()
        )

        grouped = grouped[grouped["Runs"] >= min_runs].copy()

        grouped["MatchAhead"] = horizon

        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .sort_values(
            [
                "MatchAhead",
                "MeanResidual",
            ],
            ascending=[
                True,
                False,
            ],
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
    horizons: tuple[int, ...] = (1, 2, 3, 5),
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test robustness across run definitions."""

    rows = []

    for window in windows:

        mean_column = f"ResidualMean_{window}"

        if mean_column not in df.columns:
            raise ValueError(f"{mean_column} not found.")

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

            outcomes = measure_run_outcomes(
                signalled,
                runs,
                horizons=horizons,
            )

            response = summarize_run_response(
                outcomes,
                horizons=horizons,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

            if response.empty:
                continue

            response = response.copy()

            response["Window"] = window
            response["Threshold"] = threshold

            rows.append(response)

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )[
            [
                "Window",
                "Threshold",
                "RunSignal",
                "MatchAhead",
                "N",
                "TeamSeasons",
                "MeanResidual",
                "TeamSeasonMean",
                "MedianResidual",
                "Positive_%",
                "CI_Lower",
                "CI_Upper",
                "CI_Excludes_Zero",
            ]
        ]
        .sort_values(
            [
                "MatchAhead",
                "RunSignal",
                "Window",
                "Threshold",
            ]
        )
        .reset_index(drop=True)
    )
