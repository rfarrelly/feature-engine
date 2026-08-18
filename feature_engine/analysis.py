# analysis.py

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_COLUMNS = ["League", "Season", "Team"]


# ---------------------------------------------------------------------
# Rolling residual history
# ---------------------------------------------------------------------


def add_rolling_residuals(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 5, 8),
) -> pd.DataFrame:
    """Add trailing residual means using only matches before the current one."""

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(GROUP_COLUMNS + ["Date", "Match"]).reset_index(drop=True)

    grouped = df.groupby(GROUP_COLUMNS, sort=False)["Residual"]

    for window in windows:
        df[f"ResidualMean_{window}"] = grouped.transform(
            lambda values: values.shift(1)
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
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
    """Identify positive and negative residual runs from prior history."""

    signal_column = f"ResidualMean_{window}"

    if signal_column not in df.columns:
        raise ValueError(
            f"{signal_column} not found. Run add_rolling_residuals() first."
        )

    df = df.copy()

    df["RunSignal"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    df.loc[df[signal_column] >= threshold, "RunSignal"] = "positive"
    df.loc[df[signal_column] <= -threshold, "RunSignal"] = "negative"

    previous = df.groupby(
        GROUP_COLUMNS,
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

    required = [
        *GROUP_COLUMNS,
        "Match",
        "RunID",
        "RunSignal",
        f"ResidualMean_{window}",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[df["RunID"].notna()].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                *GROUP_COLUMNS,
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

    group_columns = [*GROUP_COLUMNS, "RunID"]
    signal = f"ResidualMean_{window}"

    grouped = data.groupby(group_columns, sort=False)

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
        .apply(
            lambda values: (
                values.loc[values.abs().idxmax()] if values.notna().any() else np.nan
            )
        )
        .rename("PeakSignal")
        .reset_index()
    )

    return result.merge(
        peak,
        on=group_columns,
        how="left",
    )


# ---------------------------------------------------------------------
# Subsequent match outcomes
# ---------------------------------------------------------------------


def measure_run_outcomes(
    df: pd.DataFrame,
    runs: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """
    Attach subsequent match outcomes and market movement.

    The residual is based on the pre-closing market. For each subsequent
    match we also retain the pre-closing win probability, closing win
    probability, and the movement between them.

    MatchAhead=1 is the first match after the run.
    """

    if runs.empty:
        return runs.copy()

    required = [
        *GROUP_COLUMNS,
        "Match",
        "Residual",
        "PreCloseWinProb",
        "CloseWinProb",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Residual dataset is missing required analysis columns: " f"{missing}"
        )

    history = df[required].sort_values(GROUP_COLUMNS + ["Match"]).copy()

    history["WinProbMove"] = history["CloseWinProb"] - history["PreCloseWinProb"]

    lookup = history.set_index(GROUP_COLUMNS + ["Match"])

    result = runs.copy()

    for horizon in horizons:
        values = []

        for row in result.itertuples(index=False):
            key = (
                row.League,
                row.Season,
                row.Team,
            )
            target = int(row.EndMatch) + horizon

            try:
                match = lookup.loc[(*key, target)]
            except KeyError:
                values.append(
                    {
                        f"Residual_{horizon}": np.nan,
                        f"PreCloseWinProb_{horizon}": np.nan,
                        f"CloseWinProb_{horizon}": np.nan,
                        f"WinProbMove_{horizon}": np.nan,
                    }
                )
                continue

            values.append(
                {
                    f"Residual_{horizon}": match["Residual"],
                    f"PreCloseWinProb_{horizon}": match["PreCloseWinProb"],
                    f"CloseWinProb_{horizon}": match["CloseWinProb"],
                    f"WinProbMove_{horizon}": match["WinProbMove"],
                }
            )

        result = pd.concat(
            [
                result,
                pd.DataFrame(values, index=result.index),
            ],
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
    """Bootstrap a mean after first averaging within clusters."""

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
        size=(n_bootstrap, len(cluster_values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return tuple(np.percentile(means, [2.5, 97.5]))


# ---------------------------------------------------------------------
# Overall run response
# ---------------------------------------------------------------------


def summarize_run_response(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate post-run residual response with team-season clustering."""

    rows = []

    for signal in ("positive", "negative"):
        subset = outcomes[outcomes["RunSignal"] == signal]

        for horizon in horizons:
            column = f"Residual_{horizon}"
            valid = subset.dropna(subset=[column])

            if valid.empty:
                continue

            team_seasons = valid.groupby(
                GROUP_COLUMNS,
                sort=False,
            )[column].mean()

            ci_lower, ci_upper = bootstrap_cluster_mean_ci(
                valid,
                column,
                GROUP_COLUMNS,
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
# Market movement response
# ---------------------------------------------------------------------


def summarize_market_movement_response(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Test whether pre-close to close win-probability movement predicts
    the subsequent pre-close residual, conditional on run direction.

    Predictor:
        CloseWinProb - PreCloseWinProb

    Outcome:
        subsequent Residual, based on the subsequent pre-closing market.
    """

    rows = []

    for signal in ("positive", "negative"):
        subset = outcomes[outcomes["RunSignal"] == signal]

        for horizon in horizons:
            residual_column = f"Residual_{horizon}"
            predictor_column = f"WinProbMove_{horizon}"

            valid = subset[
                [
                    residual_column,
                    predictor_column,
                    *GROUP_COLUMNS,
                ]
            ].dropna()

            if len(valid) < 2:
                continue

            x = valid[predictor_column].to_numpy()
            y = valid[residual_column].to_numpy()

            if np.std(x) == 0:
                continue

            slope = np.polyfit(x, y, 1)[0]
            correlation = np.corrcoef(x, y)[0, 1]

            cluster_frame = (
                valid.groupby(GROUP_COLUMNS, sort=False)[
                    [predictor_column, residual_column]
                ]
                .mean()
                .dropna()
            )

            ci_lower = np.nan
            ci_upper = np.nan

            if len(cluster_frame) >= 2:
                rng = np.random.default_rng(seed)
                x_cluster = cluster_frame[predictor_column].to_numpy()
                y_cluster = cluster_frame[residual_column].to_numpy()

                indices = rng.integers(
                    0,
                    len(cluster_frame),
                    size=(n_bootstrap, len(cluster_frame)),
                )

                bootstrap_slopes = []

                for sample in indices:
                    sample_x = x_cluster[sample]
                    sample_y = y_cluster[sample]

                    if np.std(sample_x) == 0:
                        continue

                    bootstrap_slopes.append(np.polyfit(sample_x, sample_y, 1)[0])

                if bootstrap_slopes:
                    ci_lower, ci_upper = np.percentile(
                        bootstrap_slopes,
                        [2.5, 97.5],
                    )

            rows.append(
                {
                    "RunSignal": signal,
                    "MatchAhead": horizon,
                    "N": len(valid),
                    "TeamSeasons": len(cluster_frame),
                    "MeanWinProbMove": x.mean(),
                    "MeanAbsResidual": np.abs(y).mean(),
                    "Slope": slope,
                    "Correlation": correlation,
                    "CI_Lower": ci_lower,
                    "CI_Upper": ci_upper,
                    "CI_Excludes_Zero": (
                        not pd.isna(ci_lower) and (ci_lower > 0 or ci_upper < 0)
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Team-season aggregation
# ---------------------------------------------------------------------


def aggregate_team_seasons(
    outcomes: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5),
    min_runs: int = 3,
) -> pd.DataFrame:
    """Aggregate subsequent residuals to the team-season level."""

    rows = []
    group_columns = [*GROUP_COLUMNS, "RunSignal"]

    for horizon in horizons:
        column = f"Residual_{horizon}"
        valid = outcomes.dropna(subset=[column])

        if valid.empty:
            continue

        grouped = (
            valid.groupby(group_columns, sort=False)[column]
            .agg(
                Runs="count",
                MeanResidual="mean",
                MedianResidual="median",
                PositiveRate=lambda values: (values > 0).mean(),
            )
            .reset_index()
        )

        grouped = grouped[grouped["Runs"] >= min_runs].copy()
        grouped["MatchAhead"] = horizon
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(
            ["MatchAhead", "MeanResidual"],
            ascending=[True, False],
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
    """Test whether the run-response result is robust to run definitions."""

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
        pd.concat(rows, ignore_index=True)[
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
        .sort_values(["MatchAhead", "RunSignal", "Window", "Threshold"])
        .reset_index(drop=True)
    )
