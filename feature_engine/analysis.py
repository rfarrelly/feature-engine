import numpy as np
import pandas as pd

GROUPS = ["League", "Season", "Team"]


# ---------------------------------------------------------------------
# Rolling team-history features
# ---------------------------------------------------------------------


def add_rolling_features(df, windows=(3, 5)):
    """
    Add lagged rolling residual statistics.

    Rolling statistics only use matches BEFORE the current match.
    """

    df = df.copy()
    df = df.sort_values(GROUPS + ["Match"])

    group = df.groupby(GROUPS)["Residual"]

    for window in windows:

        df[f"RollingMean_{window}"] = group.transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

        df[f"RollingStd_{window}"] = group.transform(
            lambda x: x.shift(1).rolling(window).std()
        )

        df[f"ResidualZ_{window}"] = (df["Residual"] - df[f"RollingMean_{window}"]) / df[
            f"RollingStd_{window}"
        ]

    return df


# ---------------------------------------------------------------------
# Lag relationships
# ---------------------------------------------------------------------


def lag_correlation(df, lag=1):
    """
    Calculate pooled residual autocorrelation at a given lag.
    """

    df = df.sort_values(GROUPS + ["Match"]).copy()

    previous = df.groupby(GROUPS)["Residual"].shift(lag)

    valid = df["Residual"].notna() & previous.notna()

    return df.loc[valid, "Residual"].corr(previous.loc[valid])


# ---------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------


def evaluate_thresholds(
    df,
    z_column="ResidualZ_3",
    thresholds=(-1.25, -1.0, -0.75, 0.75, 1.0, 1.25),
):
    """
    Evaluate whether unusually positive/negative current residual
    signals predict the next residual.
    """

    df = df.sort_values(GROUPS + ["Match"]).copy()

    df["NextResidual"] = df.groupby(GROUPS)["Residual"].shift(-1)

    results = []

    for threshold in thresholds:

        if threshold < 0:
            mask = df[z_column] < threshold
            signal = f"Z < {threshold}"
        else:
            mask = df[z_column] > threshold
            signal = f"Z > {threshold}"

        subset = df.loc[mask & df["NextResidual"].notna()]

        if subset.empty:
            continue

        results.append(
            {
                "Signal": signal,
                "Threshold": threshold,
                "N": len(subset),
                "Mean_NextResidual": subset["NextResidual"].mean(),
                "Median_NextResidual": subset["NextResidual"].median(),
                "Positive_NextResidual_%": (subset["NextResidual"] > 0).mean(),
                "Mean_CurrentSignal": subset[z_column].mean(),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------


def bootstrap_mean_ci(
    values,
    n_bootstrap=5000,
    confidence=0.95,
    random_state=42,
):
    """
    Bootstrap confidence interval for a sample mean.
    """

    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)

    bootstrap_means = np.array(
        [
            rng.choice(
                values,
                size=len(values),
                replace=True,
            ).mean()
            for _ in range(n_bootstrap)
        ]
    )

    alpha = 1 - confidence

    return (
        np.quantile(bootstrap_means, alpha / 2),
        np.quantile(bootstrap_means, 1 - alpha / 2),
    )


def add_confidence_intervals(
    results,
    df,
    z_column="ResidualZ_3",
    confidence=0.95,
):
    """
    Add bootstrap confidence intervals to threshold results.
    """

    df = df.sort_values(GROUPS + ["Match"]).copy()

    df["NextResidual"] = df.groupby(GROUPS)["Residual"].shift(-1)

    lower = []
    upper = []

    for _, row in results.iterrows():

        threshold = row["Threshold"]

        if threshold < 0:
            mask = df[z_column] < threshold
        else:
            mask = df[z_column] > threshold

        values = df.loc[
            mask & df["NextResidual"].notna(),
            "NextResidual",
        ]

        lo, hi = bootstrap_mean_ci(
            values,
            confidence=confidence,
        )

        lower.append(lo)
        upper.append(hi)

    results = results.copy()

    results["CI_Lower"] = lower
    results["CI_Upper"] = upper

    results["CI_Excludes_Zero"] = (results["CI_Lower"] > 0) | (results["CI_Upper"] < 0)

    return results


# ---------------------------------------------------------------------
# Episode detection
# ---------------------------------------------------------------------


def identify_episodes(
    df,
    z_column="ResidualZ_3",
    positive_threshold=1.25,
    negative_threshold=-1.25,
):
    """
    Identify short-term residual episodes for each team-season.

    A positive episode begins when the team's signal is above
    positive_threshold.

    A negative episode begins when the team's signal is below
    negative_threshold.

    Consecutive qualifying matches belong to the same episode.

    Each episode receives a unique EpisodeID within team-season.
    """

    df = df.copy()
    df = df.sort_values(GROUPS + ["Match"]).reset_index(drop=True)

    signal = df[z_column]

    df["EpisodeSignal"] = np.select(
        [
            signal > positive_threshold,
            signal < negative_threshold,
        ],
        [
            "positive",
            "negative",
        ],
        default=None,
    )

    # An episode starts whenever the signal changes into a
    # qualifying state or changes from positive to negative.
    previous_signal = df.groupby(GROUPS)["EpisodeSignal"].shift(1)

    episode_start = df["EpisodeSignal"].notna() & (
        previous_signal.isna() | (df["EpisodeSignal"] != previous_signal)
    )

    # Count episode starts separately inside each team-season.
    df["EpisodeNumber"] = (
        episode_start.astype(int).groupby([df[g] for g in GROUPS]).cumsum()
    )

    df["EpisodeID"] = np.where(
        df["EpisodeSignal"].notna(),
        df["EpisodeNumber"],
        np.nan,
    )

    return df


# ---------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------


def summarize_episodes(
    df,
    z_column="ResidualZ_3",
):
    """
    Collapse individual qualifying matches into one row per episode.

    The episode is the unit of analysis.
    """

    qualifying = df.loc[df["EpisodeSignal"].notna() & df["EpisodeID"].notna()].copy()

    if qualifying.empty:
        return pd.DataFrame()

    summary = qualifying.groupby(
        GROUPS + ["EpisodeID", "EpisodeSignal"],
        as_index=False,
    ).agg(
        StartMatch=("Match", "min"),
        EndMatch=("Match", "max"),
        Length=("Match", "size"),
        MeanSignal=(z_column, "mean"),
    )

    positive = qualifying["EpisodeSignal"] == "positive"
    negative = qualifying["EpisodeSignal"] == "negative"

    positive_peak = (
        qualifying.loc[positive]
        .groupby(GROUPS + ["EpisodeID"])[z_column]
        .max()
        .rename("PeakSignal")
        .reset_index()
    )

    negative_peak = (
        qualifying.loc[negative]
        .groupby(GROUPS + ["EpisodeID"])[z_column]
        .min()
        .rename("PeakSignal")
        .reset_index()
    )

    peaks = pd.concat(
        [positive_peak, negative_peak],
        ignore_index=True,
    )

    summary = summary.merge(
        peaks,
        on=GROUPS + ["EpisodeID"],
        how="left",
    )

    return summary


# ---------------------------------------------------------------------
# Forward episode outcomes
# ---------------------------------------------------------------------


def measure_episode_outcomes(
    df,
    episodes,
    horizons=(1, 2, 3, 5),
):
    """
    Measure residual performance following each episode.

    Forward outcomes begin AFTER the episode ends.

    For example, if an episode ends at Match 12:

        Horizon 1 = residual at Match 13
        Horizon 2 = residuals at Matches 13-14
        Horizon 3 = residuals at Matches 13-15
        Horizon 5 = residuals at Matches 13-17

    Cumulative residual is the sum over the available future matches.
    """

    df = df.sort_values(GROUPS + ["Match"]).copy()

    # Fast lookup of residuals by team-season-match.
    residual_lookup = df.set_index(GROUPS + ["Match"])["Residual"]

    records = []

    for _, episode in episodes.iterrows():

        base = {
            "League": episode["League"],
            "Season": episode["Season"],
            "Team": episode["Team"],
            "EpisodeID": episode["EpisodeID"],
            "EpisodeSignal": episode["EpisodeSignal"],
            "StartMatch": episode["StartMatch"],
            "EndMatch": episode["EndMatch"],
            "Length": episode["Length"],
            "PeakSignal": episode["PeakSignal"],
            "MeanSignal": episode["MeanSignal"],
        }

        future_values = []

        for horizon in horizons:

            match_number = episode["EndMatch"] + horizon

            key = (
                episode["League"],
                episode["Season"],
                episode["Team"],
                match_number,
            )

            value = residual_lookup.get(key, np.nan)

            future_values.append(value)

            base[f"Residual_{horizon}"] = value

        for horizon in horizons:

            values = np.array(
                future_values[:horizon],
                dtype=float,
            )

            values = values[~np.isnan(values)]

            if len(values) == 0:
                cumulative = np.nan
            else:
                cumulative = values.sum()

            base[f"CumulativeResidual_{horizon}"] = cumulative

        records.append(base)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Episode-level aggregate analysis
# ---------------------------------------------------------------------


def summarize_episode_performance(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
):
    """
    Aggregate episode outcomes.

    Each episode receives equal weight.

    Results are reported separately for positive and negative episodes.
    """

    results = []

    for signal in ["positive", "negative"]:

        subset = episode_outcomes[episode_outcomes["EpisodeSignal"] == signal]

        if subset.empty:
            continue

        for horizon in horizons:

            column = f"CumulativeResidual_{horizon}"

            values = subset[column].dropna()

            if values.empty:
                continue

            ci_lower, ci_upper = bootstrap_mean_ci(
                values,
                confidence=0.95,
            )

            results.append(
                {
                    "Signal": signal,
                    "Horizon": horizon,
                    "Episodes": len(values),
                    "MeanCumulativeResidual": values.mean(),
                    "MedianCumulativeResidual": values.median(),
                    "PositiveEpisodes_%": (values > 0).mean(),
                    "CI_Lower": ci_lower,
                    "CI_Upper": ci_upper,
                    "CI_Excludes_Zero": (ci_lower > 0 or ci_upper < 0),
                }
            )

    return pd.DataFrame(results)
