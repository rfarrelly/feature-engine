import numpy as np
import pandas as pd

GROUPS = ["League", "Season", "Team"]


# ---------------------------------------------------------------------
# Rolling team-history features
# ---------------------------------------------------------------------


def add_rolling_features(df, windows=(3,)):
    """
    Add lagged rolling residual statistics.

    Rolling statistics use only matches before the current match,
    preventing look-ahead bias.
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
# Episode detection
# ---------------------------------------------------------------------


def identify_episodes(
    df,
    z_column="ResidualZ_3",
    positive_threshold=1.25,
    negative_threshold=-1.25,
):
    """
    Identify consecutive extreme-residual episodes for each team-season.

    Positive episode:
        signal > positive_threshold

    Negative episode:
        signal < negative_threshold

    Consecutive qualifying matches of the same sign belong to
    the same episode.
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

    previous_signal = df.groupby(GROUPS)["EpisodeSignal"].shift(1)

    episode_start = df["EpisodeSignal"].notna() & (
        previous_signal.isna() | (df["EpisodeSignal"] != previous_signal)
    )

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
    Collapse qualifying matches into one row per episode.

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

    positive_peak = (
        qualifying.loc[qualifying["EpisodeSignal"] == "positive"]
        .groupby(GROUPS + ["EpisodeID"])[z_column]
        .max()
        .rename("PeakSignal")
        .reset_index()
    )

    negative_peak = (
        qualifying.loc[qualifying["EpisodeSignal"] == "negative"]
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

    Forward outcomes begin after the episode ends.

    If an episode ends at Match 12:

        Horizon 1 = Match 13
        Horizon 2 = Matches 13-14
        Horizon 3 = Matches 13-15
        Horizon 5 = Matches 13-17

    Residual_N is the residual at exactly N matches ahead.

    CumulativeResidual_N is the sum of all available residuals
    from Match+1 through Match+N.
    """

    df = df.sort_values(GROUPS + ["Match"]).copy()

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

            values = np.asarray(
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
# Clustered bootstrap
# ---------------------------------------------------------------------


def cluster_bootstrap_mean_ci(
    data,
    value_column,
    cluster_columns=GROUPS,
    n_bootstrap=5000,
    confidence=0.95,
    random_state=42,
):
    """
    Cluster bootstrap confidence interval for a mean.

    Entire team-season clusters are resampled rather than individual
    episodes. This preserves dependence among episodes belonging to
    the same team-season.
    """

    data = data.loc[data[value_column].notna()].copy()

    if data.empty:
        return np.nan, np.nan

    clusters = data[cluster_columns].drop_duplicates().reset_index(drop=True)

    if clusters.empty:
        return np.nan, np.nan

    cluster_keys = list(clusters.itertuples(index=False, name=None))

    grouped = {
        key: group[value_column].to_numpy(dtype=float)
        for key, group in data.groupby(cluster_columns)
    }

    rng = np.random.default_rng(random_state)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        sampled_keys = rng.choice(
            len(cluster_keys),
            size=len(cluster_keys),
            replace=True,
        )

        sampled_values = []

        for index in sampled_keys:
            key = cluster_keys[index]
            sampled_values.extend(grouped[key])

        bootstrap_means[i] = np.mean(sampled_values)

    alpha = 1 - confidence

    return (
        np.quantile(
            bootstrap_means,
            alpha / 2,
        ),
        np.quantile(
            bootstrap_means,
            1 - alpha / 2,
        ),
    )


# ---------------------------------------------------------------------
# Clustered positive-vs-negative comparison
# ---------------------------------------------------------------------


def cluster_bootstrap_signal_difference(
    data,
    value_column,
    signal_column="EpisodeSignal",
    positive_label="positive",
    negative_label="negative",
    cluster_columns=GROUPS,
    n_bootstrap=5000,
    confidence=0.95,
    random_state=42,
):
    """
    Compare mean outcomes between negative and positive episodes
    using a team-season clustered bootstrap.

    Returns:

        negative mean - positive mean

    The difference is therefore positive when negative episodes
    have better subsequent residual performance than positive episodes.
    """

    data = data.loc[data[value_column].notna()].copy()

    positive = data[data[signal_column] == positive_label]

    negative = data[data[signal_column] == negative_label]

    if positive.empty or negative.empty:
        return np.nan, np.nan, np.nan

    clusters = data[cluster_columns].drop_duplicates().reset_index(drop=True)

    cluster_keys = list(clusters.itertuples(index=False, name=None))

    grouped_positive = {
        key: group[value_column].to_numpy(dtype=float)
        for key, group in positive.groupby(cluster_columns)
    }

    grouped_negative = {
        key: group[value_column].to_numpy(dtype=float)
        for key, group in negative.groupby(cluster_columns)
    }

    rng = np.random.default_rng(random_state)

    bootstrap_differences = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        sampled_indices = rng.choice(
            len(cluster_keys),
            size=len(cluster_keys),
            replace=True,
        )

        positive_values = []
        negative_values = []

        for index in sampled_indices:

            key = cluster_keys[index]

            if key in grouped_positive:
                positive_values.extend(grouped_positive[key])

            if key in grouped_negative:
                negative_values.extend(grouped_negative[key])

        if not positive_values or not negative_values:
            bootstrap_differences[i] = np.nan
        else:
            bootstrap_differences[i] = np.mean(negative_values) - np.mean(
                positive_values
            )

    bootstrap_differences = bootstrap_differences[~np.isnan(bootstrap_differences)]

    if len(bootstrap_differences) == 0:
        return np.nan, np.nan, np.nan

    observed_difference = negative[value_column].mean() - positive[value_column].mean()

    alpha = 1 - confidence

    lower = np.quantile(
        bootstrap_differences,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_differences,
        1 - alpha / 2,
    )

    return observed_difference, lower, upper


# ---------------------------------------------------------------------
# Episode performance
# ---------------------------------------------------------------------


def summarize_episode_performance(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
    n_bootstrap=5000,
    confidence=0.95,
):
    """
    Summarize forward episode performance.

    Confidence intervals use team-season clustered bootstrap.
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

            ci_lower, ci_upper = cluster_bootstrap_mean_ci(
                subset,
                value_column=column,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
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


# ---------------------------------------------------------------------
# Forward residual analysis
# ---------------------------------------------------------------------


def summarize_forward_residuals(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
):
    """
    Summarize residual at each exact match-ahead position.

    Unlike cumulative episode performance, this examines the residual
    at exactly Match+N.
    """

    results = []

    for signal in ["positive", "negative"]:

        subset = episode_outcomes[episode_outcomes["EpisodeSignal"] == signal]

        for horizon in horizons:

            column = f"Residual_{horizon}"

            values = subset[column].dropna()

            if values.empty:
                continue

            results.append(
                {
                    "Signal": signal,
                    "MatchAhead": horizon,
                    "N": len(values),
                    "MeanResidual": values.mean(),
                    "MedianResidual": values.median(),
                    "Positive_%": (values > 0).mean(),
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Direct signal comparison
# ---------------------------------------------------------------------


def compare_episode_signals(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
    n_bootstrap=5000,
    confidence=0.95,
):
    """
    Directly compare negative and positive episode outcomes.

    Difference = mean negative outcome - mean positive outcome.

    A positive difference means negative episodes are followed by
    better residual outcomes than positive episodes.
    """

    results = []

    for horizon in horizons:

        column = f"Residual_{horizon}"

        subset = episode_outcomes.loc[episode_outcomes[column].notna()].copy()

        if subset.empty:
            continue

        positive = subset[subset["EpisodeSignal"] == "positive"]

        negative = subset[subset["EpisodeSignal"] == "negative"]

        if positive.empty or negative.empty:
            continue

        difference, ci_lower, ci_upper = cluster_bootstrap_signal_difference(
            subset,
            value_column=column,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )

        results.append(
            {
                "MatchAhead": horizon,
                "PositiveMean": positive[column].mean(),
                "NegativeMean": negative[column].mean(),
                "NegativeMinusPositive": difference,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
                "CI_Excludes_Zero": (ci_lower > 0 or ci_upper < 0),
                "Positive_N": len(positive),
                "Negative_N": len(negative),
            }
        )

    return pd.DataFrame(results)
