import numpy as np
import pandas as pd

GROUPS = ["League", "Season", "Team"]


# ---------------------------------------------------------------------
# Rolling residual signal
# ---------------------------------------------------------------------


def add_rolling_features(df, windows=(3,)):
    """
    Calculate rolling residual statistics using only matches
    before the current match.
    """

    df = df.copy().sort_values(GROUPS + ["Match"])

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
# Episode identification
# ---------------------------------------------------------------------


def identify_episodes(
    df,
    z_column="ResidualZ_3",
    positive_threshold=1.25,
    negative_threshold=-1.25,
):
    """
    Identify consecutive runs of extreme positive or negative signals.

    A new episode begins when:
        - an extreme signal starts after no signal, or
        - the signal changes from positive to negative or vice versa.
    """

    df = df.copy().sort_values(GROUPS + ["Match"]).reset_index(drop=True)

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

    previous = df.groupby(GROUPS)["EpisodeSignal"].shift(1)

    episode_start = df["EpisodeSignal"].notna() & (
        previous.isna() | (df["EpisodeSignal"] != previous)
    )

    df["EpisodeNumber"] = (
        episode_start.astype(int).groupby([df[group] for group in GROUPS]).cumsum()
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


def summarize_episodes(df, z_column="ResidualZ_3"):
    """
    Create one row per extreme episode.
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
        qualifying[qualifying["EpisodeSignal"] == "positive"]
        .groupby(GROUPS + ["EpisodeID"])[z_column]
        .max()
        .rename("PeakSignal")
        .reset_index()
        .assign(EpisodeSignal="positive")
    )

    negative_peak = (
        qualifying[qualifying["EpisodeSignal"] == "negative"]
        .groupby(GROUPS + ["EpisodeID"])[z_column]
        .min()
        .rename("PeakSignal")
        .reset_index()
        .assign(EpisodeSignal="negative")
    )

    peaks = pd.concat(
        [positive_peak, negative_peak],
        ignore_index=True,
    )

    return summary.merge(
        peaks,
        on=GROUPS + ["EpisodeID", "EpisodeSignal"],
        how="left",
    )


# ---------------------------------------------------------------------
# Forward episode outcomes
# ---------------------------------------------------------------------


def measure_episode_outcomes(
    df,
    episodes,
    horizons=(1, 2, 3, 5),
):
    """
    Measure residuals after each episode.

    MatchAhead 1 means the first match after the episode ends.
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

            key = (
                episode["League"],
                episode["Season"],
                episode["Team"],
                episode["EndMatch"] + horizon,
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

            base[f"CumulativeResidual_{horizon}"] = (
                values.sum() if len(values) else np.nan
            )

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
    Bootstrap the mean while resampling whole team-season clusters.
    """

    data = data.loc[data[value_column].notna()].copy()

    if data.empty:
        return np.nan, np.nan

    clusters = data[cluster_columns].drop_duplicates()

    keys = list(
        clusters.itertuples(
            index=False,
            name=None,
        )
    )

    grouped = {
        key: group[value_column].to_numpy(dtype=float)
        for key, group in data.groupby(cluster_columns)
    }

    rng = np.random.default_rng(random_state)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        sampled = rng.choice(
            len(keys),
            size=len(keys),
            replace=True,
        )

        values = []

        for index in sampled:
            values.extend(grouped[keys[index]])

        bootstrap_means[i] = np.mean(values)

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
    Bootstrap the difference:

        negative mean - positive mean

    while resampling whole team-season clusters.
    """

    data = data.loc[data[value_column].notna()].copy()

    positive = data[data[signal_column] == positive_label]

    negative = data[data[signal_column] == negative_label]

    if positive.empty or negative.empty:
        return np.nan, np.nan, np.nan

    clusters = data[cluster_columns].drop_duplicates()

    keys = list(
        clusters.itertuples(
            index=False,
            name=None,
        )
    )

    grouped_positive = {
        key: group[value_column].to_numpy(dtype=float)
        for key, group in positive.groupby(cluster_columns)
    }

    grouped_negative = {
        key: group[value_column].to_numpy(dtype=float)
        for key, group in negative.groupby(cluster_columns)
    }

    rng = np.random.default_rng(random_state)

    differences = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        sampled = rng.choice(
            len(keys),
            size=len(keys),
            replace=True,
        )

        positive_values = []
        negative_values = []

        for index in sampled:

            key = keys[index]

            if key in grouped_positive:
                positive_values.extend(grouped_positive[key])

            if key in grouped_negative:
                negative_values.extend(grouped_negative[key])

        if positive_values and negative_values:

            differences[i] = np.mean(negative_values) - np.mean(positive_values)

        else:

            differences[i] = np.nan

    differences = differences[~np.isnan(differences)]

    if len(differences) == 0:
        return np.nan, np.nan, np.nan

    observed = negative[value_column].mean() - positive[value_column].mean()

    alpha = 1 - confidence

    return (
        observed,
        np.quantile(
            differences,
            alpha / 2,
        ),
        np.quantile(
            differences,
            1 - alpha / 2,
        ),
    )


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
    Summarise cumulative residual performance after episodes.
    """

    results = []

    for signal in ["positive", "negative"]:

        subset = episode_outcomes[episode_outcomes["EpisodeSignal"] == signal]

        for horizon in horizons:

            column = f"CumulativeResidual_{horizon}"

            values = subset[column].dropna()

            if values.empty:
                continue

            ci_lower, ci_upper = cluster_bootstrap_mean_ci(
                subset,
                column,
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
# Individual forward residuals
# ---------------------------------------------------------------------


def summarize_forward_residuals(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
):
    """
    Summarise the individual residual at each match after an episode.
    """

    results = []

    for signal in ["positive", "negative"]:

        subset = episode_outcomes[episode_outcomes["EpisodeSignal"] == signal]

        for horizon in horizons:

            values = subset[f"Residual_{horizon}"].dropna()

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
# Positive vs negative comparison
# ---------------------------------------------------------------------


def compare_episode_signals(
    episode_outcomes,
    horizons=(1, 2, 3, 5),
    n_bootstrap=5000,
    confidence=0.95,
):
    """
    Compare negative and positive episodes directly.

    Difference = NegativeMean - PositiveMean.
    """

    results = []

    for horizon in horizons:

        column = f"Residual_{horizon}"

        subset = episode_outcomes.loc[episode_outcomes[column].notna()].copy()

        positive = subset[subset["EpisodeSignal"] == "positive"]

        negative = subset[subset["EpisodeSignal"] == "negative"]

        if positive.empty or negative.empty:
            continue

        (
            difference,
            ci_lower,
            ci_upper,
        ) = cluster_bootstrap_signal_difference(
            subset,
            column,
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
