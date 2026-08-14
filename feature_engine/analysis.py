import numpy as np
import pandas as pd

GROUPS = ["League", "Season", "Team"]


# ---------------------------------------------------------------------
# Team history
# ---------------------------------------------------------------------


def add_rolling_features(df, windows=(3, 5)):
    """
    Add lagged rolling residual statistics.

    All rolling statistics use only matches BEFORE the current match.

    RollingMean:
        Mean residual over the previous N matches.

    RollingStd:
        Standard deviation of residuals over the previous N matches.

    RollingZ:
        RollingMean divided by RollingStd.

        This measures whether the team's recent residual performance
        has become unusually positive or negative relative to its own
        recent residual volatility.
    """

    df = df.copy()

    df = df.sort_values(GROUPS + ["Match"])

    group = df.groupby(GROUPS)["Residual"]

    for window in windows:

        # Previous N residuals only
        rolling_mean = group.transform(lambda x: x.shift(1).rolling(window).mean())

        rolling_std = group.transform(lambda x: x.shift(1).rolling(window).std())

        df[f"RollingMean_{window}"] = rolling_mean
        df[f"RollingStd_{window}"] = rolling_std

        # Recent residual performance relative to recent volatility.
        #
        # Importantly, this does NOT include the current match.
        df[f"RollingZ_{window}"] = rolling_mean / rolling_std

    return df


# ---------------------------------------------------------------------
# Lag relationships
# ---------------------------------------------------------------------


def lag_correlation(df, lag=1):
    """
    Calculate pooled residual autocorrelation at a given lag.

    The lag is calculated separately within each
    League / Season / Team group.
    """

    df = df.sort_values(GROUPS + ["Match"]).copy()

    previous = df.groupby(GROUPS)["Residual"].shift(lag)

    valid = df["Residual"].notna() & previous.notna()

    return df.loc[valid, "Residual"].corr(previous.loc[valid])


# ---------------------------------------------------------------------
# Signal → next-match outcome
# ---------------------------------------------------------------------


def add_next_residual(df):
    """
    Add the residual from the team's next match.

    The next match is determined separately within each
    League / Season / Team group.
    """

    df = df.sort_values(GROUPS + ["Match"]).copy()

    df["NextResidual"] = df.groupby(GROUPS)["Residual"].shift(-1)

    return df


def evaluate_thresholds(
    df,
    z_column="RollingZ_3",
    thresholds=(-1.25, -1.0, -0.75, 0.75, 1.0, 1.25),
):
    """
    Evaluate whether unusually positive/negative recent residual
    performance predicts the next residual.
    """

    df = add_next_residual(df)

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

    values = np.asarray(values, dtype=float)
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
        np.quantile(
            bootstrap_means,
            alpha / 2,
        ),
        np.quantile(
            bootstrap_means,
            1 - alpha / 2,
        ),
    )


def add_confidence_intervals(
    results,
    df,
    z_column="RollingZ_3",
    confidence=0.95,
):
    """
    Add bootstrap confidence intervals to threshold results.
    """

    df = add_next_residual(df)

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
