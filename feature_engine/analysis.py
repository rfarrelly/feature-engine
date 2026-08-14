import numpy as np
import pandas as pd

GROUPS = ["League", "Season", "Team"]


def add_rolling_features(df, windows=(3, 5)):
    """
    Add lagged rolling residual statistics.

    Rolling statistics only use matches BEFORE the current match.
    """

    df = df.copy()
    df = df.sort_values(GROUPS + ["Match"])

    group = df.groupby(GROUPS)["Residual"]

    for window in windows:

        # Recent residual level entering the current match
        df[f"RollingMean_{window}"] = group.transform(
            lambda x: x.shift(1).rolling(window).mean()
        )

        # Recent residual volatility entering the current match
        df[f"RollingStd_{window}"] = group.transform(
            lambda x: x.shift(1).rolling(window).std()
        )

        # Current residual relative to recent residual distribution
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

    df = df.sort_values(["League", "Season", "Team", "Match"]).copy()

    previous = df.groupby(["League", "Season", "Team"])["Residual"].shift(lag)

    valid = df["Residual"].notna() & previous.notna()

    return df.loc[valid, "Residual"].corr(previous.loc[valid])


# ---------------------------------------------------------------------
# Signal → next-match outcome
# ---------------------------------------------------------------------


def evaluate_thresholds(
    df,
    z_column="ResidualZ_3",
    thresholds=(-1.25, -1.0, -0.75, 0.75, 1.0, 1.25),
):
    """
    Evaluate whether unusually positive/negative recent residuals
    predict the next residual.
    """

    df = df.sort_values(["League", "Season", "Team", "Match"]).copy()

    df["NextResidual"] = df.groupby(["League", "Season", "Team"])["Residual"].shift(-1)

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
                "Mean_CurrentZ": subset[z_column].mean(),
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

    df = df.sort_values(["League", "Season", "Team", "Match"]).copy()

    df["NextResidual"] = df.groupby(["League", "Season", "Team"])["Residual"].shift(-1)

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
