import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from residuals import build_residual_dataset

DATA_DIR = Path("/Users/ryanfarrelly/Desktop/collector/DATA/football-data")

OUTPUT_DIR = Path("residuals")

# ---------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------


def filter_data(
    df,
    league=None,
    seasons=None,
    teams=None,
    venue=None,
):
    """
    Flexible filtering for analysis.
    """

    result = df.copy()

    if league is not None:
        result = result[result["League"] == league]

    if seasons is not None:
        if not isinstance(seasons, (list, tuple, set)):
            seasons = [seasons]

        result = result[result["Season"].isin(seasons)]

    if teams is not None:
        if not isinstance(teams, (list, tuple, set)):
            teams = [teams]

        result = result[result["Team"].isin(teams)]

    if venue is not None:
        if not isinstance(venue, (list, tuple, set)):
            venue = [venue]

        result = result[result["Venue"].isin(venue)]

    return result.copy()


# ---------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------


def residual_summary(df, group_by=None):
    """
    Summary statistics for residuals.

    group_by can be:
        None
        "Team"
        "Venue"
        "Season"
        "League"
        ["League", "Season"]
        etc.
    """

    if group_by is None:
        return pd.DataFrame(
            {
                "Mean": [df["Residual"].mean()],
                "Median": [df["Residual"].median()],
                "Std": [df["Residual"].std()],
                "Min": [df["Residual"].min()],
                "Max": [df["Residual"].max()],
                "N": [df["Residual"].count()],
            }
        )

    return (
        df.groupby(group_by)["Residual"]
        .agg(
            Mean="mean",
            Median="median",
            Std="std",
            Min="min",
            Max="max",
            N="count",
        )
        .reset_index()
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_team_residuals(
    df,
    team,
    league=None,
    season=None,
    venue=None,
):
    """
    Plot a team's residuals chronologically.
    """

    team_df = filter_data(
        df,
        league=league,
        seasons=season,
        teams=team,
        venue=venue,
    )

    if team_df.empty:
        raise ValueError(f"No data found for {team}")

    team_df = team_df.sort_values("Date").reset_index(drop=True)

    plt.figure(figsize=(12, 5))

    plt.plot(
        range(1, len(team_df) + 1),
        team_df["Residual"],
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.title(f"{team} — Residuals")

    plt.xlabel("Match")
    plt.ylabel("Residual")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------


def autocorrelation_by_group(
    df,
    group_by="Team",
    max_lag=5,
):
    """
    Calculate autocorrelation separately for each group.

    Examples:
        group_by="Team"
        group_by=["League", "Team"]
        group_by=["League", "Season", "Team"]
    """

    if isinstance(group_by, str):
        group_by = [group_by]

    results = []

    for group_values, group_df in df.groupby(group_by):

        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_df = group_df.sort_values("Date")

        row = dict(zip(group_by, group_values))

        for lag in range(1, max_lag + 1):
            row[f"Lag_{lag}"] = group_df["Residual"].autocorr(lag=lag)

        results.append(row)

    return pd.DataFrame(results)


def pooled_autocorrelation(
    df,
    max_lag=5,
):
    """
    Calculate pooled autocorrelation across all team sequences.

    Lags are generated within each Team/Season combination.
    """

    df = df.sort_values(["League", "Season", "Team", "Date"]).copy()

    results = []

    for lag in range(1, max_lag + 1):

        df[f"Lag_{lag}"] = df.groupby(["League", "Season", "Team"])["Residual"].shift(
            lag
        )

        valid = df[["Residual", f"Lag_{lag}"]].dropna()

        correlation = valid["Residual"].corr(valid[f"Lag_{lag}"])

        results.append(
            {
                "Lag": lag,
                "Correlation": correlation,
                "N": len(valid),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Lagged dataset
# ---------------------------------------------------------------------


def create_lagged_residuals(
    df,
    max_lag=5,
):
    """
    Create lagged residual columns within each
    league / season / team sequence.
    """

    result = df.sort_values(["League", "Season", "Team", "Date"]).copy()

    group = result.groupby(["League", "Season", "Team"])["Residual"]

    for lag in range(1, max_lag + 1):
        result[f"Residual_Lag_{lag}"] = group.shift(lag)

    return result


def nonlinear_residual_analysis(
    df,
    lag=1,
    bins=5,
):
    """
    Examine whether the next residual depends
    nonlinearly on the previous residual.
    """

    data = df.sort_values(["League", "Season", "Team", "Date"]).copy()

    data["PreviousResidual"] = data.groupby(["League", "Season", "Team"])[
        "Residual"
    ].shift(lag)

    data = data.dropna(subset=["PreviousResidual", "Residual"]).copy()

    data["ResidualBin"] = pd.qcut(
        data["PreviousResidual"],
        q=bins,
        duplicates="drop",
    )

    result = (
        data.groupby("ResidualBin", observed=True)["Residual"]
        .agg(
            MeanNextResidual="mean",
            MedianNextResidual="median",
            N="count",
        )
        .reset_index()
    )

    return result


def residual_magnitude_analysis(
    df,
    lag=1,
    bins=5,
):
    data = create_lagged_residuals(
        df,
        max_lag=lag,
    )

    lag_column = f"Residual_Lag_{lag}"

    data = data.dropna(subset=[lag_column, "Residual"]).copy()

    data["PreviousMagnitude"] = data[lag_column].abs()

    data["MagnitudeBin"] = pd.qcut(
        data["PreviousMagnitude"],
        q=bins,
        duplicates="drop",
    )

    return (
        data.groupby("MagnitudeBin", observed=True)["Residual"]
        .agg(
            MeanNextResidual="mean",
            MedianNextResidual="median",
            N="count",
        )
        .reset_index()
    )


from pathlib import Path
import pandas as pd


def load_all_residuals(directory="residuals"):
    files = Path(directory).glob("*/*.csv")

    frames = [pd.read_csv(file) for file in files]

    return pd.concat(
        frames,
        ignore_index=True,
    )


def process_all_leagues():
    for csv_file in DATA_DIR.glob("*/*.csv"):

        league = csv_file.parent.name
        season = csv_file.stem

        print(f"Processing {league} — {season}")

        result = build_residual_dataset(csv_file)

        output_dir = OUTPUT_DIR / league
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = output_dir / f"{season}.csv"

        result.to_csv(
            output_path,
            index=False,
        )


if __name__ == "__main__":
    process_all_leagues()
    df = load_all_residuals()

    bundesliga = filter_data(
        df,
        league="Bundesliga",
    )

    bundesliga_2526 = filter_data(
        df,
        league="Bundesliga",
        seasons=2526,
    )

    home = filter_data(
        df,
        venue="home",
    )

    greece = filter_data(
        df,
        league="Super-League-Greece",
    )

    bayern = filter_data(
        df,
        league="Bundesliga",
        teams="Bayern Munich",
    )
    breakpoint()
