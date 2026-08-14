import math
import pandas as pd

# ---------------------------------------------------------------------
# Match calculations
# ---------------------------------------------------------------------


def get_points(goals_for, goals_against):
    """Return league points from a match result."""

    if goals_for > goals_against:
        return 3

    if goals_for == goals_against:
        return 1

    return 0


def expected_points(p_win, p_draw):
    """Expected league points from win/draw probabilities."""

    return 3 * p_win + p_draw


# ---------------------------------------------------------------------
# Market calculations
# ---------------------------------------------------------------------


def get_no_vig_odds_multiway(odds, accuracy=3):
    """
    Convert bookmaker odds into fair odds using the power
    transformation used in the original analysis.
    """

    c = 1.0

    max_error = (10**-accuracy) / 2
    current_error = float("inf")

    while current_error > max_error:

        f = -1 + sum((1 / odd) ** c for odd in odds)

        f_dash = sum((1 / odd) ** c * (-math.log(odd)) for odd in odds)

        c -= f / f_dash

        total_probability = sum((1 / odd) ** c for odd in odds)

        current_error = abs(total_probability - 1)

    return tuple(odd**c for odd in odds)


def calculate_market_probabilities(df):
    """Add no-vig odds and probabilities to the match dataset."""

    df = df.copy()

    odds_columns = [
        "B365CH",
        "B365CD",
        "B365CA",
    ]

    fair_odds = df[odds_columns].apply(
        lambda row: get_no_vig_odds_multiway(row.tolist()),
        axis=1,
        result_type="expand",
    )

    fair_odds.columns = [
        "NoVigH",
        "NoVigD",
        "NoVigA",
    ]

    df[["NoVigH", "NoVigD", "NoVigA"]] = fair_odds

    df["NoVigPH"] = 1 / df["NoVigH"]
    df["NoVigPD"] = 1 / df["NoVigD"]
    df["NoVigPA"] = 1 / df["NoVigA"]

    return df


# ---------------------------------------------------------------------
# Residual definition
# ---------------------------------------------------------------------


def add_residual(df, definition="points"):
    """
    Add the chosen residual definition.

    points:
        ActualPoints - ExpectedPoints

    win:
        ActualWin - WinProbability
    """

    df = df.copy()

    if definition == "points":

        df["Residual"] = df["ActualPoints"] - df["ExpectedPoints"]

    elif definition == "win":

        actual_win = (df["ActualPoints"] == 3).astype(int)

        df["Residual"] = actual_win - df["WinProb"]

    else:

        raise ValueError(f"Unknown residual definition: {definition}")

    return df


# ---------------------------------------------------------------------
# Build team-match dataset
# ---------------------------------------------------------------------


def build_team_match_dataset(
    df,
    residual_definition="points",
):
    """
    Convert match-level data into a team-match dataset.

    Each match produces two observations:
    one for the home team and one for the away team.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    # ---------------------------------------------------------------
    # Match-level expected and actual points
    # ---------------------------------------------------------------

    df["HomeEP"] = expected_points(
        df["NoVigPH"],
        df["NoVigPD"],
    )

    df["AwayEP"] = expected_points(
        df["NoVigPA"],
        df["NoVigPD"],
    )

    df["HomePoints"] = [
        get_points(h, a)
        for h, a in zip(
            df["FTHG"],
            df["FTAG"],
        )
    ]

    df["AwayPoints"] = [
        get_points(a, h)
        for h, a in zip(
            df["FTHG"],
            df["FTAG"],
        )
    ]

    # ---------------------------------------------------------------
    # Home observations
    # ---------------------------------------------------------------

    home = pd.DataFrame(
        {
            "Date": df["Date"],
            "League": df["League"],
            "Season": df["Season"],
            "Team": df["HomeTeam"],
            "Opponent": df["AwayTeam"],
            "Venue": "home",
            "WinProb": df["NoVigPH"],
            "DrawProb": df["NoVigPD"],
            "LossProb": df["NoVigPA"],
            "ExpectedPoints": df["HomeEP"],
            "ActualPoints": df["HomePoints"],
        }
    )

    # ---------------------------------------------------------------
    # Away observations
    # ---------------------------------------------------------------

    away = pd.DataFrame(
        {
            "Date": df["Date"],
            "League": df["League"],
            "Season": df["Season"],
            "Team": df["AwayTeam"],
            "Opponent": df["HomeTeam"],
            "Venue": "away",
            "WinProb": df["NoVigPA"],
            "DrawProb": df["NoVigPD"],
            "LossProb": df["NoVigPH"],
            "ExpectedPoints": df["AwayEP"],
            "ActualPoints": df["AwayPoints"],
        }
    )

    # ---------------------------------------------------------------
    # Team-level dataset
    # ---------------------------------------------------------------

    team_df = pd.concat(
        [home, away],
        ignore_index=True,
    )

    team_df = team_df.sort_values(
        [
            "League",
            "Season",
            "Team",
            "Date",
        ]
    ).reset_index(drop=True)

    team_df["Match"] = team_df.groupby(["League", "Season", "Team"]).cumcount() + 1

    team_df = add_residual(
        team_df,
        definition=residual_definition,
    )

    return team_df


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


def build_residual_dataset(
    path,
    residual_definition="points",
):
    """
    Load raw football-data CSV and build
    the canonical team-match dataset.
    """

    df = pd.read_csv(path)

    odds_columns = [
        "B365CH",
        "B365CD",
        "B365CA",
    ]

    df[odds_columns] = df[odds_columns].astype(float)

    df = calculate_market_probabilities(df)

    return build_team_match_dataset(
        df,
        residual_definition=residual_definition,
    )
