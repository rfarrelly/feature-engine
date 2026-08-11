import math
import pandas as pd

# ---------------------------------------------------------------------
# Basic match calculations
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


def residual(actual_points, expected_points):
    """Actual points minus expected points."""
    return actual_points - expected_points


# ---------------------------------------------------------------------
# Market calculations
# ---------------------------------------------------------------------


def get_no_vig_odds_multiway(odds, accuracy=3):
    """
    Convert bookmaker odds into no-vig/fair odds using the
    Shin-style power transformation used in the original code.
    """
    c = 1.0
    target_overround = 0.0
    max_error = (10 ** (-accuracy)) / 2
    current_error = float("inf")

    while current_error > max_error:
        f = -1 - target_overround

        for odd in odds:
            f += (1 / odd) ** c

        f_dash = sum((1 / odd) ** c * (-math.log(odd)) for odd in odds)

        h = -f / f_dash
        c += h

        total_probability = sum((1 / odd) ** c for odd in odds)

        current_error = abs(total_probability - 1 - target_overround)

    return tuple(round(odd**c, 6) for odd in odds)


def calculate_market_probabilities(df):
    """
    Add no-vig odds and probabilities to the match-level DataFrame.
    """

    odds = df[["B365CH", "B365CD", "B365CA"]]

    fair_odds = odds.apply(
        lambda row: get_no_vig_odds_multiway(row.tolist()),
        axis=1,
        result_type="expand",
    )

    fair_odds.columns = ["NoVigH", "NoVigD", "NoVigA"]

    df = df.copy()
    df[["NoVigH", "NoVigD", "NoVigA"]] = fair_odds

    df["NoVigPH"] = 1 / df["NoVigH"]
    df["NoVigPD"] = 1 / df["NoVigD"]
    df["NoVigPA"] = 1 / df["NoVigA"]

    return df


# ---------------------------------------------------------------------
# Match-level dataset
# ---------------------------------------------------------------------


def prepare_matches(df):
    """
    Prepare the raw match-level DataFrame.

    Does not filter by league, season, team or venue.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["HomeEP"] = expected_points(
        df["NoVigPH"],
        df["NoVigPD"],
    )

    df["AwayEP"] = expected_points(
        df["NoVigPA"],
        df["NoVigPD"],
    )

    df["HomePoints"] = [get_points(hg, ag) for hg, ag in zip(df["FTHG"], df["FTAG"])]

    df["AwayPoints"] = [get_points(ag, hg) for hg, ag in zip(df["FTHG"], df["FTAG"])]

    df["HomeGoalDifference"] = df["FTHG"] - df["FTAG"]
    df["AwayGoalDifference"] = df["FTAG"] - df["FTHG"]

    df["HomeResidual"] = df["HomePoints"] - df["HomeEP"]

    df["AwayResidual"] = df["AwayPoints"] - df["AwayEP"]

    return df


# ---------------------------------------------------------------------
# Team-match dataset
# ---------------------------------------------------------------------


def build_team_match_dataset(df):
    """
    Convert each match into two team-level observations.

    One row = one team's experience of one match.

    This is the main dataset used by the analysis module.
    """

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
            "GoalDifference": df["HomeGoalDifference"],
            "GoalsFor": df["FTHG"],
            "GoalsAgainst": df["FTAG"],
            "Residual": df["HomeResidual"],
        }
    )

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
            "GoalDifference": df["AwayGoalDifference"],
            "GoalsFor": df["FTAG"],
            "GoalsAgainst": df["FTHG"],
            "Residual": df["AwayResidual"],
        }
    )

    team_df = pd.concat(
        [home, away],
        ignore_index=True,
    )

    team_df = team_df.sort_values(["League", "Season", "Team", "Date"]).reset_index(
        drop=True
    )

    team_df["Match"] = team_df.groupby(["League", "Season", "Team"]).cumcount() + 1

    team_df["ResBin"] = (team_df["Residual"] > 0).astype(int).replace({0: -1})

    team_df["AbsResidual"] = team_df["Residual"].abs()

    return team_df


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


def add_residual_definition(df, name="points"):
    """
    Add a residual column according to the requested definition.
    """

    df = df.copy()

    if name == "points":
        df["Residual"] = df["ActualPoints"] - df["ExpectedPoints"]

    elif name == "win":
        actual_win = (df["ActualPoints"] == 3).astype(int)

        df["Residual"] = actual_win - df["WinProb"]

    else:
        raise ValueError(f"Unknown residual definition: {name}")

    return df


def build_residual_dataset(path):
    """
    Load a raw football-data CSV and return the canonical
    team-match residual dataset.
    """

    df = pd.read_csv(path)
    df[["B365CH", "B365CD", "B365CA"]] = df[["B365CH", "B365CD", "B365CA"]].astype(
        float
    )

    df = calculate_market_probabilities(df)
    df = prepare_matches(df)
    df = build_team_match_dataset(df)
    df = add_residual_definition(df, "win")

    return df


df = build_residual_dataset(
    "/Users/ryanfarrelly/Desktop/collector/DATA/football-data/Super-League-Greece/Super-League-Greece_2122.csv"
)
