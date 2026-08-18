# residuals.py

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Match calculations
# ---------------------------------------------------------------------


def get_points(goals_for, goals_against) -> int:
    """Return league points from a match result."""

    if goals_for > goals_against:
        return 3

    if goals_for == goals_against:
        return 1

    return 0


def expected_points(
    p_win: pd.Series,
    p_draw: pd.Series,
) -> pd.Series:
    """Expected league points from win/draw probabilities."""

    return 3 * p_win + p_draw


# ---------------------------------------------------------------------
# Market calculations
# ---------------------------------------------------------------------


def get_no_vig_odds_multiway(
    odds,
    accuracy: int = 3,
) -> tuple[float, float, float]:
    """
    Remove bookmaker overround using the power transformation.
    """

    c = 1.0
    max_error = (10**-accuracy) / 2
    current_error = float("inf")

    while current_error > max_error:
        probabilities = [(1 / odd) ** c for odd in odds]

        f = -1 + sum(probabilities)

        f_dash = sum(
            probability * (-math.log(odd))
            for probability, odd in zip(probabilities, odds)
        )

        c -= f / f_dash

        total_probability = sum((1 / odd) ** c for odd in odds)

        current_error = abs(total_probability - 1)

    return tuple(odd**c for odd in odds)


def add_no_vig_market(
    df: pd.DataFrame,
    odds_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Add no-vig odds and probabilities for a three-way market.

    Example:
        prefix="PreClose"
        odds_columns=["B365H", "B365D", "B365A"]
    """

    df = df.copy()

    fair_odds = df[odds_columns].apply(
        lambda row: get_no_vig_odds_multiway(row.tolist()),
        axis=1,
        result_type="expand",
    )

    fair_odds.columns = [
        f"{prefix}NoVigH",
        f"{prefix}NoVigD",
        f"{prefix}NoVigA",
    ]

    probability_columns = [
        f"{prefix}NoVigPH",
        f"{prefix}NoVigPD",
        f"{prefix}NoVigPA",
    ]

    df[
        [
            f"{prefix}NoVigH",
            f"{prefix}NoVigD",
            f"{prefix}NoVigA",
        ]
    ] = fair_odds

    df[probability_columns] = (
        1
        / df[
            [
                f"{prefix}NoVigH",
                f"{prefix}NoVigD",
                f"{prefix}NoVigA",
            ]
        ]
    )

    return df


def calculate_market_probabilities(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate no-vig probabilities for both pre-closing and
    closing Bet365 three-way markets.
    """

    df = df.copy()

    df = add_no_vig_market(
        df,
        odds_columns=[
            "B365H",
            "B365D",
            "B365A",
        ],
        prefix="PreClose",
    )

    df = add_no_vig_market(
        df,
        odds_columns=[
            "B365CH",
            "B365CD",
            "B365CA",
        ],
        prefix="Close",
    )

    return df


# ---------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------


def add_residual(
    df: pd.DataFrame,
    definition: str = "points",
) -> pd.DataFrame:
    """Add the selected residual definition."""

    df = df.copy()

    if definition == "points":

        df["Residual"] = df["ActualPoints"] - df["ExpectedPoints"]

    elif definition == "win":

        actual_win = (df["ActualPoints"] == 3).astype(int)

        df["Residual"] = actual_win - df["PreCloseWinProb"]

    else:

        raise ValueError(f"Unknown residual definition: {definition}")

    return df


# ---------------------------------------------------------------------
# Team-match dataset
# ---------------------------------------------------------------------


def build_team_match_dataset(
    df: pd.DataFrame,
    residual_definition: str = "points",
) -> pd.DataFrame:
    """
    Convert match-level data into one observation per team per match.

    The residual is based on the pre-closing market.

    Both pre-closing and closing probabilities are retained so that
    future market movement can be measured without reconstructing
    the raw bookmaker data.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Expected points from pre-closing market
    # ---------------------------------------------------------------

    df["HomeEP"] = expected_points(
        df["PreCloseNoVigPH"],
        df["PreCloseNoVigPD"],
    )

    df["AwayEP"] = expected_points(
        df["PreCloseNoVigPA"],
        df["PreCloseNoVigPD"],
    )

    # ---------------------------------------------------------------
    # Actual points
    # ---------------------------------------------------------------

    df["HomePoints"] = [
        get_points(home, away)
        for home, away in zip(
            df["FTHG"],
            df["FTAG"],
        )
    ]

    df["AwayPoints"] = [
        get_points(away, home)
        for home, away in zip(
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
            "GoalsFor": df["FTHG"],
            "GoalsAgainst": df["FTAG"],
            "GoalDifference": (df["FTHG"] - df["FTAG"]),
            "PreCloseWinProb": df["PreCloseNoVigPH"],
            "PreCloseDrawProb": df["PreCloseNoVigPD"],
            "PreCloseLossProb": df["PreCloseNoVigPA"],
            "CloseWinProb": df["CloseNoVigPH"],
            "CloseDrawProb": df["CloseNoVigPD"],
            "CloseLossProb": df["CloseNoVigPA"],
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
            "GoalsFor": df["FTAG"],
            "GoalsAgainst": df["FTHG"],
            "GoalDifference": (df["FTAG"] - df["FTHG"]),
            "PreCloseWinProb": df["PreCloseNoVigPA"],
            "PreCloseDrawProb": df["PreCloseNoVigPD"],
            "PreCloseLossProb": df["PreCloseNoVigPH"],
            "CloseWinProb": df["CloseNoVigPA"],
            "CloseDrawProb": df["CloseNoVigPD"],
            "CloseLossProb": df["CloseNoVigPH"],
            "ExpectedPoints": df["AwayEP"],
            "ActualPoints": df["AwayPoints"],
        }
    )

    # ---------------------------------------------------------------
    # Combine
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

    team_df["Match"] = (
        team_df.groupby(
            [
                "League",
                "Season",
                "Team",
            ]
        ).cumcount()
        + 1
    )

    return add_residual(
        team_df,
        definition=residual_definition,
    )


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def build_residual_dataset(
    path: str | Path,
    residual_definition: str = "points",
) -> pd.DataFrame:
    """
    Load one football-data CSV and build the team-match dataset.
    """

    path = Path(path)

    df = pd.read_csv(path)

    required_odds = [
        "B365H",
        "B365D",
        "B365A",
        "B365CH",
        "B365CD",
        "B365CA",
    ]

    missing = [column for column in required_odds if column not in df.columns]

    if missing:
        raise ValueError(f"{path} is missing required odds columns: {missing}")

    df[required_odds] = df[required_odds].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = df.dropna(subset=required_odds).reset_index(drop=True)

    df = calculate_market_probabilities(df)

    return build_team_match_dataset(
        df,
        residual_definition=residual_definition,
    )
