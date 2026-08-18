# residuals.py

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

GROUP_COLUMNS = ["League", "Season", "Team"]


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


def expected_points(p_win, p_draw):
    """Expected league points from win/draw probabilities."""

    return 3 * p_win + p_draw


# ---------------------------------------------------------------------
# Market calculations
# ---------------------------------------------------------------------


def get_no_vig_odds_multiway(
    odds,
    accuracy: int = 3,
):
    """Remove bookmaker overround using the power transformation."""

    c = 1.0
    max_error = (10**-accuracy) / 2

    while True:
        probabilities = [(1 / odd) ** c for odd in odds]
        f = -1 + sum(probabilities)

        f_dash = sum(
            probability * (-math.log(odd))
            for probability, odd in zip(probabilities, odds)
        )

        c -= f / f_dash

        total_probability = sum((1 / odd) ** c for odd in odds)

        if abs(total_probability - 1) <= max_error:
            break

    return tuple(odd**c for odd in odds)


def add_no_vig_market(
    df: pd.DataFrame,
    odds_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Add no-vig odds and probabilities for a three-way market."""

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

    df[
        [
            f"{prefix}NoVigH",
            f"{prefix}NoVigD",
            f"{prefix}NoVigA",
        ]
    ] = fair_odds

    df[f"{prefix}NoVigPH"] = 1 / df[f"{prefix}NoVigH"]
    df[f"{prefix}NoVigPD"] = 1 / df[f"{prefix}NoVigD"]
    df[f"{prefix}NoVigPA"] = 1 / df[f"{prefix}NoVigA"]

    return df


def calculate_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate no-vig probabilities for pre-closing and closing markets."""

    df = df.copy()

    df = add_no_vig_market(
        df,
        ["B365H", "B365D", "B365A"],
        "PreClose",
    )

    df = add_no_vig_market(
        df,
        ["B365CH", "B365CD", "B365CA"],
        "Close",
    )

    return df


# ---------------------------------------------------------------------
# Team-match dataset
# ---------------------------------------------------------------------


def build_team_match_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level data into one observation per team per match.

    Expected points and the residual use pre-closing probabilities.
    Closing probabilities are retained only for measuring market movement.
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["HomeExpectedPoints"] = expected_points(
        df["PreCloseNoVigPH"],
        df["PreCloseNoVigPD"],
    )

    df["AwayExpectedPoints"] = expected_points(
        df["PreCloseNoVigPA"],
        df["PreCloseNoVigPD"],
    )

    df["HomePoints"] = [
        get_points(home, away) for home, away in zip(df["FTHG"], df["FTAG"])
    ]

    df["AwayPoints"] = [
        get_points(away, home) for home, away in zip(df["FTHG"], df["FTAG"])
    ]

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
            "GoalDifference": df["FTHG"] - df["FTAG"],
            "PreCloseWinProb": df["PreCloseNoVigPH"],
            "PreCloseDrawProb": df["PreCloseNoVigPD"],
            "PreCloseLossProb": df["PreCloseNoVigPA"],
            "CloseWinProb": df["CloseNoVigPH"],
            "CloseDrawProb": df["CloseNoVigPD"],
            "CloseLossProb": df["CloseNoVigPA"],
            "ExpectedPoints": df["HomeExpectedPoints"],
            "ActualPoints": df["HomePoints"],
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
            "GoalsFor": df["FTAG"],
            "GoalsAgainst": df["FTHG"],
            "GoalDifference": df["FTAG"] - df["FTHG"],
            "PreCloseWinProb": df["PreCloseNoVigPA"],
            "PreCloseDrawProb": df["PreCloseNoVigPD"],
            "PreCloseLossProb": df["PreCloseNoVigPH"],
            "CloseWinProb": df["CloseNoVigPA"],
            "CloseDrawProb": df["CloseNoVigPD"],
            "CloseLossProb": df["CloseNoVigPH"],
            "ExpectedPoints": df["AwayExpectedPoints"],
            "ActualPoints": df["AwayPoints"],
        }
    )

    team_df = pd.concat([home, away], ignore_index=True)

    team_df = team_df.sort_values(GROUP_COLUMNS + ["Date"]).reset_index(drop=True)

    team_df["Match"] = team_df.groupby(GROUP_COLUMNS).cumcount() + 1

    team_df["Residual"] = team_df["ActualPoints"] - team_df["ExpectedPoints"]

    return team_df


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def build_residual_dataset(path: Path) -> pd.DataFrame:
    """Load one raw football-data CSV and build the residual dataset."""

    df = pd.read_csv(path)

    odds_columns = [
        "B365H",
        "B365D",
        "B365A",
        "B365CH",
        "B365CD",
        "B365CA",
    ]

    missing = [column for column in odds_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required odds columns: {missing}")

    df[odds_columns] = df[odds_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = df.dropna(subset=odds_columns + ["FTHG", "FTAG", "Date"]).reset_index(
        drop=True
    )

    if df.empty:
        raise ValueError(f"{path} contains no complete matches after cleaning.")

    if (df[odds_columns] <= 0).any().any():
        raise ValueError(f"{path} contains non-positive odds.")

    df = calculate_market_probabilities(df)

    return build_team_match_dataset(df)
