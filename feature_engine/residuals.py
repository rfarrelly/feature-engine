# residuals.py

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def get_points(goals_for: int, goals_against: int) -> int:
    """Return league points from a match result."""

    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def expected_points(
    win_probability: pd.Series,
    draw_probability: pd.Series,
) -> pd.Series:
    """Calculate expected league points."""

    return 3 * win_probability + draw_probability


def get_no_vig_odds_multiway(
    odds: list[float],
    accuracy: int = 3,
) -> tuple[float, float, float]:
    """
    Remove bookmaker overround using the power transformation.
    """

    c = 1.0
    max_error = (10**-accuracy) / 2

    for _ in range(100):
        probabilities = [(1 / odd) ** c for odd in odds]

        total = sum(probabilities)
        error = total - 1

        if abs(error) <= max_error:
            break

        derivative = sum(
            probability * (-math.log(odd))
            for probability, odd in zip(odds, probabilities)
        )

        c -= error / derivative

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

    df[f"{prefix}NoVigH"] = fair_odds.iloc[:, 0]
    df[f"{prefix}NoVigD"] = fair_odds.iloc[:, 1]
    df[f"{prefix}NoVigA"] = fair_odds.iloc[:, 2]

    df[f"{prefix}NoVigPH"] = 1 / df[f"{prefix}NoVigH"]
    df[f"{prefix}NoVigPD"] = 1 / df[f"{prefix}NoVigD"]
    df[f"{prefix}NoVigPA"] = 1 / df[f"{prefix}NoVigA"]

    return df


def calculate_market_probabilities(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate no-vig probabilities for pre-close and close markets."""

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


def add_residual(
    df: pd.DataFrame,
    definition: str = "points",
) -> pd.DataFrame:
    """Add the selected market residual."""

    df = df.copy()

    if definition == "points":
        df["Residual"] = df["ActualPoints"] - df["ExpectedPoints"]

    elif definition == "win":
        actual_win = (df["ActualPoints"] == 3).astype(int)

        df["Residual"] = actual_win - df["PreCloseWinProb"]

    else:
        raise ValueError(f"Unknown residual definition: {definition}")

    return df


def build_team_match_dataset(
    df: pd.DataFrame,
    residual_definition: str = "points",
) -> pd.DataFrame:
    """
    Convert match-level data into one row per team per match.

    Expected points and residuals use PRE-CLOSING odds.

    Both pre-closing and closing probabilities are retained because
    the later analysis examines how the market moved after the
    pre-closing price was available.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    df["HomeEP"] = expected_points(
        df["PreCloseNoVigPH"],
        df["PreCloseNoVigPD"],
    )

    df["AwayEP"] = expected_points(
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
            "ExpectedPoints": df["HomeEP"],
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
            "ExpectedPoints": df["AwayEP"],
            "ActualPoints": df["AwayPoints"],
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

    return add_residual(
        team_df,
        definition=residual_definition,
    )


def build_residual_dataset(
    path: str | Path,
    residual_definition: str = "points",
) -> pd.DataFrame:
    """Load one raw football-data CSV and build the residual dataset."""

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

    df = df.dropna(subset=required_odds + ["FTHG", "FTAG"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{path} contains no usable matches after filtering.")

    df = calculate_market_probabilities(df)

    return build_team_match_dataset(
        df,
        residual_definition=residual_definition,
    )
