import math
import pandas as pd


def get_points(g_for, g_against):
    if g_for > g_against:
        return 3
    elif g_for == g_against:
        return 1
    else:
        return 0


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    return df


def get_all_teams(df):
    return sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))


def expected_points(p_win, p_draw):
    return 3 * p_win + p_draw


def residual(points, expected_points):
    return points - expected_points


def get_no_vig_odds_multiway(odds: list):
    """
    :param odds: List of original odds for a multi-way market.
    :return: Tuple of no-vig (fair) odds calculated using the iterative method.
    """
    c, target_overround, accuracy, current_error = 1, 0, 3, 1000
    max_error = (10 ** (-accuracy)) / 2

    fair_odds = list()
    while current_error > max_error:
        f = -1 - target_overround
        for o in odds:
            f += (1 / o) ** c

        f_dash = 0
        for o in odds:
            f_dash += ((1 / o) ** c) * (-math.log(o))

        h = -f / f_dash
        c = c + h

        t = 0
        for o in odds:
            t += (1 / o) ** c
        current_error = abs(t - 1 - target_overround)

        fair_odds = list()
        for o in odds:
            fair_odds.append(round(o**c, 3))

    return tuple(fair_odds)


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    no_vig_odds_cols = ["NoVigH", "NoVigD", "NoVigA"]

    df[no_vig_odds_cols] = df[["B365CH", "B365CD", "B365CA"]].apply(
        lambda row: get_no_vig_odds_multiway(row.tolist()), axis=1, result_type="expand"
    )

    no_vig_prob_cols = ["NoVigPH", "NoVigPD", "NoVigPA"]

    df[no_vig_prob_cols] = 1 / df[no_vig_odds_cols]
    df["NoVigProbCheck"] = df[no_vig_prob_cols].sum(axis=1).round(2)
    df[no_vig_prob_cols] = df[no_vig_prob_cols].round(2)

    return df
