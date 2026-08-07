from typing import Tuple

import pandas as pd
from add_games_played import add_home_away_gp
from residuals import get_points, get_all_teams, load_and_prepare_data


# -------------------------
# Core Calculations
# -------------------------
def calculate_team_stats(games_so_far, all_teams):
    stats = {
        team: {
            "GP": 0,
            "Pts": 0,
            "Home_GP": 0,
            "Home_Pts": 0,
            "Away_GP": 0,
            "Away_Pts": 0,
        }
        for team in all_teams
    }

    for _, row in games_so_far.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["FTHG"], row["FTAG"]

        hp = get_points(hg, ag)
        ap = get_points(ag, hg)

        stats[h]["GP"] += 1
        stats[h]["Pts"] += hp
        stats[h]["Home_GP"] += 1
        stats[h]["Home_Pts"] += hp

        stats[a]["GP"] += 1
        stats[a]["Pts"] += ap
        stats[a]["Away_GP"] += 1
        stats[a]["Away_Pts"] += ap

    return stats


def calculate_team_ppgs(stats):
    team_ppgs = {}

    for team, s in stats.items():
        ppg = s["Pts"] / s["GP"] if s["GP"] > 0 else 0
        home_ppg = s["Home_Pts"] / s["Home_GP"] if s["Home_GP"] > 0 else 0
        away_ppg = s["Away_Pts"] / s["Away_GP"] if s["Away_GP"] > 0 else 0

        team_ppgs[team] = {
            "PPG": ppg,
            "Home_PPG": home_ppg,
            "Away_PPG": away_ppg,
        }

    return team_ppgs


def calculate_ppi_snapshot(games_so_far, team_ppgs, all_teams, current_date):
    records = []

    for team in all_teams:
        team_games = games_so_far[
            (games_so_far["HomeTeam"] == team) | (games_so_far["AwayTeam"] == team)
        ]

        opponent_strength_sum = 0
        opponent_count = 0

        if not team_games.empty:
            for _, row in team_games.iterrows():
                if row["HomeTeam"] == team:
                    opponent = row["AwayTeam"]
                    opp_strength = team_ppgs[opponent]["Away_PPG"]
                else:
                    opponent = row["HomeTeam"]
                    opp_strength = team_ppgs[opponent]["Home_PPG"]

                opponent_strength_sum += opp_strength
                opponent_count += 1

            opp_ppg_avg = (
                opponent_strength_sum / opponent_count if opponent_count > 0 else 0
            )

            current_ppg = team_ppgs[team]["PPG"]
            ppi = current_ppg * opp_ppg_avg

            records.append(
                {
                    "Date": current_date,
                    "Team": team,
                    "PPG": current_ppg,
                    "Opponent_PPG": opp_ppg_avg,
                    "PPI": ppi,
                }
            )

    return records


def build_ppi_dataframe(df, all_teams, apply_shift=True):
    ppi_records = []

    for current_date in df["Date"].unique():
        games_so_far = df[df["Date"] <= current_date]

        stats = calculate_team_stats(games_so_far, all_teams)
        team_ppgs = calculate_team_ppgs(stats)

        ppi_records.extend(
            calculate_ppi_snapshot(games_so_far, team_ppgs, all_teams, current_date)
        )

    ppi_df = pd.DataFrame(ppi_records)
    ppi_df.sort_values(["Date", "PPI"], ascending=[True, False], inplace=True)

    if apply_shift:
        cols_to_shift = ["PPG", "Opponent_PPG", "PPI"]
        ppi_df[cols_to_shift] = ppi_df.groupby("Team")[cols_to_shift].shift(1)

    return ppi_df.sort_values(["Team", "Date"])


# -------------------------
# Merge Back to Match Data
# -------------------------
def merge_ppi_into_matches(df: pd.DataFrame, ppi_df: pd.DataFrame) -> pd.DataFrame:
    historical_columns = ["FTHG", "FTAG"]

    if all(col in df.columns for col in historical_columns):
        merged_df = df.merge(
            ppi_df, left_on=["Date", "HomeTeam"], right_on=["Date", "Team"]
        )
        merged_df = merged_df.merge(
            ppi_df, left_on=["Date", "AwayTeam"], right_on=["Date", "Team"]
        )
    else:
        ppi_df.drop("Date", inplace=True, axis=1)
        merged_df = df.merge(ppi_df, left_on=["HomeTeam"], right_on=["Team"])
        merged_df = merged_df.merge(ppi_df, left_on=["AwayTeam"], right_on=["Team"])

    merged_df = merged_df.rename(
        columns={
            "PPG_x": "HomeTeamTotalPPG",
            "PPG_y": "AwayTeamTotalPPG",
            "Opponent_PPG_x": "HomeTeamOpponentPPG",
            "Opponent_PPG_y": "AwayTeamOpponentPPG",
            "PPI_x": "HomeTeamPPI",
            "PPI_y": "AwayTeamPPI",
        }
    )

    base_columns = [
        "League",
        "Season",
        "Date",
        "Time",
        "Day",
        "HomeGP",
        "AwayGP",
        "HomeTeam",
        "AwayTeam",
        "FTR",
        "B365CH",
        "B365CD",
        "B365CA",
    ]

    metrics_columns = [
        "HomeTeamTotalPPG",
        "AwayTeamTotalPPG",
        "HomeTeamOpponentPPG",
        "AwayTeamOpponentPPG",
        "HomeTeamPPI",
        "AwayTeamPPI",
    ]

    existing_historical = [c for c in historical_columns if c in df.columns]

    final_columns = (
        base_columns  # up to "AwayTeam"
        + existing_historical  # insert right after Home/Away
        + metrics_columns
    )

    merged_df = merged_df[final_columns]
    merged_df["PPIDiff"] = abs(merged_df["HomeTeamPPI"] - merged_df["AwayTeamPPI"])
    merged_df[merged_df.columns[17:]] = merged_df[merged_df.columns[17:]].round(2)

    return merged_df


# -------------------------
# Execution
# -------------------------


def compute_ppi(
    file_path: str, shift: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_and_prepare_data(file_path)
    df = add_home_away_gp(df)
    all_teams = get_all_teams(df)
    return df, build_ppi_dataframe(df, all_teams, apply_shift=shift)


# def compute_ppi_for_fixtures(fixtures_path: str, historical_path: str):
#     fixtures_df = pd.read_csv(fixtures_path)
#     fixtures = filter_date_range(fixtures_df, TODAY, END_DATE)

#     if fixtures.empty:
#         print("No Fixtures for this date range")
#         return None

#     _, ppi = compute_ppi(historical_path, shift=False)
#     latest_ppi = ppi.loc[ppi.groupby("Team")["Date"].idxmax()]

#     return merge_ppi_into_matches(fixtures, latest_ppi).dropna(how="any", axis="index")


def compute_historical_ppi(file_path: str) -> pd.DataFrame:
    matches, ppi_shifted = compute_ppi(file_path, shift=True)
    return merge_ppi_into_matches(matches, ppi_shifted).dropna(how="any", axis="index")
