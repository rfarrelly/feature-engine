import pandas as pd


def add_home_away_gp(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
        - HomeGP: total games played by home team before this match
        - AwayGP: total games played by away team before this match
    """

    df = matches.copy()

    # Ensure chronological order
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df["_match_id"] = df.index

    # Create long format (one row per team per match)
    long = pd.concat(
        [
            df[["_match_id", "Date"]].assign(Team=df["HomeTeam"], Side="HomeTeam"),
            df[["_match_id", "Date"]].assign(Team=df["AwayTeam"], Side="AwayTeam"),
        ],
        ignore_index=True,
    ).sort_values(["Team", "Date", "_match_id"])

    # Count prior games per team (no current match included)
    long["GP"] = long.groupby("Team").cumcount()

    # Split back out
    home_gp = long[long["Side"] == "HomeTeam"][["_match_id", "GP"]].rename(
        columns={"GP": "HomeGP"}
    )

    away_gp = long[long["Side"] == "AwayTeam"][["_match_id", "GP"]].rename(
        columns={"GP": "AwayGP"}
    )

    # Merge back to original matches
    df = df.merge(home_gp, on="_match_id")
    df = df.merge(away_gp, on="_match_id")

    return df.drop(columns="_match_id")
