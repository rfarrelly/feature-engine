from datetime import datetime

import pandas as pd


def filter_date_range(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    filtered_df = df[
        (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    ]
    return filtered_df


def format_date(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    return df
