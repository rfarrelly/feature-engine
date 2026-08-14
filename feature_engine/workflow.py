from residuals import build_residual_dataset
import pandas as pd
from analysis import (
    add_rolling_features,
    evaluate_thresholds,
    add_confidence_intervals,
)

from pathlib import Path

DATA_DIR = Path("/Users/ryanfarrelly/Desktop/collector/DATA/football-data")

OUTPUT_DIR = Path("residuals")


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


df = load_all_residuals(directory="residuals")

df = add_rolling_features(
    df,
    windows=(3, 5),
)

results = evaluate_thresholds(
    df,
    z_column="ResidualZ_3",
)

results = add_confidence_intervals(
    results,
    df,
)


home = df[df["Venue"] == "home"]
away = df[df["Venue"] == "away"]

home_results = evaluate_thresholds(
    home,
    z_column="ResidualZ_3",
)

away_results = evaluate_thresholds(
    away,
    z_column="ResidualZ_3",
)

print("Overall\r")
print(results)
print("Home\r")
print(home_results)
print("Away\r")
print(away_results)
