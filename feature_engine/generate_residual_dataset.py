# generate_residual_dataset.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from residuals import build_residual_dataset

RAW_DATA_DIR = Path("/Users/ryanfarrelly/Desktop/collector/DATA/football-data")

OUTPUT_DIR = Path("residuals")


def find_raw_files(directory: Path = RAW_DATA_DIR) -> list[Path]:
    """Find raw football-data CSV files one directory below the root."""

    files = sorted(directory.glob("*/*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    return files


def validate_residual_dataset(
    df: pd.DataFrame,
    source: Path,
) -> None:
    """Run structural checks on one generated residual dataset."""

    required = [
        "Date",
        "League",
        "Season",
        "Team",
        "Opponent",
        "Venue",
        "Match",
        "GoalsFor",
        "GoalsAgainst",
        "GoalDifference",
        "PreCloseWinProb",
        "PreCloseDrawProb",
        "PreCloseLossProb",
        "CloseWinProb",
        "CloseDrawProb",
        "CloseLossProb",
        "ExpectedPoints",
        "ActualPoints",
        "Residual",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{source}: generated dataset is missing columns: {missing}")

    if df.empty:
        raise ValueError(f"{source}: generated dataset is empty.")

    probability_columns = [
        "PreCloseWinProb",
        "PreCloseDrawProb",
        "PreCloseLossProb",
        "CloseWinProb",
        "CloseDrawProb",
        "CloseLossProb",
    ]

    probabilities = df[probability_columns]

    if probabilities.isna().any().any():
        raise ValueError(f"{source}: generated probabilities contain NaN values.")

    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError(f"{source}: generated probabilities are outside [0, 1].")

    pre_close_sum = (
        df["PreCloseWinProb"] + df["PreCloseDrawProb"] + df["PreCloseLossProb"]
    )

    close_sum = df["CloseWinProb"] + df["CloseDrawProb"] + df["CloseLossProb"]

    if not (pre_close_sum.sub(1).abs() < 0.002).all():
        raise ValueError(f"{source}: pre-closing probabilities do not sum to 1.")

    if not (close_sum.sub(1).abs() < 0.002).all():
        raise ValueError(f"{source}: closing probabilities do not sum to 1.")

    expected_goal_difference = df["GoalsFor"] - df["GoalsAgainst"]

    if not df["GoalDifference"].equals(expected_goal_difference):
        raise ValueError(f"{source}: GoalDifference does not match the score.")

    expected_residual = df["ActualPoints"] - df["ExpectedPoints"]

    if not (df["Residual"].sub(expected_residual).abs().lt(1e-10).all()):
        raise ValueError(
            f"{source}: Residual does not equal ActualPoints - ExpectedPoints."
        )

    duplicate_matches = df.duplicated(["League", "Season", "Team", "Match"])

    if duplicate_matches.any():
        raise ValueError(f"{source}: duplicate team-match observations found.")


def process_file(
    input_path: Path,
    raw_directory: Path = RAW_DATA_DIR,
    output_directory: Path = OUTPUT_DIR,
) -> dict:
    """Generate and validate one residual CSV."""

    residuals = build_residual_dataset(input_path)

    validate_residual_dataset(
        residuals,
        input_path,
    )

    relative_path = input_path.relative_to(raw_directory)
    output_path = output_directory / relative_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals.to_csv(
        output_path,
        index=False,
    )

    return {
        "Input": str(input_path),
        "Output": str(output_path),
        "Rows": len(residuals),
        "Matches": len(residuals) // 2,
        "Status": "OK",
    }


def process_all_files(
    raw_directory: Path = RAW_DATA_DIR,
    output_directory: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Generate validated residual datasets for every raw CSV."""

    files = find_raw_files(raw_directory)

    print(f"Found {len(files)} raw files.")

    results = []

    for number, input_path in enumerate(files, start=1):
        try:
            result = process_file(
                input_path,
                raw_directory=raw_directory,
                output_directory=output_directory,
            )

            print(f"[{number}/{len(files)}] OK    {input_path}")

        except Exception as exc:
            result = {
                "Input": str(input_path),
                "Output": "",
                "Rows": 0,
                "Matches": 0,
                "Status": f"ERROR: {exc}",
            }

            print(f"[{number}/{len(files)}] ERROR {input_path}")
            print(f"    {exc}")

        results.append(result)

    summary = pd.DataFrame(results)

    ok = summary["Status"].eq("OK")

    print("\nPROCESSING COMPLETE")
    print(f"Files found:     {len(files)}")
    print(f"Files processed: {ok.sum()}")
    print(f"Files failed:    {(~ok).sum()}")

    return summary


if __name__ == "__main__":
    summary = process_all_files()

    summary_path = Path("residual_processing_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nSummary written to {summary_path}")
