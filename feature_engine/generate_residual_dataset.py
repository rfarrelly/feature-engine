# generate_residual_dataset.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from residuals import build_residual_dataset

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DATA_DIR = Path("/Users/ryanfarrelly/Desktop/collector/DATA/football-data")

OUTPUT_DIR = Path("residuals")

RESIDUAL_DEFINITION = "points"


# ---------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------


def find_raw_files(
    directory: Path = RAW_DATA_DIR,
) -> list[Path]:
    """Find all raw football-data CSV files."""

    files = sorted(directory.glob("*/*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    return files


# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------


def process_file(
    input_path: Path,
    raw_directory: Path,
    output_directory: Path,
) -> dict:
    """Process one raw CSV and save its residual dataset."""

    output_path = output_directory / input_path.relative_to(raw_directory)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = build_residual_dataset(
        input_path,
        residual_definition=RESIDUAL_DEFINITION,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return {
        "Input": str(input_path),
        "Output": str(output_path),
        "Rows": len(result),
        "Status": "OK",
    }


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------


def process_all_files(
    raw_directory: Path = RAW_DATA_DIR,
    output_directory: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Process every raw football-data CSV."""

    files = find_raw_files(raw_directory)

    rows = []

    print(f"Found {len(files)} raw files.\n")

    for number, input_path in enumerate(
        files,
        start=1,
    ):

        try:

            result = process_file(
                input_path=input_path,
                raw_directory=raw_directory,
                output_directory=output_directory,
            )

            rows.append(result)

            print(f"[{number}/{len(files)}] " f"OK  {input_path}")

        except Exception as exc:

            result = {
                "Input": str(input_path),
                "Output": "",
                "Rows": 0,
                "Status": f"ERROR: {exc}",
            }

            rows.append(result)

            print(f"[{number}/{len(files)}] " f"ERROR  {input_path}")

            print(f"        {exc}")

    summary = pd.DataFrame(rows)

    successful = (summary["Status"] == "OK").sum()

    failed = len(summary) - successful

    print("\nPROCESSING COMPLETE")
    print(f"Files found:     {len(files)}")
    print(f"Files processed: {successful}")
    print(f"Files failed:    {failed}")

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


if __name__ == "__main__":

    summary = process_all_files()

    summary.to_csv(
        "residual_processing_summary.csv",
        index=False,
    )

    print("\nSUMMARY")
    print(summary.to_string(index=False))
