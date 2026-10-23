
from pathlib import Path
import json
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
REGISTRATION_PATH = PROJECT_DIR / "data" / "data_registration.json"

# Expected dataset columns
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def register_and_validate_data():
    print("Starting tourism dataset registration...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Remove unnecessary spreadsheet index columns if present
    df = df.loc[
        :, ~df.columns.astype(str).str.match(r"^Unnamed")
    ]

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset contains no rows.")

    if df["ProdTaken"].isna().any():
        raise ValueError("Target column 'ProdTaken' contains missing values.")

    if not set(df["ProdTaken"].unique()).issubset({0, 1}):
        raise ValueError("'ProdTaken' must contain only 0 and 1.")

    # Save the validated dataset and its registration metadata
    df.to_csv(DATA_PATH, index=False)

    metadata = {
        "dataset": DATA_PATH.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "target_column": "ProdTaken",
        "target_distribution": {
            str(key): int(value)
            for key, value in df["ProdTaken"].value_counts().items()
        },
        "unexpected_columns": unexpected_columns,
        "validation_status": "passed",
    }

    REGISTRATION_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nDataset registration successful!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Target distribution:\n{df['ProdTaken'].value_counts()}")
    print(f"Registration metadata saved to: {REGISTRATION_PATH}")


if __name__ == "__main__":
    register_and_validate_data()
