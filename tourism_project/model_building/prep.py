
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

TEST_SIZE = 0.20
RANDOM_STATE = 42
TARGET_COLUMN = "ProdTaken"
ID_COLUMN = "CustomerID"


# Expected numeric columns and initial validation ranges.
# These ranges flag unusual values; they do not delete rows.
EXPECTED_RANGES = {
    "Age": (18, 100),
    "CityTier": (1, 3),
    "NumberOfPersonVisiting": (1, 20),
    "PreferredPropertyStar": (1, 5),
    "NumberOfTrips": (0, 100),
    "Passport": (0, 1),
    "OwnCar": (0, 1),
    "NumberOfChildrenVisiting": (0, 20),
    "MonthlyIncome": (0, 1_000_000),
    "PitchSatisfactionScore": (1, 5),
    "NumberOfFollowups": (0, 20),
    "DurationOfPitch": (0, 240),
}

BINARY_COLUMNS = ["Passport", "OwnCar"]


# ==================================================
# 2. LOAD DATA
# ==================================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # First try reading as a CSV.
    try:
        df = pd.read_csv(DATA_PATH)

        # Detect a possible Excel file with a .csv extension.
        if len(df.columns) == 1:
            try:
                excel_df = pd.read_excel(DATA_PATH)
                if len(excel_df.columns) > 1:
                    df = excel_df
            except Exception:
                pass

    except (UnicodeDecodeError, pd.errors.ParserError):
        df = pd.read_excel(DATA_PATH)

    if df.empty:
        raise ValueError("The dataset is empty.")

    # Standardize column names.
    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        duplicates = df.columns[
            df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names found: {duplicates}"
        )

    # Remove exported index columns.
    unnamed_columns = [
        col for col in df.columns
        if col.lower().startswith("unnamed:")
    ]

    df = df.drop(
        columns=unnamed_columns,
        errors="ignore",
    )

    return df


# ==================================================
# 3. CLEAN AND VALIDATE DATA
# ==================================================

def clean_dataset(df):
    report = {
        "source_rows": int(len(df)),
        "source_columns": int(len(df.columns)),
        "removed_index_columns": [],
        "missing_target_rows_removed": 0,
        "exact_duplicate_rows_removed": 0,
        "category_changes": {},
        "invalid_numeric_values": {},
        "out_of_range_values": {},
        "missing_values_remaining": {},
    }

    report["removed_index_columns"] = [
        col for col in df.columns
        if col.lower().startswith("unnamed:")
    ]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target '{TARGET_COLUMN}' is missing."
        )

    # Remove whitespace from column names and text values.
    for col in df.select_dtypes(
        include=["object", "string"]
    ).columns:
        df[col] = df[col].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

        df[col] = df[col].replace("", pd.NA)

    # Remove records that have no target.
    missing_target_count = int(
        df[TARGET_COLUMN].isna().sum()
    )

    report["missing_target_rows_removed"] = (
        missing_target_count
    )

    df = df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    # Validate target labels.
    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target = (
        target.isna()
        | ~target.isin([0, 1])
    )

    if invalid_target.any():
        invalid_values = df.loc[
            invalid_target, TARGET_COLUMN
        ].head(10).tolist()

        raise ValueError(
            "ProdTaken must contain only 0 and 1. "
            f"Invalid examples: {invalid_values}"
        )

    df[TARGET_COLUMN] = target.astype(int)

    # Standardize known gender spelling variations.
    if "Gender" in df.columns:
        before = df["Gender"].value_counts(
            dropna=False
        ).to_dict()

        gender_mapping = {
            "male": "Male",
            "female": "Female",
            "fe male": "Female",
        }

        df["Gender"] = df["Gender"].map(
            lambda value: gender_mapping.get(
                value.casefold(), value
            ) if isinstance(value, str) else value
        )

        after = df["Gender"].value_counts(
            dropna=False
        ).to_dict()

        report["category_changes"]["Gender"] = {
            "before": {
                str(k): int(v)
                for k, v in before.items()
            },
            "after": {
                str(k): int(v)
                for k, v in after.items()
            },
        }

    # Standardize known occupation spelling variations.
    if "Occupation" in df.columns:
        occupation_mapping = {
            "free lancer": "Freelancer",
            "freelancer": "Freelancer",
        }

        df["Occupation"] = df["Occupation"].map(
            lambda value: occupation_mapping.get(
                value.casefold(), value
            ) if isinstance(value, str) else value
        )

    # Convert known numeric fields.
    for col in EXPECTED_RANGES:
        if col not in df.columns:
            continue

        original_not_null = df[col].notna()

        converted = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        newly_invalid = (
            original_not_null
            & converted.isna()
        )

        if newly_invalid.any():
            report["invalid_numeric_values"][col] = int(
                newly_invalid.sum()
            )

        df[col] = converted

    # Binary columns must contain only 0 or 1 when present.
    for col in BINARY_COLUMNS:
        if col not in df.columns:
            continue

        invalid_binary = (
            df[col].notna()
            & ~df[col].isin([0, 1])
        )

        if invalid_binary.any():
            report.setdefault(
                "invalid_binary_values", {}
            )[col] = {
                "count": int(invalid_binary.sum()),
                "examples": df.loc[
                    invalid_binary, col
                ].head(10).tolist(),
            }

    # Report suspicious ranges without deleting records.
    for col, (minimum, maximum) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue

        out_of_range = (
            df[col].notna()
            & ~df[col].between(minimum, maximum)
        )

        if out_of_range.any():
            report["out_of_range_values"][col] = {
                "count": int(out_of_range.sum()),
                "allowed_range": [
                    minimum,
                    maximum,
                ],
                "examples": df.loc[
                    out_of_range, col
                ].head(10).tolist(),
            }

    # Remove exact duplicate records.
    duplicate_count = int(df.duplicated().sum())

    report["exact_duplicate_rows_removed"] = (
        duplicate_count
    )

    df = df.drop_duplicates().copy()

    # Report duplicate identifiers; do not assume they are errors.
    if ID_COLUMN in df.columns:
        report["duplicate_customer_ids"] = int(
            df[ID_COLUMN].duplicated().sum()
        )

    # Record missing values for the training pipeline.
    report["missing_values_remaining"] = {
        col: int(count)
        for col, count in df.isna().sum().items()
        if count > 0
    }

    # Ensure both classes are available.
    if set(df[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(
            "Both target classes 0 and 1 must be present."
        )

    class_counts = df[TARGET_COLUMN].value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Not enough examples in each class "
            "for a stratified split."
        )

    report["cleaned_rows"] = int(len(df))
    report["cleaned_columns"] = int(len(df.columns))

    report["purchase_distribution"] = {
        str(k): int(v)
        for k, v in class_counts.items()
    }

    report["purchase_rate"] = float(
        df[TARGET_COLUMN].mean()
    )

    return df, report


# ==================================================
# 4. CREATE TRAINING AND TESTING DATA
# ==================================================

def prepare_data():
    df = load_dataset()

    df, report = clean_dataset(df)

    # Keep CustomerID for traceability in the split files,
    # but exclude it from the model features.
    X = df.drop(
        columns=[TARGET_COLUMN, ID_COLUMN],
        errors="ignore",
    )

    y = df[TARGET_COLUMN]

    if X.empty:
        raise ValueError(
            "No feature columns remain after cleaning."
        )

    # Stratification preserves the purchase-rate balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = X_train.copy()
    test_df = X_test.copy()


    train_df[TARGET_COLUMN] = y_train
    test_df[TARGET_COLUMN] = y_test

    # Keep a consistent column order.
    output_columns = list(X.columns) + [TARGET_COLUMN]

    train_df = train_df[output_columns]
    test_df = test_df[output_columns]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"
    report_path = OUTPUT_DIR / "cleaning_report.json"

    train_df.to_csv(
        train_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    report["training_rows"] = int(len(train_df))
    report["testing_rows"] = int(len(test_df))

    report["training_purchase_rate"] = float(
        y_train.mean()
    )

    report["testing_purchase_rate"] = float(
        y_test.mean()
    )

    report["model_features"] = list(X.columns)

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    # Display a concise summary.
    print("\nDATA PREPARATION COMPLETED")
    print("=" * 35)
    print(f"Source rows: {report['source_rows']}")
    print(f"Cleaned rows: {report['cleaned_rows']}")
    print(
        "Exact duplicates removed:",
        report["exact_duplicate_rows_removed"],
    )
    print(
        "Rows removed due to missing target:",
        report["missing_target_rows_removed"],
    )
    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")
    print(
        f"Training purchase rate: "
        f"{y_train.mean():.2%}"
    )
    print(
        f"Testing purchase rate: "
        f"{y_test.mean():.2%}"
    )
    print(f"Training file: {train_path}")
    print(f"Testing file: {test_path}")
    print(f"Cleaning report: {report_path}")

    if report["invalid_numeric_values"]:
        print(
            "\nWARNING: Invalid numeric values:",
            report["invalid_numeric_values"],
        )

    if report.get("invalid_binary_values"):
        print(
            "\nWARNING: Invalid binary values:",
            report["invalid_binary_values"],
        )

    if report["out_of_range_values"]:
        print(
            "\nWARNING: Values outside configured ranges:"
        )
        print(
            json.dumps(
                report["out_of_range_values"],
                indent=2,
            )
        )

    if report.get("duplicate_customer_ids", 0):
        print(
            "\nWARNING: Duplicate CustomerIDs:",
            report["duplicate_customer_ids"],
        )

    if report["missing_values_remaining"]:
        print(
            "\nMissing values remain for the training "
            "pipeline's imputers:"
        )
        print(report["missing_values_remaining"])


if __name__ == "__main__":
    prepare_data()
