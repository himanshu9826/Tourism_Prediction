from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve paths relative to this script
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

TEST_SIZE = 0.20
RANDOM_STATE = 42
TARGET_COLUMN = "ProdTaken"


def prepare_data():
    # Load the original dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Remove unnecessary index columns, if present
    df = df.loc[
        :, ~df.columns.astype(str).str.match(r"^Unnamed")
    ]

    # Validate the target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    # Remove rows without a target value
    df = df.dropna(subset=[TARGET_COLUMN])

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Stratify to preserve the target-class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Recombine features and target for CSV output
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    # Save split files locally
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Data preparation completed successfully.")
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    print(f"Training file: {train_path}")
    print(f"Testing file: {test_path}")


if __name__ == "__main__":
    prepare_data()
