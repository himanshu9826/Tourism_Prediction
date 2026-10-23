from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "tourism_project" / "deployment" / "best_model.joblib"
TEST_PATH = ROOT / "tourism_project" / "data" / "processed" / "test.csv"
TRAIN_PATH = ROOT / "tourism_project" / "data" / "processed" / "train.csv"
METRICS_PATH = (
    ROOT / "tourism_project" / "deployment" / "evaluation_results.json"
)
OUTPUT_PATH = (
    ROOT / "tourism_project" / "deployment" / "all_test_predictions.csv"
)

TARGET = "ProdTaken"
ID_COLUMN = "CustomerID"


# --------------------------------------------------
# SHARED TEST FIXTURES
# --------------------------------------------------

@pytest.fixture(scope="session")
def model():
    """Load the existing saved model once for the test session."""
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def train_data():
    """Load the processed training dataset."""
    assert TRAIN_PATH.exists(), f"Training data not found: {TRAIN_PATH}"
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="session")
def test_data():
    """Load the held-out test dataset."""
    assert TEST_PATH.exists(), f"Test data not found: {TEST_PATH}"
    return pd.read_csv(TEST_PATH)


@pytest.fixture(scope="session")
def prediction_results(model, test_data):
    """Generate predictions for every held-out test record."""
    assert TARGET in test_data.columns, f"Missing target column: {TARGET}"

    X = test_data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    y = test_data[TARGET]

    predictions = model.predict(X)
    probabilities_all = model.predict_proba(X)

    classes = list(model.classes_)
    assert 1 in classes, "Model does not contain purchase class 1."

    purchase_index = classes.index(1)
    probabilities = probabilities_all[:, purchase_index]

    results = test_data.copy()
    results["Actual"] = y.to_numpy()
    results["Predicted"] = predictions
    results["Purchase_Probability"] = probabilities
    results["Correct"] = results["Actual"] == results["Predicted"]

    return {
        "X": X,
        "y": y,
        "predictions": predictions,
        "probabilities_all": probabilities_all,
        "probabilities": probabilities,
        "results": results,
    }


# --------------------------------------------------
# TEST 1: DATA VALIDATION
# --------------------------------------------------

def test_01_processed_datasets_are_valid(train_data, test_data):
    """Check required columns, target values and non-empty datasets."""
    for name, df in [("train", train_data), ("test", test_data)]:
        assert not df.empty, f"{name} dataset is empty."
        assert TARGET in df.columns, f"{name} dataset lacks {TARGET}."
        assert df[TARGET].notna().all(), f"{name} target contains missing values."
        assert set(df[TARGET].unique()).issubset({0, 1}), (
            f"{name} target contains values other than 0 and 1."
        )

    assert set(train_data.columns) == set(test_data.columns), (
        "Train/test columns differ."
    )

    assert train_data.shape[1] >= 2, "Too few columns in training data."


def test_02_customer_id_is_not_used_as_a_feature(test_data):
    """Verify the identifier is excluded from model input."""
    X = test_data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    assert ID_COLUMN not in X.columns


def test_03_target_distribution_is_reasonable(train_data, test_data):
    """Print class distribution and ensure both classes are represented."""
    for name, df in [("train", train_data), ("test", test_data)]:
        counts = df[TARGET].value_counts()
        assert 0 in counts.index and 1 in counts.index, (
            f"{name} dataset must contain both target classes."
        )

        rate = float(df[TARGET].mean())
        print(f"\n{name} purchase rate: {rate:.2%}")


# --------------------------------------------------
# TEST 2: SAVED MODEL VALIDATION
# --------------------------------------------------

def test_04_saved_model_loads(model):
    """Confirm that the model loads and supports prediction."""
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "classes_")
    assert 0 in model.classes_
    assert 1 in model.classes_


def test_05_model_predictions_cover_every_test_record(
    test_data, prediction_results
):
    """Ensure every held-out record receives exactly one prediction."""
    predictions = prediction_results["predictions"]

    assert len(predictions) == len(test_data)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert len(predictions) > 0


def test_06_prediction_probabilities_are_valid(prediction_results):
    """Check probability ranges and probability row sums."""
    probabilities_all = prediction_results["probabilities_all"]
    purchase_probabilities = prediction_results["probabilities"]

    assert probabilities_all.shape[0] == len(purchase_probabilities)
    assert np.isfinite(probabilities_all).all()
    assert np.all((probabilities_all >= 0) & (probabilities_all <= 1))
    np.testing.assert_allclose(
        probabilities_all.sum(axis=1),
        np.ones(probabilities_all.shape[0]),
        atol=1e-6,
    )

    assert np.all(
        (purchase_probabilities >= 0) & (purchase_probabilities <= 1)
    )


def test_07_model_predictions_are_deterministic(model, prediction_results):
    """Check repeat predictions on the same rows."""
    X = prediction_results["X"]
    first = model.predict(X)
    second = model.predict(X)

    np.testing.assert_array_equal(first, second)


# --------------------------------------------------
# TEST 3: MODEL PERFORMANCE
# --------------------------------------------------

def test_08_model_performance_meets_minimum_requirements(
    prediction_results,
):
    """
    Basic regression gates based on the project's previous results.
    Adjust these minimums only when you intentionally revise acceptance criteria.
    """
    y = prediction_results["y"]
    predictions = prediction_results["predictions"]
    probabilities = prediction_results["probabilities"]

    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, probabilities)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    print("\n========== MODEL PERFORMANCE ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    # Baselines consistent with the project's existing evaluation.
    assert accuracy >= 0.80, "Accuracy is below the minimum of 0.80."
    assert auc >= 0.85, "ROC-AUC is below the minimum of 0.85."
    assert len(np.unique(predictions)) == 2, (
        "The model predicts only one class."
    )


def test_09_classification_report_and_confusion_matrix(
    prediction_results,
):
    """Verify evaluation output can be computed for all test rows."""
    y = prediction_results["y"]
    predictions = prediction_results["predictions"]

    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    report = classification_report(
        y, predictions, labels=[0, 1], zero_division=0
    )

    assert matrix.shape == (2, 2)
    assert matrix.sum() == len(y)
    assert isinstance(report, str) and len(report) > 0

    print("\nConfusion matrix:\n", matrix)
    print("\nClassification report:\n", report)


# --------------------------------------------------
# TEST 4: PREDICTION REPORT FOR ALL CUSTOMERS
# --------------------------------------------------

def test_10_save_predictions_for_every_test_record(
    prediction_results,
):
    """Save actual values, predicted values and probabilities for review."""
    results = prediction_results["results"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    assert OUTPUT_PATH.exists()
    saved = pd.read_csv(OUTPUT_PATH)
    assert len(saved) == len(results)
    assert {
        "Actual",
        "Predicted",
        "Purchase_Probability",
        "Correct",
    }.issubset(saved.columns)


def test_11_prediction_error_summary(prediction_results):
    """Report false positives and false negatives without requiring zero errors."""
    results = prediction_results["results"]
    y = prediction_results["y"]
    predictions = prediction_results["predictions"]
    probabilities = prediction_results["probabilities"]

    false_positives = results[
        (results["Actual"] == 0) & (results["Predicted"] == 1)
    ]
    false_negatives = results[
        (results["Actual"] == 1) & (results["Predicted"] == 0)
    ]
    errors = results[~results["Correct"]]

    print("\n========== ALL-RECORD PREDICTION SUMMARY ==========")
    print("Total test records:", len(results))
    print("Correct predictions:", int(results["Correct"].sum()))
    print("Incorrect predictions:", len(errors))
    print("False positives:", len(false_positives))
    print("False negatives:", len(false_negatives))
    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(y, probabilities), 4))

    print("\nFirst 20 incorrect predictions:")
    if errors.empty:
        print("No incorrect predictions.")
    else:
        print(errors.head(20).to_string(index=False))

    assert len(errors) + int(results["Correct"].sum()) == len(results)


# --------------------------------------------------
# TEST 5: SAVED EVALUATION ARTIFACT
# --------------------------------------------------

def test_12_evaluation_metrics_file_is_valid():
    """Check that the saved evaluation JSON exists and contains metrics."""
    assert METRICS_PATH.exists(), (
        f"Evaluation metrics file not found: {METRICS_PATH}"
    )

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        evaluation = json.load(file)

    assert isinstance(evaluation, dict)
    assert isinstance(evaluation.get("metrics"), dict), (
        "Expected a 'metrics' dictionary in evaluation_results.json."
    )


# --------------------------------------------------
# TEST 6: INPUT FEATURE CONSISTENCY
# --------------------------------------------------

def test_13_model_accepts_test_feature_schema(model, test_data):
    """Ensure the model accepts the expected held-out feature columns."""
    X = test_data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")

    # This exercises the saved model's complete preprocessing + prediction pipeline.
    predictions = model.predict(X)

    assert len(predictions) == len(test_data)


def test_14_no_duplicate_test_columns(test_data):
    """Catch accidental duplicate columns in the processed test data."""
    assert not test_data.columns.duplicated().any(), (
        "Processed test data contains duplicate column names."
    )