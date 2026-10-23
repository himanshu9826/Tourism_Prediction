
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from streamlit.testing.v1 import AppTest


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "tourism_project"
DATA_DIR = PROJECT / "data"
DEPLOY_DIR = PROJECT / "deployment"

MODEL_PATH = DEPLOY_DIR / "best_model.joblib"
METRICS_PATH = DEPLOY_DIR / "evaluation_results.json"
TRAIN_PATH = DATA_DIR / "processed" / "train.csv"
TEST_PATH = DATA_DIR / "processed" / "test.csv"
APP_PATH = DEPLOY_DIR / "app.py"

TARGET = "ProdTaken"


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.is_file(), f"Missing model: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def train_data():
    assert TRAIN_PATH.is_file(), f"Missing training data: {TRAIN_PATH}"
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_data():
    assert TEST_PATH.is_file(), f"Missing test data: {TEST_PATH}"
    return pd.read_csv(TEST_PATH)


# ==================================================
# STEP 1: DATA VALIDATION
# ==================================================

def test_1_data_validation(train_data, test_data):
    for name, df in [("train", train_data), ("test", test_data)]:
        assert not df.empty, f"{name} dataset is empty"
        assert TARGET in df.columns, f"{TARGET} missing from {name}"
        assert df[TARGET].notna().all(), f"Missing target in {name}"
        assert set(df[TARGET].unique()) == {0, 1}, (
            f"{name} target must contain classes 0 and 1"
        )

    assert list(train_data.drop(columns=[TARGET]).columns) == list(
        test_data.drop(columns=[TARGET]).columns
    ), "Train/test columns do not match"

    print("\nSTEP 1 PASSED: Data validation")


# ==================================================
# STEP 2: MODEL AND PREDICTION VALIDATION
# ==================================================

def test_2_model_predictions(model, test_data):
    X = test_data.drop(
        columns=[TARGET, "CustomerID"], errors="ignore"
    )

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert 1 in classes

    assert probabilities.shape == (len(X), len(classes))
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    print("\nSTEP 2 PASSED: Model and predictions")


# ==================================================
# STEP 3: MODEL PERFORMANCE VALIDATION
# ==================================================

def test_3_model_performance(model, test_data):
    X = test_data.drop(
        columns=[TARGET, "CustomerID"], errors="ignore"
    )
    y = test_data[TARGET]

    predictions = model.predict(X)
    purchase_index = list(model.classes_).index(1)
    purchase_probabilities = model.predict_proba(X)[:, purchase_index]

    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, purchase_probabilities)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    print("Predicted classes:", sorted(np.unique(predictions).tolist()))

    # Broad regression thresholds, not guarantees of model quality.
    assert accuracy >= 0.80, f"Accuracy too low: {accuracy:.4f}"
    assert auc >= 0.85, f"ROC-AUC too low: {auc:.4f}"
    assert len(np.unique(predictions)) == 2, (
        "Model predicts only one class on the test set"
    )

    print("STEP 3 PASSED: Performance validation")


# ==================================================
# STEP 4: STREAMLIT APPLICATION VALIDATION
# ==================================================

def test_4_streamlit_application():
    assert APP_PATH.is_file(), f"Missing Streamlit app: {APP_PATH}"

    app = AppTest.from_file(
        str(APP_PATH), default_timeout=30
    ).run()

    assert not app.exception, (
        f"App failed to render: {app.exception}"
    )

    assert any(
        "Tourism Package Purchase Predictor" in item.value
        for item in app.title
    ), "Expected app title was not rendered"

    # Find and submit the customer form.
    submit_buttons = [
        button for button in app.button
        if button.label == "Predict Purchase"
    ]

    assert submit_buttons, "Prediction form button not found"

    submit_buttons[0].click().run()

    assert not app.exception, (
        f"App failed after form submission: {app.exception}"
    )

    assert any(
        metric.label == "Purchase Probability"
        for metric in app.metric
    ), "Purchase probability was not displayed"

    print("\nSTEP 4 PASSED: Streamlit form and prediction")


# ==================================================
# STEP 5: MODEL ARTIFACT VALIDATION
# ==================================================


def test_5_saved_artifacts(model):
    assert MODEL_PATH.is_file(), "Saved model is missing"
    assert METRICS_PATH.is_file(), "Evaluation metrics are missing"

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    assert isinstance(results, dict), (
        "Evaluation results must be a JSON object"
    )

    # Your training script stores metrics inside "metrics".
    metrics = results.get("metrics")

    assert isinstance(metrics, dict), (
        "Missing or invalid 'metrics' section in evaluation_results.json"
    )

    for key in ["accuracy", "roc_auc"]:
        assert key in metrics, f"Missing metric: {key}"

        value = float(metrics[key])

        assert np.isfinite(value), f"Invalid metric value: {key}"
        assert 0 <= value <= 1, f"Metric out of range: {key}"

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    print("\nSTEP 5 PASSED: Saved artifacts and metrics")
