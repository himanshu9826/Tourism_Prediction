"""
Automated tests for Tourism Package Purchase Predictor.

Run from the project root:
    python -m pytest test_tourism_project.py -v

These tests check the saved model, processed data, evaluation JSON,
prediction shape/classes/probabilities, and the Streamlit app source.
They do not launch a browser or require GitHub.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "tourism_project"
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
DATA_PATH = DATA_DIR / "tourism.csv"
MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"
RESULTS_PATH = DEPLOYMENT_DIR / "evaluation_results.json"
APP_PATH = DEPLOYMENT_DIR / "app.py"
TARGET = "ProdTaken"


def _load_model():
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def _sample_input(model):
    """Build one valid input row using the model's trained feature names."""
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        # CalibratedClassifierCV may not expose feature_names_in_ consistently.
        feature_names = None

    if feature_names is None:
        source = pd.read_csv(TRAIN_PATH)
        feature_names = [
            c for c in source.columns if c != TARGET and c != "CustomerID"
        ]

    train = pd.read_csv(TRAIN_PATH)
    row = {}
    for col in feature_names:
        if col not in train.columns:
            raise AssertionError(
                f"Model expects feature '{col}', but it is missing from train.csv"
            )
        series = train[col]
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            row[col] = float(non_null.median()) if not non_null.empty else 0
            # Preserve integer-like numeric values where possible.
            if pd.api.types.is_integer_dtype(series):
                row[col] = int(row[col])
        else:
            non_null = series.dropna()
            row[col] = str(non_null.mode().iloc[0]) if not non_null.empty else "Unknown"
    return pd.DataFrame([row], columns=feature_names)


def test_required_project_files_exist():
    required = [
        DATA_PATH, TRAIN_PATH, TEST_PATH, MODEL_PATH, RESULTS_PATH, APP_PATH
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, "Missing required project files:\n" + "\n".join(missing)


def test_source_and_processed_data_have_target():
    for path in [DATA_PATH, TRAIN_PATH, TEST_PATH]:
        df = pd.read_csv(path)
        assert not df.empty, f"{path.name} is empty"
        assert TARGET in df.columns, f"{TARGET} missing from {path.name}"
        assert set(pd.to_numeric(df[TARGET], errors="raise").dropna().unique()).issubset({0, 1})


def test_customer_id_is_not_in_processed_model_features():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert TARGET in train.columns and TARGET in test.columns
    assert "CustomerID" not in train.drop(columns=[TARGET]).columns, (
        "CustomerID is still in training features. Remove it in prep.py and rerun prep.py/train.py."
    )
    assert "CustomerID" not in test.drop(columns=[TARGET]).columns, (
        "CustomerID is still in testing features. Remove it in prep.py and rerun prep.py/train.py."
    )


def test_train_test_feature_columns_match():
    train = pd.read_csv(TRAIN_PATH).drop(columns=[TARGET])
    test = pd.read_csv(TEST_PATH).drop(columns=[TARGET])
    assert list(train.columns) == list(test.columns), (
        "Train/test feature columns or their order do not match."
    )


def test_saved_model_loads():
    model = _load_model()
    assert hasattr(model, "predict"), "Saved model has no predict method"
    assert hasattr(model, "predict_proba"), "Saved model has no predict_proba method"


def test_model_features_match_processed_data():
    model = _load_model()
    train_features = list(pd.read_csv(TRAIN_PATH).drop(columns=[TARGET]).columns)
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        assert model_features == train_features, (
            f"Model feature mismatch. Model={model_features}; data={train_features}"
        )
    else:
        # Validate using a real row generated from the model's training schema.
        sample = _sample_input(model)
        prediction = model.predict(sample)
        assert len(prediction) == 1


def test_prediction_and_probability_are_valid():
    model = _load_model()
    sample = _sample_input(model)

    prediction = model.predict(sample)
    probabilities = model.predict_proba(sample)

    assert len(prediction) == 1
    assert int(prediction[0]) in (0, 1)
    assert probabilities.shape == (1, 2), (
        f"Expected probability shape (1, 2), got {probabilities.shape}"
    )
    assert np.isfinite(probabilities).all(), "Probabilities contain NaN/inf"
    assert ((probabilities >= 0) & (probabilities <= 1)).all(), (
        "Probabilities must be between 0 and 1"
    )
    assert np.isclose(probabilities.sum(axis=1)[0], 1.0, atol=1e-6), (
        "Class probabilities should sum to 1"
    )
    classes = list(model.classes_)
    assert 0 in classes and 1 in classes
    purchase_probability = probabilities[0, classes.index(1)]
    assert 0 <= purchase_probability <= 1


def test_evaluation_results_json_is_valid():
    assert RESULTS_PATH.exists(), f"Missing file: {RESULTS_PATH}"
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    assert isinstance(results, dict)
    assert "metrics" in results, "Evaluation JSON missing 'metrics'"
    metrics = results["metrics"]
    for name in ["accuracy", "precision", "recall", "f1_score"]:
        assert name in metrics, f"Missing metric: {name}"
        value = float(metrics[name])
        assert 0 <= value <= 1, f"{name} must be between 0 and 1"

    if "roc_auc" in metrics:
        assert 0 <= float(metrics["roc_auc"]) <= 1
    if "brier_score" in metrics:
        assert 0 <= float(metrics["brier_score"]) <= 1


def test_streamlit_app_has_model_prediction_logic():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "joblib.load" in source, "Streamlit app does not load a saved model"
    assert "predict" in source, "Streamlit app has no prediction call"
    assert "predict_proba" in source, "Streamlit app has no probability prediction"


def test_dataset_split_sizes_are_nonzero():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) <= len(pd.read_csv(DATA_PATH)), (
        "Split rows exceed original dataset size"
    )
