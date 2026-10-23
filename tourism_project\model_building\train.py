from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder


# Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"

TARGET_COLUMN = "ProdTaken"
RANDOM_STATE = 42


def train_model():
    # 1. Load train and test datasets
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Training/test splits not found. Run prep.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Missing target: {TARGET_COLUMN}")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Missing target: {TARGET_COLUMN}")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # Ensure the target is numeric and binary
    y_train = pd.to_numeric(y_train, errors="raise").astype(int)
    y_test = pd.to_numeric(y_test, errors="raise").astype(int)

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training target must contain both 0 and 1.")

    # 2. Identify numeric and categorical features
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # 3. Define model and hyperparameter grid
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    # 4. Configure MLflow experiment tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Tourism_Package_Prediction")

    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("random_state", RANDOM_STATE)

        # 5. Tune the model using cross-validation
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric(
            "best_cv_f1",
            float(grid_search.best_score_),
        )

        # 6. Evaluate the best model on the held-out test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "test_recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "test_f1": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "test_roc_auc": roc_auc_score(y_test, y_prob),
        }

        mlflow.log_metrics({
            name: float(value)
            for name, value in metrics.items()
        })

        print("\nBest hyperparameters:")
        print(grid_search.best_params_)

        print("\nTest metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test, y_pred, zero_division=0
        ))

        # 7. Log the trained model to MLflow
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
        )

        # 8. Save the best model for the Streamlit application
        DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        print(f"\nBest model saved to: {MODEL_PATH}")
        print("Training and MLflow tracking completed.")


if __name__ == "__main__":
    train_model()
