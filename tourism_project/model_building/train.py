
from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, train_test_split


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = DEPLOYMENT_DIR / "best_model.joblib"
RESULTS_PATH = DEPLOYMENT_DIR / "evaluation_results.json"

TARGET_COLUMN = "ProdTaken"
RANDOM_STATE = 42


def train_model():
    # 1. Load training and testing datasets
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Training/test splits not found. Run prep.py first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df_name, df in [
        ("training", train_df),
        ("testing", test_df),
    ]:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' is missing "
                f"from the {df_name} dataset."
            )

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = pd.to_numeric(
        train_df[TARGET_COLUMN], errors="raise"
    ).astype(int)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = pd.to_numeric(
        test_df[TARGET_COLUMN], errors="raise"
    ).astype(int)

    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            "Training target must contain both 0 and 1."
        )

    if set(y_test.unique()) - {0, 1}:
        raise ValueError(
            "Testing target must contain only 0 and 1."
        )

    # Remove identifier columns from features if still present.
    # Keep this consistent with the features collected by app.py.
    identifier_columns = ["CustomerID"]
    X_train = X_train.drop(
        columns=identifier_columns, errors="ignore"
    )
    X_test = X_test.drop(
        columns=identifier_columns, errors="ignore"
    )

    # Ensure training and testing columns match
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

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

    # 3. Define the Random Forest pipeline
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

    # 4. Configure MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Tourism_Package_Prediction")

    with mlflow.start_run(
        run_name="RandomForest_GridSearch_Calibration"
    ):
        mlflow.log_param(
            "model_type", "RandomForestClassifier"
        )
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("calibration", "sigmoid")


        # 5. Split training data into fit and validation sets.
        # The existing test set is kept untouched for final evaluation.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        # 6. Hyperparameter tuning only on the fit subset
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        grid_search.fit(X_fit, y_fit)
        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric(
            "best_cv_f1",
            float(grid_search.best_score_),
        )

        def positive_probabilities(model, X):
            classes = list(model.classes_)
            if 1 not in classes:
                raise ValueError("Purchase class 1 not found.")
            return model.predict_proba(X)[:, classes.index(1)]

        def evaluate_model(model, X, y):
            predictions = model.predict(X)
            probabilities = positive_probabilities(model, X)

            return {
                "accuracy": float(accuracy_score(y, predictions)),
                "precision": float(
                    precision_score(y, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(y, predictions, zero_division=0)
                ),
                "f1_score": float(
                    f1_score(y, predictions, zero_division=0)
                ),
                "roc_auc": float(roc_auc_score(y, probabilities)),
                "brier_score": float(
                    brier_score_loss(y, probabilities)
                ),
                "log_loss": float(
                    log_loss(y, probabilities, labels=[0, 1])
                ),
            }


        # 6. Use the original, uncalibrated Random Forest model
        selected_method = "uncalibrated"

        mlflow.log_param("calibration_method", "none")
        mlflow.log_param("selected_model_method", selected_method)

        # 7. Use the best Random Forest found during grid search.
        # No sigmoid or isotonic calibration is applied.
        final_model = grid_search.best_estimator_

        # 10. Save the exact model used by the app
        DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, MODEL_PATH)



        # Evaluate the uncalibrated model on the test set
        test_metrics = evaluate_model(final_model, X_test, y_test)

        print("\nFINAL TEST METRICS")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        test_predictions = final_model.predict(X_test)

        report = classification_report(
            y_test,
            test_predictions,
            output_dict=True,
            zero_division=0,
        )




        evaluation_results = {
            "best_parameters": grid_search.best_params_,
            "selected_model": selected_method,
            "validation_results": {},
            "metrics": test_metrics,
            "classification_report": report,
        }

        with open(RESULTS_PATH, "w", encoding="utf-8") as file:
            json.dump(
                evaluation_results,
                file,
                indent=4,
                allow_nan=False,
            )

        # Log final test metrics and deployed model to MLflow
        mlflow.log_metrics({
            f"test_{name}": value
            for name, value in test_metrics.items()
        })

        mlflow.sklearn.log_model(
            final_model,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration._SigmoidCalibration",
                "sklearn.tree._tree.Tree",
            ],
        )



        print(f"\nBest model saved to: {MODEL_PATH}")
        print(f"Evaluation results saved to: {RESULTS_PATH}")
        print("Training and MLflow tracking completed.")


if __name__ == "__main__":
    train_model()
