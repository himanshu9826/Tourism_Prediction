
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "tourism_project" / "deployment" / "best_model.joblib"
TEST_PATH = ROOT / "tourism_project" / "data" / "processed" / "test.csv"
OUTPUT_PATH = ROOT / "tourism_project" / "deployment" / "all_test_predictions.csv"

TARGET = "ProdTaken"

# Load the existing saved model and untouched test dataset.
model = joblib.load(MODEL_PATH)
df = pd.read_csv(TEST_PATH)

X = df.drop(columns=[TARGET, "CustomerID"], errors="ignore")
y = df[TARGET]

# Predict every held-out customer.
predictions = model.predict(X)
classes = list(model.classes_)
purchase_index = classes.index(1)
probabilities = model.predict_proba(X)[:, purchase_index]

# Create a record-by-record comparison.
results = df.copy()
results["Actual"] = y.to_numpy()
results["Predicted"] = predictions
results["Purchase_Probability"] = probabilities
results["Correct"] = results["Actual"] == results["Predicted"]

# Summary
print("\n========== MODEL VALIDATION ==========")
print("Total test records:", len(results))
print("Correct predictions:", int(results["Correct"].sum()))
print("Incorrect predictions:", int((~results["Correct"]).sum()))
print("Accuracy:", round(accuracy_score(y, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y, probabilities), 4))

print("\nClassification report:")
print(classification_report(y, predictions, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y, predictions))

# Inspect the incorrect predictions.
errors = results.loc[~results["Correct"]]
print("\nFirst 20 incorrect predictions:")
print(errors.head(20).to_string(index=False))

# Save all predictions for detailed review.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print("\nAll predictions saved to:", OUTPUT_PATH)
print("Incorrect predictions saved in the same file for review.")
