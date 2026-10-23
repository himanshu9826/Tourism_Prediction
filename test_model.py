
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "tourism_project" / "deployment" / "best_model.joblib"
TEST_PATH = ROOT / "tourism_project" / "data" / "processed" / "test.csv"

# Load the saved model and held-out test dataset
model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = "ProdTaken"
X_test = test_df.drop(columns=[TARGET, "CustomerID"], errors="ignore")
y_test = test_df[TARGET]

# Generate predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nMODEL TEST")
print("-" * 45)
print("Test rows:", len(X_test))
print("Predictions generated:", len(predictions))
print("Valid class predictions:", set(predictions).issubset({0, 1}))
print("Probabilities between 0 and 1:",
      ((probabilities >= 0) & (probabilities <= 1)).all())
print("Predicted no purchase (0):", (predictions == 0).sum())
print("Predicted purchase (1):", (predictions == 1).sum())

# Compare predictions with actual outcomes
results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
    "Purchase_Probability": probabilities
})
results["Correct"] = results["Actual"] == results["Predicted"]

print("\nPrediction accuracy on test data:",
      round(results["Correct"].mean(), 4))
print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))

# Save detailed predictions for inspection
output_path = ROOT / "tourism_project" / "deployment" / "prediction_test_results.csv"
results.to_csv(output_path, index=False)
print("\nDetailed results saved to:", output_path)
