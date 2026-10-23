
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
model = joblib.load(
    ROOT / "tourism_project" / "deployment" / "best_model.joblib"
)
df = pd.read_csv(
    ROOT / "tourism_project" / "data" / "processed" / "test.csv"
)

X = df.drop(columns=["ProdTaken", "CustomerID"], errors="ignore")
y = df["ProdTaken"]

pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

print("Classification report:")
print(classification_report(y, pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y, pred))

print("ROC-AUC:", round(roc_auc_score(y, prob), 4))
