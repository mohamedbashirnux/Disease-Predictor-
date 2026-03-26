from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "disease_symptoms.csv"
MODELS_DIR = APP_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Prepare data
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train Naive Bayes
print("\nTraining Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_accuracy = nb_model.score(X_test, y_test)
print(f"Naive Bayes Accuracy: {nb_accuracy:.2%}")
nb_pred = nb_model.predict(X_test)
nb_report = classification_report(y_test, nb_pred, output_dict=True, zero_division=0)

# Train Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
rf_pred = rf_model.predict(X_test)
rf_report = classification_report(y_test, rf_pred, output_dict=True, zero_division=0)

# Save models
joblib.dump(nb_model, MODELS_DIR / "naive_bayes_model.pkl")
joblib.dump(rf_model, MODELS_DIR / "random_forest_model.pkl")
joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_names.pkl")

print("\nModels saved successfully!")
print(f"Best model: {'Random Forest' if rf_accuracy > nb_accuracy else 'Naive Bayes'}")

# Save a small training report for your "AI/project documentation"
report = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": {
        "path": str(DATA_PATH),
        "rows": int(df.shape[0]),
        "num_symptoms": int(X.shape[1]),
        "num_diseases": int(y.nunique()),
    },
    "split": {
        "test_size": 0.2,
        "random_state": 42,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    },
    "models": {
        "naive_bayes": {
            "accuracy": float(nb_accuracy),
            "macro_f1": float(nb_report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(nb_report.get("weighted avg", {}).get("f1-score", 0.0)),
        },
        "random_forest": {
            "accuracy": float(rf_accuracy),
            "macro_f1": float(rf_report.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(rf_report.get("weighted avg", {}).get("f1-score", 0.0)),
        },
    },
}

(MODELS_DIR / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Training report saved to: {MODELS_DIR / 'training_report.json'}")
