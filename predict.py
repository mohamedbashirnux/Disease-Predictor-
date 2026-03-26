from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np

class DiseasePredictor:
    def __init__(self, models_dir: str | os.PathLike = "models"):
        self.models_dir = Path(models_dir).resolve()
        self.nb_model = joblib.load(self.models_dir / "naive_bayes_model.pkl")
        self.rf_model = joblib.load(self.models_dir / "random_forest_model.pkl")
        self.feature_names = joblib.load(self.models_dir / "feature_names.pkl")

        if not isinstance(self.feature_names, list) or not all(isinstance(x, str) for x in self.feature_names):
            raise ValueError("feature_names.pkl must be a list[str]")
    
    def get_available_models(self) -> list[str]:
        return ["random_forest", "naive_bayes"]

    def get_known_symptoms(self) -> list[str]:
        return list(self.feature_names)

    def get_known_diseases(self) -> list[str]:
        # Both sklearn classifiers expose classes_
        diseases = set()
        for m in (self.nb_model, self.rf_model):
            classes = getattr(m, "classes_", None)
            if classes is not None:
                diseases.update([str(c) for c in classes.tolist()])
        return sorted(diseases)

    def _vectorize_symptoms(self, symptoms: list[str]) -> np.ndarray:
        features = np.zeros(len(self.feature_names), dtype=np.float32)
        unknown = []
        for symptom in symptoms:
            if symptom in self.feature_names:
                idx = self.feature_names.index(symptom)
                features[idx] = 1.0
            else:
                unknown.append(symptom)
        return features.reshape(1, -1), unknown

    def predict(self, symptoms: list[str], model_type: str = "random_forest", top_k: int = 1):
        # Create feature vector
        if not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms):
            raise ValueError("symptoms must be a list of strings")

        symptoms = [s.strip() for s in symptoms if s and s.strip()]
        if len(symptoms) == 0:
            raise ValueError("symptoms list is empty")

        try:
            top_k = int(top_k)
        except Exception:
            raise ValueError("top_k must be an integer")
        if top_k < 1 or top_k > 10:
            raise ValueError("top_k must be between 1 and 10")

        features, unknown_symptoms = self._vectorize_symptoms(symptoms)
        
        # Predict
        if model_type == "naive_bayes":
            model = self.nb_model
        elif model_type == "random_forest":
            model = self.rf_model
        else:
            raise ValueError(f"Unknown model '{model_type}'. Use one of: {', '.join(self.get_available_models())}")

        proba = model.predict_proba(features)[0]
        classes = [str(c) for c in model.classes_.tolist()]

        # Top-K
        order = np.argsort(proba)[::-1]
        top_idx = order[:top_k]
        predictions = [
            {"disease": classes[i], "probability": float(proba[i]), "confidence": f"{proba[i] * 100:.2f}%"}
            for i in top_idx
        ]

        best = predictions[0]
        return {
            "disease": best["disease"],
            "confidence": best["confidence"],
            "probability": best["probability"],
            "model_used": model_type,
            "top_k": predictions,
            "unknown_symptoms": unknown_symptoms,
            "recognized_symptoms": [s for s in symptoms if s not in unknown_symptoms],
        }

if __name__ == "__main__":
    predictor = DiseasePredictor()
    
    # Example prediction
    test_symptoms = ['fever', 'cough', 'fatigue']
    result = predictor.predict(test_symptoms)
    print(f"Symptoms: {test_symptoms}")
    print(f"Predicted Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']}")
