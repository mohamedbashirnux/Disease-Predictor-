from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from predict import DiseasePredictor

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

app = Flask(__name__)

# Initialize predictor
try:
    predictor = DiseasePredictor(models_dir=MODELS_DIR)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    predictor = None


def _ensure_predictor():
    if predictor is None:
        return jsonify({"error": "Models not loaded"}), 500
    return None


def _parse_request_payload():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return None, None, None, jsonify({"error": "Please provide symptoms in JSON format"}), 400

    symptoms = data["symptoms"]
    model_type = data.get("model", "random_forest")
    top_k = data.get("top_k", 1)

    if not isinstance(symptoms, list):
        return None, None, None, jsonify({"error": "Symptoms must be a list"}), 400

    return symptoms, model_type, top_k, None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_info():
    return jsonify({
        'message': 'Disease Prediction API',
        'endpoints': {
            '/predict': 'POST - Predict disease from symptoms',
            '/predict/compare': 'POST - Compare naive_bayes vs random_forest',
            '/metadata': 'GET - Get symptoms, diseases, models',
            '/stats': 'GET - Model and dataset stats',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'models_loaded': predictor is not None})

@app.route('/metadata')
def metadata():
    err = _ensure_predictor()
    if err:
        return err

    return jsonify({
        "models": predictor.get_available_models(),
        "symptoms": predictor.get_known_symptoms(),
        "diseases": predictor.get_known_diseases(),
    })


@app.route('/stats')
def stats():
    err = _ensure_predictor()
    if err:
        return err

    report_path = MODELS_DIR / "training_report.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}

    return jsonify({
        "counts": {
            "models": len(predictor.get_available_models()),
            "symptoms": len(predictor.get_known_symptoms()),
            "diseases": len(predictor.get_known_diseases()),
        },
        "training_report": report,
    })

@app.route('/predict', methods=['POST'])
def predict():
    err = _ensure_predictor()
    if err:
        return err

    symptoms, model_type, top_k, payload_error, code = _parse_request_payload()
    if payload_error is not None:
        return payload_error, code

    try:
        result = predictor.predict(symptoms, model_type=model_type, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/compare', methods=['POST'])
def predict_compare():
    err = _ensure_predictor()
    if err:
        return err

    symptoms, _, top_k, payload_error, code = _parse_request_payload()
    if payload_error is not None:
        return payload_error, code

    try:
        rf_result = predictor.predict(symptoms, model_type="random_forest", top_k=top_k)
        nb_result = predictor.predict(symptoms, model_type="naive_bayes", top_k=top_k)
        return jsonify({
            "symptoms": symptoms,
            "top_k": int(top_k),
            "comparison": {
                "random_forest": rf_result,
                "naive_bayes": nb_result,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# For Vercel serverless
app = app
