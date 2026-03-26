import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Test cases
test_cases = [
    {
        "name": "Test 1: Flu symptoms",
        "symptoms": ["fever", "cough", "fatigue"],
        "model": "random_forest"
    },
    {
        "name": "Test 2: COVID-19 symptoms",
        "symptoms": ["fever", "cough", "difficulty_breathing"],
        "model": "naive_bayes"
    },
    {
        "name": "Test 3: Migraine symptoms",
        "symptoms": ["headache", "fatigue"],
        "model": "random_forest"
    },
    {
        "name": "Test 4: Common Cold symptoms",
        "symptoms": ["runny_nose", "sore_throat", "cough"],
        "model": "random_forest"
    }
]

print("=" * 50)
print("Testing Disease Prediction API")
print("=" * 50)

for test in test_cases:
    print(f"\n{test['name']}")
    print(f"Symptoms: {test['symptoms']}")
    
    data = {
        "symptoms": test["symptoms"],
        "model": test["model"]
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Predicted Disease: {result['disease']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Model: {result['model_used']}")
    else:
        print(f"✗ Error: {response.text}")

print("\n" + "=" * 50)
