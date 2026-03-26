# 🏥 Disease Predictor

AI-powered disease prediction system using machine learning. Predict diseases based on symptoms with 94%+ accuracy.

## 🌟 Features

- **53 Diseases** including cancers, heart disease, diabetes, HIV/AIDS, and more
- **123 Symptoms** for accurate predictions
- **2 ML Models**: Random Forest & Naive Bayes
- **Interactive Web UI** with real-time search
- **Top-K Predictions** - see multiple possible diseases ranked by confidence
- **94%+ Accuracy** on test data

## 🚀 Live Demo

[Coming soon - Deploy on Vercel]

## 📊 Diseases Covered

Cancers, Heart Disease, Diabetes, HIV/AIDS, Respiratory diseases, Neurological conditions, Autoimmune diseases, Mental health conditions, and many more.

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **ML Models**: scikit-learn (Random Forest, Naive Bayes)
- **Frontend**: HTML, CSS, JavaScript
- **Data**: Pandas, NumPy

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/mohamedbashirnux/Disease-Predictor-.git
cd Disease-Predictor-

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python create_sample_dataset.py

# Train models
python train.py

# Run the app
python app.py
```

Visit `http://localhost:5000` in your browser.

## 🎯 Usage

1. Select symptoms from the list (search available)
2. Choose ML model (Random Forest or Naive Bayes)
3. Select how many predictions you want (Top 1, 3, or 5)
4. Click "Predict Disease"
5. View results with confidence scores

## 📁 Project Structure

```
├── app.py                      # Flask web application
├── train.py                    # Model training script
├── predict.py                  # Prediction logic
├── create_sample_dataset.py    # Dataset generation
├── models/                     # Trained ML models
├── data/                       # Dataset files
├── templates/                  # HTML templates
└── requirements.txt            # Python dependencies
```

## 🤖 Models

- **Random Forest**: 93.87% accuracy
- **Naive Bayes**: 94.46% accuracy

## ⚠️ Disclaimer

This is an educational project. Always consult healthcare professionals for medical advice. Do not use this for actual medical diagnosis.

## 📄 License

MIT License

## 👨‍💻 Author

Mohamed Bashir

---

Made with ❤️ using Flask and Machine Learning
