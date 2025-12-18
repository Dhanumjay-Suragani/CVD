# app/inference.py
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "logistic_tuned.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.joblib"))

FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
THRESHOLD = 0.5

def predict(sample: dict):
    # Build DataFrame in the exact feature order used during training
    df = pd.DataFrame([sample], columns=FEATURES)
    df_scaled = scaler.transform(df)

    # model.predict_proba returns [[P(class=0), P(class=1)]]
    # In your trained model, class=1 is very likely "NO disease".
    # So we invert it to get probability of heart disease.
    proba_no_disease = model.predict_proba(df_scaled)[0][1]
    proba_disease = 1.0 - proba_no_disease

    pred = int(proba_disease >= THRESHOLD)

    return {
        "prediction": pred,                                  # 1 = heart disease, 0 = no heart disease
        "probability": round(float(proba_disease), 3),       # probability of heart disease
        "risk": "High" if pred == 1 else "Low",
        "advice": "Consult a cardiologist." if pred == 1 else "Maintain a healthy lifestyle."
    }
