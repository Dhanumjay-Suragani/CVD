import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Dynamically detect absolute path to your data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure the models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset safely
df = pd.read_csv(DATA_PATH)

# Split
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression(max_iter=200, solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)

# Save artifacts
joblib.dump(model, os.path.join(MODEL_DIR, "logistic_tuned.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "standard_scaler.joblib"))

print("✅ Model and scaler saved successfully in /models/")
