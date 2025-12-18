"""
evaluate_model.py
---------------------------------
Purpose:
    Evaluate the trained Heart Disease Prediction model
    with full metrics, visuals, and saved reports.

Author:
    Commander Jay
Version:
    2.0 - Future-Ready & Robust Evaluation Script
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

# -----------------------------
# 🌍 1️⃣ PATH CONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_tuned.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "standard_scaler.joblib")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Ensure report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# 📦 2️⃣ LOAD DATA AND ARTIFACTS
# -----------------------------
print("🚀 Loading dataset and trained artifacts...")

df = pd.read_csv(DATA_PATH)
X = df.drop("target", axis=1)
y = df["target"]

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

print("✅ Data and models loaded successfully.\n")

# -----------------------------
# 📊 3️⃣ EVALUATION METRICS
# -----------------------------
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)
conf_matrix = confusion_matrix(y, y_pred)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": roc_auc
}

print("📈 MODEL PERFORMANCE METRICS")
print("-" * 35)
for k, v in metrics.items():
    print(f"{k:10s}: {v:.4f}")
print("\nConfusion Matrix:\n", conf_matrix, "\n")

# Save metrics to file
metrics_path = os.path.join(REPORT_DIR, "model_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("HEART DISEASE PREDICTION MODEL REPORT\n")
    f.write("=" * 45 + "\n\n")
    for k, v in metrics.items():
        f.write(f"{k:10s}: {v:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
print(f"📝 Metrics saved to: {metrics_path}\n")

# -----------------------------
# 📉 4️⃣ VISUALIZATION REPORTS
# -----------------------------
print("🎨 Generating visual reports...")

# 4.1 Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
conf_fig_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
plt.savefig(conf_fig_path)
plt.close()

# 4.2 ROC Curve
fpr, tpr, _ = roc_curve(y, y_prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
roc_fig_path = os.path.join(REPORT_DIR, "roc_curve.png")
plt.savefig(roc_fig_path)
plt.close()

# 4.3 Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)
plt.figure(figsize=(5, 4))
plt.plot(recall_vals, precision_vals, color="green", label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
pr_fig_path = os.path.join(REPORT_DIR, "precision_recall_curve.png")
plt.savefig(pr_fig_path)
plt.close()

print("✅ Visual reports saved:")
print(f"   • {conf_fig_path}")
print(f"   • {roc_fig_path}")
print(f"   • {pr_fig_path}\n")

# -----------------------------
# 🧩 5️⃣ SUMMARY REPORT
# -----------------------------
summary_path = os.path.join(REPORT_DIR, "summary_report.csv")
report_df = pd.DataFrame([metrics])
report_df.to_csv(summary_path, index=False)
print(f"📊 Summary CSV exported to: {summary_path}")

print("\n✅ Evaluation complete — all reports are stored in the /reports folder.")
