# 🚀 Heart Disease Risk Prediction System (CVD)
### ML-powered API and Web Client for Early, Explainable Cardiac Risk Screening

---

## 📌 Project Overview

Cardiovascular Disease (CVD) is one of the leading causes of mortality worldwide. Many patients are diagnosed only after symptoms become severe, reducing the effectiveness of preventive care. Traditional screening approaches often lack **speed**, **consistency**, and **scalability**.

This project presents an **end-to-end Heart Disease Risk Prediction System** that leverages **Machine Learning** and **modern web technologies** to predict cardiac risk using routinely collected clinical parameters.

The system is designed to be:
- **Accurate** – trained on validated heart disease datasets  
- **Explainable** – probability-based risk interpretation  
- **Deployable** – REST API + web interface  
- **Reusable** – modular and extensible architecture  

---

## 🎯 Objectives

- Predict the likelihood of heart disease using clinical data
- Provide early risk assessment to support preventive healthcare
- Offer an easy-to-use interface for non-technical users
- Build a production-ready ML pipeline suitable for academic and portfolio evaluation

---

## 🧠 Key Features

- ✔️ End-to-end ML pipeline (data → model → prediction)
- ✔️ Supervised ML model for binary heart disease classification
- ✔️ FastAPI-based REST API for real-time inference
- ✔️ Web-based UI for entering patient details
- ✔️ Reproducible training with standardized preprocessing
- ✔️ Rich evaluation metrics and visual reports
- ✔️ Modular, scalable, and clean project structure

---

## 🏗️ System Architecture & Workflow

### 🔹 High-Level Architecture

User (Web UI)  
↓  
Frontend (HTML / CSS / JS)  
↓  
FastAPI Backend  
↓  
Preprocessing (StandardScaler)  
↓  
ML Model (Logistic Regression)  
↓  
Risk Prediction + Probability  

---

### 🔁 Workflow Explanation

#### 1️⃣ Data Collection
- Clinical datasets containing attributes such as:
  - Age
  - Sex
  - Chest pain type
  - Blood pressure
  - Cholesterol
  - ECG indicators
  - Heart rate, etc.

#### 2️⃣ Data Preprocessing
- Handling missing values
- Feature selection
- Feature scaling using **StandardScaler**

#### 3️⃣ Model Training (Offline)
- Supervised ML algorithm (Logistic Regression)
- Train–test split
- Model persistence using `joblib`

#### 4️⃣ Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- ROC–AUC
- Confusion Matrix
- ROC & Precision–Recall curves

#### 5️⃣ Inference (Online)
- API loads trained model & scaler
- Accepts JSON input
- Returns risk probability and classification

#### 6️⃣ User Interaction
- User enters patient parameters via web UI
- Prediction displayed with interpretation

---

## ⚙️ Technology Stack

### 🧪 Machine Learning
- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib

### 🔧 Backend
- FastAPI
- Uvicorn
- Pydantic

### 🎨 Frontend
- HTML5
- CSS3
- JavaScript (Vanilla)

### 🛠️ Tools
- Jupyter Notebook
- Cursor / VS Code
- Git & GitHub
- Virtual Environment (venv)

---

## 📂 Project Structure

```text
CD/
├── Test/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── inference.py         # Model loading & prediction
│   │   ├── schemas.py           # Pydantic schemas
│   │   └── requirements.txt
│   ├── data/
│   │   └── heart.csv            # Training dataset
│   ├── models/
│   │   ├── logistic_tuned.joblib
│   │   └── standard_scaler.joblib
│   ├── frontend/
│   │   ├── index.html
│   │   ├── script.js
│   │   └── style.css
│   ├── reports/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── precision_recall_curve.png
│   │   ├── model_metrics.txt
│   │   └── summary_report.csv
│   ├── train_model.py
│   └── evaluate_model.py
├── main/
│   └── Copy_of_Heart_Disease_Predictions.ipynb  # EDA & experimentation
└── README.md
```

---

## ▶️ Setup & Execution Guide

### ✅ Prerequisites

- Python 3.9+
- `pip`
- Git
- Web browser

### 1️⃣ Create Virtual Environment

From the project root:

```bash
cd Test
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r app/requirements.txt
```

### 3️⃣ Train the Model

```bash
python train_model.py
```

This will:
- Train the ML model  
- Save model & scaler to `models/`  

### 4️⃣ Evaluate the Model (Optional)

```bash
python evaluate_model.py
```

Evaluation reports and plots will be saved in `reports/`.

### 5️⃣ Start the FastAPI Server

```bash
cd app
uvicorn main:app --reload
```

API available at:
- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

### 6️⃣ Run the Frontend

In a new terminal:

```bash
cd Test/frontend
python -m http.server 5500
```

Open in browser:

```text
http://127.0.0.1:5500
```

---

## 🧪 Sample API Request

```json
{
  "age": 54,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 2
}
```

---

## 📊 Output

- Binary Prediction: **Low Risk / High Risk**
- Risk Probability: value between **0 and 1**
- Visual evaluation metrics
- Human-readable risk interpretation

---

## 📦 Datasets

Due to GitHub size limits, datasets are not included in this repository.

Recommended sources:
- UCI Machine Learning Repository
- Kaggle – Heart Disease datasets

Place CSV files inside:

```bash
Test/data/
```

---

## 🔐 Disclaimer

⚠️ This project is for **educational and research purposes only**.  
It must **not** be used for real-world clinical diagnosis or treatment decisions.

---

## 🚧 Future Enhancements

- SHAP / LIME explainability
- Advanced models (XGBoost, CatBoost)
- Dockerization & CI/CD
- Authentication & security layers
- Enhanced UI/UX
- Cloud deployment

---

## 🤝 Contributions

Contributions are welcome.  
Please fork the repository, create a feature branch, and submit a pull request with proper documentation.

---

## 📜 License

This project is licensed under the **MIT License**.