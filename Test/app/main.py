from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 👈 import this
from app.schemas import PatientData
from app.inference import predict

app = FastAPI(
    title="Heart Disease Predictor",
    description="Predicts heart disease likelihood based on patient parameters",
    version="1.0"
)

# ✅ Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later to ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_heart_disease(data: PatientData):
    result = predict(data.dict())
    return result

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Model running successfully"}

@app.get("/")
def root():
    return {"message": "Welcome Commander Jay! Heart Disease Predictor API is live. Visit /docs to test."}
