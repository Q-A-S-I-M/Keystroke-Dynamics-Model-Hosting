from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import numpy as np
import joblib

mlp = joblib.load("mlp_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")

FEATURE_KEYS = [
    "avgHoldTime", "medianIKD", "holdTimeStdDev", "tempoChangeRate",
    "typingSpeedWPM", "entropyIKD", "maxBurstLength", "commonDigraphTiming",
    "skewnessIKD", "ikdStdDev", "correctionLatencyMean", "backspaceRatio"
]

app = FastAPI(title="Keystroke MLP Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeatureData(BaseModel):
    avgHoldTime: float
    medianIKD: float
    holdTimeStdDev: float
    tempoChangeRate: float
    typingSpeedWPM: float
    entropyIKD: float
    maxBurstLength: float
    commonDigraphTiming: float
    skewnessIKD: float
    ikdStdDev: float
    correctionLatencyMean: float
    backspaceRatio: float

@app.post("/predict")
def predict_user(data: FeatureData):
    try:
        # 1. Extract values in the specific order your model expects
        feature_dict = data.dict()
        ordered_features = [feature_dict[key] for key in FEATURE_KEYS]
        
        # 2. Process and Predict
        input_data = np.array(ordered_features).reshape(1, -1)
        data_scaled = scaler.transform(input_data)
        
        pred_label_enc = mlp.predict(data_scaled)[0]
        pred_user = encoder.inverse_transform([pred_label_enc])[0]

        return {"predicted_user": str(pred_user)}
    
    except Exception as e:
        return {"error": str(e)}
