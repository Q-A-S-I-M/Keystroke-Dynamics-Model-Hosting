from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load trained objects
mlp = joblib.load("mlp_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")

features = [
    "avgHoldTime","medianIKD","holdTimeStdDev","tempoChangeRate",
    "typingSpeedWPM","entropyIKD","maxBurstLength","commonDigraphTiming",
    "skewnessIKD","ikdStdDev","correctionLatencyMean","backspaceRatio"
]

app = FastAPI(title="Keystroke MLP Prediction API")

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_user(request: PredictRequest):
    if len(request.features) != len(features):
        return {"error": f"Expected {len(features)} features, got {len(request.features)}"}
    
    data = np.array(request.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    pred_label_enc = mlp.predict(data_scaled)[0]
    pred_user = encoder.inverse_transform([pred_label_enc])[0]

    return {"predicted_user": pred_user}
