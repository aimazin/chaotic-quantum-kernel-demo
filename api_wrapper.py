from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
# Import your chaotic_feature_map and model here

app = FastAPI(title="RYDD-CIST Chaotic Kernel API")

class DataInput(BaseModel):
    features: list  # e.g., [[x1, y1], [x2, y2]]

@app.post("/predict")
async def predict_liquidity_state(data: DataInput):
    X = np.array(data.features)
    # 1. Apply your Chaotic Mapping
    X_mapped = chaotic_feature_map(X)
    # 2. Run Inference
    prediction = clf.predict(X_mapped)
    return {"state": prediction.tolist(), "status": "healed"}