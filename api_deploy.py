import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List

# --- CORE RYDD-CIST LOGIC ---
def chaotic_feature_map(X: np.ndarray, beta: float = 0.7, freq: float = 3.0) -> np.ndarray:
    """The Engine: Maps raw data into the stable chaotic SU(n) manifold."""
    X = np.asarray(X)
    fourier = np.sin(freq * np.pi * X) + np.cos(freq * np.pi * X**2)
    chaos = np.sum(np.log1p(np.abs(X)) ** beta, axis=1, keepdims=True)
    return np.hstack([fourier, chaos])

# --- DATA SCHEMAS ---
class LiquidityData(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"points": [[0.5, 0.1], [-1.2, 0.8]]}]})
    points: List[List[float]] = Field(..., min_length=1)

class HealingResponse(BaseModel):
    status: str
    mapped_vectors: List[List[float]]

# --- API GATEWAY ---
app = FastAPI(title="RYDD-CIST QaaS Node")

@app.post("/heal", response_model=HealingResponse)
async def heal_manifold(input_data: LiquidityData):
    try:
        X_raw = np.array(input_data.points)
        X_mapped = chaotic_feature_map(X_raw)
        return {
            "status": "Success: Manifold Stabilized",
            "mapped_vectors": X_mapped.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)