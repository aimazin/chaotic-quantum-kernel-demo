import uvicorn
import numpy as np
import threading
import nest_asyncio
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from typing import List
from pyngrok import ngrok

# --- 1. SETUP ENVIRONMENT ---
# Allows FastAPI to run inside a notebook environment
nest_asyncio.apply()

# --- 2. RYDD-CIST CORE LOGIC ---
def chaotic_feature_map(X: np.ndarray, beta: float = 0.7, freq: float = 3.0) -> np.ndarray:
    X = np.asarray(X)
    fourier = np.sin(freq * np.pi * X) + np.cos(freq * np.pi * X**2)
    chaos = np.sum(np.log1p(np.abs(X)) ** beta, axis=1, keepdims=True)
    return np.hstack([fourier, chaos])

# --- 3. PRODUCTION SCHEMAS (Pydantic v2.10 Standard) ---
class LiquidityData(BaseModel):
    # Modern Pydantic v2 metadata handling
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"points": [[0.5, 0.1], [-1.2, 0.8]]}]
        }
    )
    points: List[List[float]] = Field(..., min_length=1)

class HealingResponse(BaseModel):
    status: str
    healed_dimensions: int
    mapped_vectors: List[List[float]]

# --- 4. THE API GATEWAY ---
app = FastAPI(title="RYDD-CIST QaaS Gateway")

@app.post("/heal", response_model=HealingResponse)
async def heal_manifold(input_data: LiquidityData):
    try:
        X_raw = np.array(input_data.points)
        X_mapped = chaotic_feature_map(X_raw)
        return {
            "status": "Manifold Stabilized",
            "healed_dimensions": X_mapped.shape[1],
            "mapped_vectors": X_mapped.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. BACKGROUND DEPLOYMENT ---
def launch_qaas(token: str):
    # Authenticate and create tunnel
    ngrok.set_auth_token(token)
    public_url = ngrok.connect(8000).public_url
    print(f"\n🚀 QaaS IS LIVE: {public_url}")
    print(f"🔗 Documentation: {public_url}/docs\n")
    
    # Run Uvicorn in background thread
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="error")
    server = uvicorn.Server(config)
    
    threading.Thread(target=server.run).start()

# --- START THE ENGINE ---
# Replace 'YOUR_NGROK_TOKEN' with your actual token
NGROK_TOKEN = 39X6wCA9wuGCr8RwqwaOXRlwkJh_7MzzTfinwSnUAaQMESsd9 
launch_qaas(NGROK_TOKEN)