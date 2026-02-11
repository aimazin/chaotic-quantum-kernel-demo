import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# --- CORE RYDD-CIST LOGIC (Imported from your demo) ---
def chaotic_feature_map(X, beta=0.7, freq=3.0):
    """Encodes raw data into the chaotic SU(n) manifold."""
    X = np.asarray(X)
    fourier = np.sin(freq * np.pi * X) + np.cos(freq * np.pi * X**2)
    chaos = np.sum(np.log1p(np.abs(X)) ** beta, axis=1, keepdims=True)
    return np.hstack([fourier, chaos])

# --- API DATA SCHEMAS ---
class LiquidityData(BaseModel):
    # Expects a list of coordinate pairs, e.g., [[0.1, 0.5], [1.2, -0.3]]
    points: List[List[float]] = Field(
        ..., 
        example=[[0.5, 0.5], [-1.0, 0.2]],
        description="List of 2D coordinates representing systemic data points."
    )

class HealingResponse(BaseModel):
    status: str
    healed_dimensions: int
    chaos_coefficient: float
    mapped_vectors: List[List[float]]

# --- FASTAPI APP CONFIGURATION ---
app = FastAPI(
    title="RYDD-CIST QaaS API",
    description="Quantum-Inspired Chaotic Kernel for Systemic Healing",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "RYDD-CIST Gateway Active. System: Healthy."}

@app.post("/heal", response_model=HealingResponse)
async def heal_data(input_data: LiquidityData):
    """
    Applies the Chaotic Quantum Kernel to 'heal' (stabilize) 
    incoming high-entropy data.
    """
    try:
        # Convert input to NumPy
        X_raw = np.array(input_data.points)
        
        # 1. Apply the Chaotic Feature Map (Phase 2 Logic)
        X_mapped = chaotic_feature_map(X_raw)
        
        # 2. Return the stabilized (healed) manifold
        return {
            "status": "Success: Manifold Stabilized",
            "healed_dimensions": X_mapped.shape[1],
            "chaos_coefficient": 0.7, # Static for demo, dynamic in Phase 3
            "mapped_vectors": X_mapped.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- SERVER ENTRY POINT ---
if __name__ == "__main__":
    print("Launching RYDD-CIST QaaS Node on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)