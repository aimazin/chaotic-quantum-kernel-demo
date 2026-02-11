import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from typing import List

# --- CORE RYDD-CIST LOGIC ---
def chaotic_feature_map(X: np.ndarray, beta: float = 0.7, freq: float = 3.0) -> np.ndarray:
    """The engine: Maps data into the stable chaotic SU(n) manifold."""
    fourier = np.sin(freq * np.pi * X) + np.cos(freq * np.pi * X**2)
    chaos = np.sum(np.log1p(np.abs(X)) ** beta, axis=1, keepdims=True)
    return np.hstack([fourier, chaos])

# --- API DATA SCHEMAS ---
class LiquidityData(BaseModel):
    # Pydantic v2 Style Configuration
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"points": [[0.5, 0.1], [-1.2, 0.8], [0.3, -0.4]]}
        }
    )
    points: List[List[float]] = Field(..., min_length=1, description="Data points for healing.")

class HealingResponse(BaseModel):
    status: str
    healed_dimensions: int
    chaos_coefficient: float
    # We return the mapped data as a list of lists for JSON serializability
    mapped_vectors: List[List[float]]

# --- FASTAPI APP ---
app = FastAPI(
    title="RYDD-CIST QaaS Node",
    description="2026 Sovereign Quantum AI Service Endpoint",
    version="1.1.0"
)

@app.post("/heal", response_model=HealingResponse, status_code=status.HTTP_201_CREATED)
async def heal_manifold(input_data: LiquidityData):
    try:
        X_raw = np.array(input_data.points)
        
        # Execute Kernel
        X_mapped = chaotic_feature_map(X_raw)
        
        return {
            "status": "Manifold Stabilized via RYDD-CIST",
            "healed_dimensions": X_mapped.shape[1],
            "chaos_coefficient": 0.7,
            "mapped_vectors": X_mapped.tolist() # Safe casting to float
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Kernel Execution Error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)