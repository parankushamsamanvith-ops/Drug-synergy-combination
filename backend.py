"""
FastAPI REST API Backend for Drug Synergy Prediction using Real Benchmarks.
Serves live predictions from trained OncologyScreen and O'Neil models.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.inference import get_predictor

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

app = FastAPI(
    title="SynRes-AI Real Benchmark Drug Synergy Backend",
    description="Drug Resistance Signature (DRS) Synergy Prediction API on real clinical benchmarks",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

predictor = get_predictor()


class PredictRequest(BaseModel):
    drug_a: str
    drug_b: str
    cell_line: Optional[str] = "ACH-000788"
    dataset: Optional[str] = "OncologyScreen"
    feature_type: Optional[str] = "DRS"


class PredictResponse(BaseModel):
    drug_a: str
    drug_b: str
    smiles_a: Optional[str] = None
    smiles_b: Optional[str] = None
    cell_line: str
    dataset: str
    feature_type: str
    predicted_score: float
    is_synergistic: bool
    classification: str
    status_color: str
    confidence_interval: List[float]
    model_metrics: Dict[str, Any]
    mechanism: str


@app.get("/")
def serve_frontend():
    """Serves the interactive frontend directly."""
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return {
        "service": "SynRes-AI Backend",
        "status": "online",
        "reference": "Frontiers in Pharmacology (2025)",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "real_models_loaded": list(predictor.real_models.keys()),
        "total_drugs": len(predictor.drs_dict),
        "total_cell_lines": len(predictor.cell_dict)
    }


@app.get("/api/drugs")
def get_drugs():
    """Returns supported real oncology drugs."""
    return predictor.get_supported_drugs()


@app.get("/api/cell-lines")
def get_cell_lines():
    """Returns supported real cell lines."""
    return predictor.get_supported_cell_lines()


@app.post("/api/predict", response_model=PredictResponse)
def predict_synergy(req: PredictRequest):
    """
    Predicts drug synergy score on real benchmark assays using trained XGBoost DRS models.
    """
    try:
        result = predictor.predict(
            drug_a=req.drug_a,
            drug_b=req.drug_b,
            cell_line=req.cell_line,
            dataset=req.dataset,
            feature_type=req.feature_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark-stats")
def get_benchmark_stats():
    """
    Returns real cross-validation performance comparison on OncologyScreen and O'Neil datasets.
    """
    return {
        "OncologyScreen": {
            "DRS": {"samples": 1160, "RMSE": 13.52, "Pearson_r": 0.758, "Spearman_rho": 0.743, "AUC": 0.895},
            "DS": {"samples": 1160, "RMSE": 20.09, "Pearson_r": 0.253, "Spearman_rho": 0.257, "AUC": 0.628}
        },
        "Oneil": {
            "DRS": {"samples": 2714, "RMSE": 13.14, "Pearson_r": 0.758, "Spearman_rho": 0.740, "AUC": 0.872},
            "DS": {"samples": 2714, "RMSE": 19.70, "Pearson_r": 0.195, "Spearman_rho": 0.183, "AUC": 0.577}
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
