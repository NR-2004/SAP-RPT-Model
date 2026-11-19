"""FastAPI app and endpoints."""

import os
from datetime import date, timedelta

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

from app.config import DEPLOYMENT_ID, RESOURCE_GROUP, log, reload_env, require_config, validate_config
from app.evaluation import evaluate
from app.predictor import predict
from app.s4_client import fetch_purchase_orders
from app.schemas import PredictDataRequest, PredictRequest

app = FastAPI(title="SAP RPT-1 PO Delay Prediction")

ERRORS = (requests.RequestException, RuntimeError, KeyError, ValueError, TypeError)


@app.get("/config-check")
def config_check():
    """Validates every required input (env vars, destination key, CSV + columns) with no SAP/AI Core calls."""
    problems = validate_config()
    return {"ready": not problems, "missing": problems}


@app.get("/health")
def health():
    reload_env()
    return {"status": "ok", "model": "SAP-RPT-1 (AI Core)", "deployment_id": DEPLOYMENT_ID,
            "resource_group": RESOURCE_GROUP, "s4_configured": bool(os.getenv("SAP_BASE_URL"))}


@app.get("/evaluate")
def evaluate_endpoint(test_rows: int = 256):
    require_config()
    try:
        return evaluate(test_rows)
    except ERRORS as exc:
        log.exception("Evaluation failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/predict-delay")
def predict_delay(request: PredictRequest):
    require_config()
    try:
        created_from = request.created_from
        if not request.purchase_orders and not created_from:
            created_from = date.today() - timedelta(days=7)
        rows = fetch_purchase_orders(request.purchase_orders, created_from, request.created_to, request.level,
                                     request.max_pos)
        results = predict(rows)
        return {"Count": len(results), "Model": "SAP-RPT-1", "Predictions": results}
    except ERRORS as exc:
        log.exception("PO delay prediction failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/predict-delay-data")
def predict_delay_data(request: PredictDataRequest):
    """Same prediction, but PO lines come from the request body instead of S/4 (for testing)."""
    require_config()
    if not request.items:
        raise HTTPException(status_code=400, detail="items is empty")
    try:
        rows = pd.DataFrame([item.model_dump() for item in request.items])
        rows["PurchaseOrderDate"] = rows["PurchaseOrderDate"].fillna(rows["CreationDate"])
        results = predict(rows)
        return {"Count": len(results), "Model": "SAP-RPT-1", "Predictions": results}
    except ERRORS as exc:
        log.exception("PO delay prediction (data) failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
