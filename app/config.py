"""Environment configuration, constants and input validation."""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("po_delay")

# SAP AI Core (SAP-RPT-1) via BTP Destination service
DESTINATION_NAME = os.getenv("DESTINATION_NAME", "").strip()
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP", "default").strip()
DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID", "").strip()
DEST_SERVICE_KEY_PATH = BASE_DIR / os.getenv("DEST_SERVICE_KEY_PATH", "destination_service_key.json").strip()
RPT_CONTEXT_ROWS = int(os.getenv("RPT_CONTEXT_ROWS", "2000"))  # sap-rpt-1-small max 2048
RPT_BATCH_SIZE = 128                                           # max prediction rows per call

# Training data: completed PO items from this S/4 system. Override with TRAINING_CSV in .env.
CSV_PATH = BASE_DIR / os.getenv("TRAINING_CSV", "po_history/po_delay_training_s4.csv")
PO_SERVICE = "API_PURCHASEORDER_PROCESS_SRV"

INDEX = "RowId"
TARGET = "DelayDays"
PLACEHOLDER = "[PREDICT]"
# Sent to SAP-RPT-1. Chosen on hold-out tests with the S/4 history (po_delay_training_s4.csv):
# this set had the lowest MAE; PurchasingOrg/Group, Plant, unit, month and weekday added noise.
FEATURES = ["Supplier", "Material", "MaterialGroup", "LeadTimeDays", "OrderQuantity"]
# History rows with |DelayDays| above this are treated as data errors (e.g. receipt years later) and skipped.
MAX_ABS_DELAY = int(os.getenv("MAX_ABS_DELAY", "180"))

# Never sent to the model: IDs, leakage fields, and constant/empty/duplicate columns.
EXCLUDED = ["PurchaseOrder", "PurchaseOrderItem", "ScheduleLine", "CompanyCode", "ScheduleLineOrderQuantity",
            "ActualGoodsReceiptDate", "DeliveryStatus", "PurchaseOrderType"]

# Required inputs for this use case -> missing any of these means the app cannot run.
REQUIRED_ENV_VARS = [
    "DESTINATION_NAME", "DEPLOYMENT_ID",
    "SAP_BASE_URL", "SAP_USERNAME", "SAP_PASSWORD",
]
REQUIRED_CSV_COLUMNS = [
    "PurchaseOrder", "PurchaseOrderItem", "ScheduleLine", "Supplier", "Material", "PurchasingOrganization",
    "PurchasingGroup", "Plant", "PurchaseOrderType", "MaterialGroup", "OrderQuantityUnit", "OrderQuantity",
    "PurchaseOrderDate", "CreationDate", "ScheduleLineDeliveryDate", "ActualGoodsReceiptDate", "DelayDays",
]


def reload_env():
    """Re-read .env so edits apply without restarting the server."""
    load_dotenv(BASE_DIR / ".env", override=True)


def validate_config():
    """Checks every required input (env vars, destination key file, CSV + its columns).
    Returns a list of problem strings; empty list = ready to run."""
    reload_env()
    problems = []

    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var, "").strip():
            problems.append(f"Missing .env value: {var}")

    if not DEST_SERVICE_KEY_PATH.is_file():
        problems.append(f"Destination service key not found: {DEST_SERVICE_KEY_PATH}")
    else:
        try:
            key = json.loads(DEST_SERVICE_KEY_PATH.read_text(encoding="utf-8"))
            uaa = key.get("uaa") or key
            for field in ("url", "clientid", "clientsecret"):
                if not uaa.get(field):
                    problems.append(f"Destination service key missing uaa.{field}")
            if not key.get("uri"):
                problems.append("Destination service key missing 'uri'")
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"Destination service key unreadable: {exc}")

    if not CSV_PATH.is_file():
        problems.append(f"Historical CSV not found: {CSV_PATH}")
    else:
        try:
            header = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
            missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in header]
            if missing_cols:
                problems.append(f"CSV missing columns: {', '.join(missing_cols)}")
        except (pd.errors.ParserError, OSError) as exc:
            problems.append(f"CSV unreadable: {exc}")

    return problems


def require_config():
    problems = validate_config()
    if problems:
        raise HTTPException(status_code=400, detail={"ready": False, "missing": problems})
