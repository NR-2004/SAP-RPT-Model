import json
import logging
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator


# ---------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("po_delay")

app = FastAPI(title="SAP RPT-1 PO Delay Prediction")

# SAP AI Core (SAP-RPT-1) via BTP Destination service
DESTINATION_NAME = os.getenv("DESTINATION_NAME", "").strip()
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP", "default").strip()
DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID", "").strip()
DEST_SERVICE_KEY_PATH = BASE_DIR / os.getenv("DEST_SERVICE_KEY_PATH", "destination_service_key.json").strip()
RPT_CONTEXT_ROWS = int(os.getenv("RPT_CONTEXT_ROWS", "2000"))  # sap-rpt-1-small max 2048
RPT_BATCH_SIZE = 128                                           # max prediction rows per call

# SAP S/4HANA (purchase orders): fill in .env
SAP_BASE_URL = os.getenv("SAP_BASE_URL", "").strip().rstrip("/")
SAP_CLIENT = os.getenv("SAP_CLIENT", "").strip()
SAP_USERNAME = os.getenv("SAP_USERNAME", "").strip()
SAP_PASSWORD = os.getenv("SAP_PASSWORD", "")
VERIFY = os.getenv("SAP_VERIFY_SSL", "true").lower() == "true"

CSV_PATH = BASE_DIR / "historical_po_delay_training.csv"
PO_SERVICE = "API_PURCHASEORDER_PROCESS_SRV"
MATDOC_SERVICE = "API_MATERIAL_DOCUMENT_SRV"

INDEX = "RowId"
TARGET = "DelayDays"
PLACEHOLDER = "[PREDICT]"
# Sent to SAP-RPT-1. Tested on hold-out data: Supplier carries almost all the signal; Material, Plant,
# PurchasingOrg/Group, unit and month/weekday added noise (lower R2), so they are not sent.
FEATURES = ["Supplier", "LeadTimeDays", "OrderQuantity"]

# Never sent to the model: IDs, leakage fields, and constant/empty/duplicate columns.
EXCLUDED = ["PurchaseOrder", "PurchaseOrderItem", "ScheduleLine", "CompanyCode", "MaterialGroup", "ScheduleLineOrderQuantity", "ActualGoodsReceiptDate", "DeliveryStatus", "PurchaseOrderType"]

# Goods receipt movement types: +1 receipt, -1 reversal/return.
GR_MOVEMENTS = {"101": 1, "102": -1, "103": 1, "104": -1, "122": -1, "123": 1}

# Required inputs for this use case -> missing any of these means the app cannot run.
REQUIRED_ENV_VARS = [
    "DESTINATION_NAME", "DEPLOYMENT_ID",
    "SAP_BASE_URL", "SAP_USERNAME", "SAP_PASSWORD",
]
REQUIRED_CSV_COLUMNS = [
    "PurchaseOrder", "PurchaseOrderItem", "ScheduleLine", "Supplier", "Material", "PurchasingOrganization", "PurchasingGroup",
    "Plant", "PurchaseOrderType", "OrderQuantityUnit", "OrderQuantity",
    "CreationDate", "ScheduleLineDeliveryDate", "ActualGoodsReceiptDate", "DelayDays",
]


def validate_config():
    """Checks every required input (env vars, destination key file, CSV + its columns).
    Returns a list of problem strings; empty list = ready to run."""
    load_dotenv(BASE_DIR / ".env", override=True)
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


# ---------------------------------------------------------
# 2. REQUEST MODEL
# ---------------------------------------------------------

class PredictRequest(BaseModel):
    purchase_orders: list[str] | None = None   # 10-digit SAP PO numbers, exact format, e.g. "4500001234"
    created_from: date | None = None           # ISO yyyy-mm-dd
    created_to: date | None = None              # ISO yyyy-mm-dd
    level: str = "schedule_line"                # schedule_line | item
    max_pos: int = 100                          # newest N POs (SAP-RPT-1 handles 128 rows per call)

    @model_validator(mode="after")
    def check_level(self):
        if self.level not in ("schedule_line", "item"):
            raise ValueError("level must be 'schedule_line' or 'item'")
        return self


# ---------------------------------------------------------
# 3. GENERAL UTILITIES
# ---------------------------------------------------------

def sap_date(value):
    """Convert SAP OData V2 '/Date(ms)/' or ISO text to a date."""
    if value in (None, ""):
        return None
    match = re.fullmatch(r"/Date\((-?\d+)(?:[+-]\d{4})?\)/", str(value))
    if match:
        return datetime.fromtimestamp(int(match.group(1)) / 1000, tz=timezone.utc).date()
    return date.fromisoformat(str(value)[:10])


def esc(value):
    return str(value).replace("'", "''")


def norm_key(value):
    """'0017000105' -> '17000105' so SAP keys match the CSV."""
    text = str(value or "").strip()
    return (text.lstrip("0") or "0") if text.isdigit() else text.upper()


def or_filter(field, values):
    return "(" + " or ".join(f"{field} eq '{esc(v)}'" for v in values) + ")"


def status(delay):
    return "Delayed" if delay > 0 else "On Time" if delay == 0 else "Early"


# ---------------------------------------------------------
# 4. BTP DESTINATION SERVICE -> AI CORE URL AND TOKEN
# ---------------------------------------------------------

_cache = {"expires": 0.0}


def client_credentials_token(token_url, client_id, client_secret):
    response = requests.post(token_url, data={"grant_type": "client_credentials"},
                             auth=(client_id, client_secret), verify=True, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Token request failed ({response.status_code}): {response.text[:300]}")
    body = response.json()
    return body["access_token"], int(body.get("expires_in", 3600))


def ai_core_connection():
    """Read destination DESTINATION_NAME -> (AI API base URL, bearer token). Cached until expiry."""
    if time.time() < _cache["expires"]:
        return _cache["url"], _cache["token"]
    if not DESTINATION_NAME or not DEPLOYMENT_ID:
        raise RuntimeError("DESTINATION_NAME and DEPLOYMENT_ID must be set in .env")
    if not DEST_SERVICE_KEY_PATH.is_file():
        raise RuntimeError(f"Destination service key missing: {DEST_SERVICE_KEY_PATH}")

    key = json.loads(DEST_SERVICE_KEY_PATH.read_text(encoding="utf-8"))
    uaa = key.get("uaa") or key
    dest_token, _ = client_credentials_token(uaa["url"].rstrip("/") + "/oauth/token",
                                             uaa["clientid"], uaa["clientsecret"])
    response = requests.get(
        key["uri"].rstrip("/") + "/destination-configuration/v1/destinations/" + quote(DESTINATION_NAME, safe=""),
        headers={"Authorization": "Bearer " + dest_token}, verify=True, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Destination {DESTINATION_NAME} lookup failed ({response.status_code}): {response.text[:300]}")
    payload = response.json()
    config = payload.get("destinationConfiguration") or {}
    if not config.get("URL"):
        raise RuntimeError(f"Destination {DESTINATION_NAME} has no URL")

    # OAuth2ClientCredentials destinations return a ready AI Core token in authTokens.
    tokens = [t for t in payload.get("authTokens") or [] if t.get("value")]
    if tokens:
        if tokens[0].get("error"):
            raise RuntimeError(f"Destination token error: {tokens[0]['error']}")
        token, expires = tokens[0]["value"], int(tokens[0].get("expires_in", 3600))
    elif config.get("tokenServiceURL") and config.get("clientId"):
        token, expires = client_credentials_token(config["tokenServiceURL"], config["clientId"],
                                                  config["clientSecret"])
    else:
        raise RuntimeError(f"Destination {DESTINATION_NAME} provides no OAuth token or client credentials")

    url = config["URL"].rstrip("/")
    url = url[:-3] if url.endswith("/v2") else url
    _cache.update(url=url, token=token, expires=time.time() + expires - 120)
    return url, token


# ---------------------------------------------------------
# 5. SAP-RPT-1 INFERENCE
# ---------------------------------------------------------

def rpt_predict(context, queries):
    """In-context regression: known rows (context) + [PREDICT] rows -> {RowId: delay}."""
    base_url, token = ai_core_connection()
    endpoint = f"{base_url}/v2/inference/deployments/{DEPLOYMENT_ID}/predict"
    rows = ([{INDEX: str(r[INDEX]), **{c: r[c] for c in FEATURES}, TARGET: float(r[TARGET])}
             for r in context.to_dict("records")]
            + [{INDEX: str(r[INDEX]), **{c: r[c] for c in FEATURES}, TARGET: PLACEHOLDER}
               for r in queries.to_dict("records")])
    body = {
        "prediction_config": {"target_columns": [
            {"name": TARGET, "prediction_placeholder": PLACEHOLDER, "task_type": "regression"}]},
        "index_column": INDEX,
        "parse_data_types": True,
        "rows": rows,
    }
    response = requests.post(endpoint, json=body, verify=True, timeout=300, headers={
        "Authorization": "Bearer " + token,
        "AI-Resource-Group": RESOURCE_GROUP,
        "Content-Type": "application/json",
    })
    if not response.ok:
        raise RuntimeError(f"SAP-RPT-1 HTTP {response.status_code}: {response.text[:1000]}")

    result = {}
    for item in response.json().get("predictions", []):
        value = item.get(TARGET)
        while isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, dict):
            value = value.get("prediction")
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError("SAP-RPT-1 returned a non-finite prediction")
        result[str(item[INDEX])] = number
    missing = set(queries[INDEX].astype(str)) - set(result)
    if missing:
        raise RuntimeError(f"SAP-RPT-1 response lacks predictions for {len(missing)} rows: {response.text[:500]}")
    return result


def select_context(history, queries):
    """Most relevant history first: same supplier, then same material, then most recent."""
    suppliers, materials = set(queries["Supplier"]), set(queries["Material"])
    score = history["Supplier"].isin(suppliers) * 2 + history["Material"].isin(materials)
    ranked = history.assign(_score=score).sort_values(["_score", "CreationDate"], ascending=False)
    return ranked.head(RPT_CONTEXT_ROWS).drop(columns="_score")


def predict_delays(history, queries):
    """Batch queries (<=128 per call), each with its own relevant context. Returns array aligned to queries."""
    predictions = {}
    for start in range(0, len(queries), RPT_BATCH_SIZE):
        batch = queries.iloc[start:start + RPT_BATCH_SIZE]
        predictions.update(rpt_predict(select_context(history, batch), batch))
    return np.array([predictions[str(i)] for i in queries[INDEX]])


# ---------------------------------------------------------
# 6. HISTORICAL DATA (existing CSV, validated, not modified)
# ---------------------------------------------------------

def build_features(df):
    """Same feature logic for CSV rows and SAP rows."""
    out = pd.DataFrame(index=df.index)
    out["Supplier"] = df["Supplier"].map(norm_key)
    out["Material"] = df["Material"].map(norm_key)
    for col in ("PurchasingOrganization", "PurchasingGroup", "Plant", "OrderQuantityUnit"):
        out[col] = df[col].astype(str).str.strip().str.upper()
    planned = pd.to_datetime(df["ScheduleLineDeliveryDate"])
    out["OrderQuantity"] = pd.to_numeric(df["OrderQuantity"], errors="coerce").round(3)
    out["LeadTimeDays"] = (planned - pd.to_datetime(df["CreationDate"])).dt.days
    out["PlannedMonth"] = planned.dt.month
    out["PlannedDayOfWeek"] = planned.dt.day_of_week
    return out


def load_history():
    text_cols = ["PurchaseOrder", "PurchaseOrderItem", "ScheduleLine", "CompanyCode", "Supplier", "Material", "PurchasingOrganization",
                 "PurchasingGroup", "Plant", "PurchaseOrderType", "OrderQuantityUnit"]
    df = pd.read_csv(CSV_PATH, dtype={c: str for c in text_cols},
                     parse_dates=["CreationDate", "ScheduleLineDeliveryDate", "ActualGoodsReceiptDate"])
    mismatch = int(((df["ActualGoodsReceiptDate"] - df["ScheduleLineDeliveryDate"]).dt.days != df[TARGET]).sum())
    log.info("History rows: %s | missing values: %s | DelayDays/date mismatches: %s",
             len(df), int(df.isna().sum().sum()), mismatch)
    df = df.dropna(subset=["Supplier", "ScheduleLineDeliveryDate", "CreationDate", TARGET])
    history = build_features(df)
    history[INDEX] = "H" + df["PurchaseOrder"] + "-" + df["PurchaseOrderItem"] + "-" + df["ScheduleLine"]
    history["CreationDate"] = df["CreationDate"]
    history[TARGET] = df[TARGET]
    return history.sort_values("CreationDate").reset_index(drop=True)


# ---------------------------------------------------------
# 7. EVALUATION (time-based hold-out, no future data in context)
# ---------------------------------------------------------

def evaluate(test_rows=256):
    history = load_history()
    cut = int(len(history) * 0.8)
    past, future = history.iloc[:cut], history.iloc[cut:]
    test = future.sample(min(test_rows, len(future)), random_state=42)
    y_true = test[TARGET].to_numpy(float)
    y_pred = predict_delays(past, test)

    def scores(pred):
        err = y_true - pred
        return {"MAE": round(float(np.abs(err).mean()), 3),
                "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
                "R2": round(float(1 - (err ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()), 3)}

    return {
        "TestRows": len(test),
        "ContextRowsPerCall": min(RPT_CONTEXT_ROWS, len(past)),
        "Split": "train = oldest 80% of POs, test = sample of newest 20%",
        "SAP_RPT_1": scores(y_pred),
        "Baseline_AverageDelay": scores(np.full_like(y_true, past[TARGET].mean())),
        "Features": FEATURES,
        "Excluded": EXCLUDED,
    }


# ---------------------------------------------------------
# 8. SAP S/4HANA ODATA REQUEST HANDLER
# ---------------------------------------------------------

HTTP_HINTS = {
    401: "wrong credentials / client, or user locked",
    403: "authenticated but missing authorization (check SU53)",
    404: "wrong base URL or OData service not activated (/IWFND/MAINT_SERVICE)",
}


def sap_get(service, entity, params=None):
    # Re-read .env on every call so edits apply without restarting the server.
    load_dotenv(BASE_DIR / ".env", override=True)
    base_url = os.getenv("SAP_BASE_URL", "").strip().rstrip("/")
    client = os.getenv("SAP_CLIENT", "").strip()
    username = os.getenv("SAP_USERNAME", "").strip()
    password = os.getenv("SAP_PASSWORD", "")
    verify = os.getenv("SAP_VERIFY_SSL", "true").lower() == "true"
    missing = [n for n, v in (("SAP_BASE_URL", base_url), ("SAP_USERNAME", username),
                              ("SAP_PASSWORD", password)) if not v]
    if missing:
        raise RuntimeError(f"Fill these in .env: {', '.join(missing)}")
    query = {"$format": "json", **(params or {})}
    if client:
        query["sap-client"] = client
    try:
        response = requests.get(f"{base_url}/sap/opu/odata/sap/{service}/{entity}", params=query,
                                headers={"Accept": "application/json"}, auth=(username, password),
                                verify=verify, timeout=60)
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach SAP at {base_url} (network/VPN?): {exc}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"SAP HTTP {response.status_code} on {entity}: "
                           f"{HTTP_HINTS.get(response.status_code, response.text[:300])}")
    data = response.json()
    if "d" not in data:
        raise RuntimeError("Unexpected SAP OData response: missing d property")
    return data["d"]


def sap_rows(service, entity, select, filter_expr=None, expand=None, page_size=500, max_rows=50000,
             orderby=None, limit=None):
    """limit: stop after this many rows (no error); max_rows: hard safety cap."""
    output, skip = [], 0
    while True:
        top = page_size if limit is None else min(page_size, limit - len(output))
        params = {"$select": ",".join(select), "$top": str(top), "$skip": str(skip)}
        if orderby:
            params["$orderby"] = orderby
        if filter_expr:
            params["$filter"] = filter_expr
        if expand:
            params["$expand"] = expand
        batch = sap_get(service, entity, params).get("results", [])
        output.extend(batch)
        if len(output) > max_rows:
            raise RuntimeError(f"SAP result exceeded {max_rows} rows: {entity}")
        if len(batch) < top or (limit is not None and len(output) >= limit):
            return output
        skip += len(batch)


# ---------------------------------------------------------
# 9. NEW PURCHASE ORDERS (header + item + schedule line)
# ---------------------------------------------------------

def fetch_purchase_orders(po_numbers=None, created_from=None, created_to=None, level="schedule_line", max_pos=100):
    conditions = []
    if po_numbers:
        conditions.append(or_filter("PurchaseOrder", po_numbers))
    if created_from:
        conditions.append(f"CreationDate ge datetime'{created_from}T00:00:00'")
    if created_to:
        conditions.append(f"CreationDate le datetime'{created_to}T00:00:00'")

    headers = sap_rows(PO_SERVICE, "A_PurchaseOrder",
                       ["PurchaseOrder", "Supplier", "PurchasingOrganization", "PurchasingGroup",
                        "PurchaseOrderType", "CompanyCode", "CreationDate"],
                       " and ".join(conditions) or None, orderby="CreationDate desc,PurchaseOrder desc", limit=max_pos)
    header_by_po = {h["PurchaseOrder"]: h for h in headers}
    po_list = list(header_by_po)

    rows = []
    for i in range(0, len(po_list), 20):  # keep $filter URLs short
        items = sap_rows(
            PO_SERVICE, "A_PurchaseOrderItem",
            ["PurchaseOrder", "PurchaseOrderItem", "Material", "Plant", "OrderQuantity",
             "PurchaseOrderQuantityUnit", "MaterialGroup", "PurchasingDocumentDeletionCode",
             "to_ScheduleLine/ScheduleLine", "to_ScheduleLine/ScheduleLineDeliveryDate",
             "to_ScheduleLine/ScheduleLineOrderQuantity"],
            or_filter("PurchaseOrder", po_list[i:i + 20]), expand="to_ScheduleLine")

        for item in items:
            if item.get("PurchasingDocumentDeletionCode"):
                continue  # deleted / blocked item
            header = header_by_po[item["PurchaseOrder"]]
            lines = [l for l in (item.get("to_ScheduleLine") or {}).get("results", [])
                     if l.get("ScheduleLineDeliveryDate")]
            if not lines:
                continue
            if level == "item":  # item is complete when its last schedule line arrives
                last = max(lines, key=lambda l: sap_date(l["ScheduleLineDeliveryDate"]))
                lines = [{**last, "ScheduleLine": ",".join(str(l["ScheduleLine"]) for l in lines)}]
            for line in lines:
                rows.append({
                    "PurchaseOrder": item["PurchaseOrder"],
                    "PurchaseOrderItem": str(item["PurchaseOrderItem"]).lstrip("0"),
                    "ScheduleLine": str(line["ScheduleLine"]),
                    "Supplier": header["Supplier"],
                    "Material": item["Material"],
                    "PurchasingOrganization": header["PurchasingOrganization"],
                    "PurchasingGroup": header["PurchasingGroup"],
                    "PurchaseOrderType": header["PurchaseOrderType"],
                    "Plant": item["Plant"],
                    "OrderQuantity": float(item["OrderQuantity"] or 0),
                    "OrderQuantityUnit": item["PurchaseOrderQuantityUnit"],
                    "CreationDate": sap_date(header["CreationDate"]),
                    "ScheduleLineDeliveryDate": sap_date(line["ScheduleLineDeliveryDate"]),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(["PurchaseOrder", "PurchaseOrderItem", "ScheduleLine"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------
# 10. GOODS RECEIPTS (history validation only, never for prediction)
# ---------------------------------------------------------

def actual_gr_dates(po_numbers):
    """Actual GR date per PO item, net of reversals (102) and returns (122)."""
    items = sap_rows(MATDOC_SERVICE, "A_MaterialDocumentItem",
                     ["PurchaseOrder", "PurchaseOrderItem", "Material", "Plant", "GoodsMovementType",
                      "PostingDate", "MaterialDocument", "MaterialDocumentYear", "MaterialDocumentItem",
                      "QuantityInEntryUnit", "EntryUnit"],
                     f"{or_filter('PurchaseOrder', po_numbers)} and {or_filter('GoodsMovementType', list(GR_MOVEMENTS))}")
    df = pd.DataFrame(items)
    if df.empty:
        return {}
    df = df.drop_duplicates(["MaterialDocumentYear", "MaterialDocument", "MaterialDocumentItem"])
    df["PostingDate"] = df["PostingDate"].map(sap_date)
    df["Net"] = df["QuantityInEntryUnit"].astype(float) * df["GoodsMovementType"].map(GR_MOVEMENTS)

    result = {}
    for (po, item), g in df.groupby(["PurchaseOrder", "PurchaseOrderItem"]):
        daily = g.groupby("PostingDate")["Net"].sum().sort_index().cumsum()
        if daily.iloc[-1] <= 0:
            continue  # fully reversed
        zero = daily[daily <= 0]
        valid = daily[daily.index > zero.index[-1]] if len(zero) else daily
        result[(po, str(item).lstrip("0"))] = str(valid.index[0])
    return result


# ---------------------------------------------------------
# 11. PREDICTION FOR NEW POs
# ---------------------------------------------------------

def predict(rows):
    if rows.empty:
        return []
    history = load_history()
    queries = build_features(rows)
    queries[INDEX] = "Q" + rows.index.astype(str)
    delays = np.rint(predict_delays(history, queries)).astype(int)

    known_suppliers, known_materials = set(history["Supplier"]), set(history["Material"])
    output = []
    for (_, row), (_, f), delay in zip(rows.iterrows(), queries.iterrows(), delays):
        notes = []
        if f["Supplier"] not in known_suppliers:
            notes.append("new supplier")
        if f["Material"] not in known_materials:
            notes.append("material not in history")
        if row["PurchaseOrderType"] != "NB":
            notes.append(f"PO type {row['PurchaseOrderType']} not in history")
        output.append({
            "PurchaseOrder": row["PurchaseOrder"],
            "PurchaseOrderItem": row["PurchaseOrderItem"],
            "ScheduleLine": row["ScheduleLine"],
            "Supplier": f["Supplier"],
            "Material": f["Material"],
            "Plant": row["Plant"],
            "OrderQuantity": row["OrderQuantity"],
            "OrderQuantityUnit": row["OrderQuantityUnit"],
            "PlannedDeliveryDate": str(row["ScheduleLineDeliveryDate"]),
            "PredictedDelayDays": int(delay),
            "PredictedDeliveryDate": str(row["ScheduleLineDeliveryDate"] + timedelta(days=int(delay))),
            "PredictedDeliveryStatus": status(delay),
            "Notes": "; ".join(notes),
        })
    return output


# ---------------------------------------------------------
# 12. ENDPOINTS
# ---------------------------------------------------------

ERRORS = (requests.RequestException, RuntimeError, KeyError, ValueError, TypeError)


@app.get("/config-check")
def config_check():
    """Validates every required input (env vars, destination key, CSV + columns) with no SAP/AI Core calls."""
    problems = validate_config()
    return {"ready": not problems, "missing": problems}


@app.get("/health")
def health():
    return {"status": "ok", "model": "SAP-RPT-1 (AI Core)", "deployment_id": DEPLOYMENT_ID,
            "resource_group": RESOURCE_GROUP, "s4_configured": bool(load_dotenv(BASE_DIR / ".env", override=True) and os.getenv("SAP_BASE_URL"))}


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


@app.get("/goods-receipts/{purchase_order}")
def goods_receipts(purchase_order: str):
    require_config()
    try:
        dates = actual_gr_dates([purchase_order])
        return {"PurchaseOrder": purchase_order,
                "Items": [{"PurchaseOrderItem": item, "ActualGoodsReceiptDate": d}
                          for (_, item), d in dates.items()]}
    except ERRORS as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------
# 13. LOCAL RUN:  python main.py   |   python main.py evaluate [rows]   |   python main.py config-check
# ---------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "config-check":
        print(json.dumps({"ready": not validate_config(), "missing": validate_config()}, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        print(json.dumps(evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 256), indent=2))
    else:
        import threading
        import uvicorn
        import webbrowser
        import socket

        problems = validate_config()
        if problems:
            sys.exit("\n  Missing required inputs:\n  - " + "\n  - ".join(problems) + "\n")

        host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                sys.exit(f"\n  Port {port} is already in use - the server is probably already running "
                         f"in another terminal.\n  Use that one ({f'http://127.0.0.1:{port}/docs'}), "
                         f"stop it with CTRL+C, or set PORT=8001 in .env.\n")
        docs = f"http://127.0.0.1:{port}/docs"
        print(f"\n  Open in browser: {docs}\n  Stop with CTRL+C\n", flush=True)
        threading.Timer(2, webbrowser.open, [docs]).start()
        uvicorn.run(app, host=host, port=port)