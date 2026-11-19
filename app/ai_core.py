"""SAP AI Core access (BTP destination -> URL + token) and SAP-RPT-1 inference."""

import json
import math
import time
from urllib.parse import quote

import numpy as np
import requests

from app.config import (DEPLOYMENT_ID, DEST_SERVICE_KEY_PATH, DESTINATION_NAME, FEATURES, INDEX, PLACEHOLDER,
                        RESOURCE_GROUP, RPT_BATCH_SIZE, RPT_CONTEXT_ROWS, TARGET, log)

_cache = {"expires": 0.0}


# ---------------------------------------------------------
# BTP DESTINATION SERVICE -> AI CORE URL AND TOKEN
# ---------------------------------------------------------

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
# SAP-RPT-1 INFERENCE
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
    for attempt in range(3):   # AI Core sometimes answers 500/502/503/504 under load -> retry
        response = requests.post(endpoint, json=body, verify=True, timeout=300, headers={
            "Authorization": "Bearer " + token,
            "AI-Resource-Group": RESOURCE_GROUP,
            "Content-Type": "application/json",
        })
        if response.status_code not in (500, 502, 503, 504):
            break
        log.warning("SAP-RPT-1 HTTP %s, retrying (%s/3)", response.status_code, attempt + 1)
        time.sleep(5 * (attempt + 1))
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
    ranked = history.assign(_score=score).sort_values(["_score", "PurchaseOrderDate"], ascending=False)
    return ranked.head(RPT_CONTEXT_ROWS).drop(columns="_score")


def predict_delays(history, queries):
    """Batch queries (<=128 per call), each with its own relevant context. Returns array aligned to queries."""
    predictions = {}
    for start in range(0, len(queries), RPT_BATCH_SIZE):
        batch = queries.iloc[start:start + RPT_BATCH_SIZE]
        predictions.update(rpt_predict(select_context(history, batch), batch))
    return np.array([predictions[str(i)] for i in queries[INDEX]])
