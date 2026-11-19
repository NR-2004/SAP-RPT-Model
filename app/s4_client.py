"""SAP S/4HANA OData access and fetching new purchase orders (header + item + schedule line)."""

import os

import pandas as pd
import requests

from app.config import PO_SERVICE, reload_env
from app.utils import or_filter, sap_date

HTTP_HINTS = {
    401: "wrong credentials / client, or user locked",
    403: "authenticated but missing authorization (check SU53)",
    404: "wrong base URL or OData service not activated (/IWFND/MAINT_SERVICE)",
}


# ---------------------------------------------------------
# ODATA REQUEST HANDLER
# ---------------------------------------------------------

def sap_get(service, entity, params=None):
    # Re-read .env on every call so edits apply without restarting the server.
    reload_env()
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
# NEW PURCHASE ORDERS
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
                        "PurchaseOrderType", "CompanyCode", "CreationDate", "PurchaseOrderDate"],
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
                    "MaterialGroup": item["MaterialGroup"],
                    "OrderQuantity": float(item["OrderQuantity"] or 0),
                    "OrderQuantityUnit": item["PurchaseOrderQuantityUnit"],
                    "CreationDate": sap_date(header["CreationDate"]),
                    "PurchaseOrderDate": sap_date(header["PurchaseOrderDate"] or header["CreationDate"]),
                    "ScheduleLineDeliveryDate": sap_date(line["ScheduleLineDeliveryDate"]),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(["PurchaseOrder", "PurchaseOrderItem", "ScheduleLine"]).reset_index(drop=True)
    return df
