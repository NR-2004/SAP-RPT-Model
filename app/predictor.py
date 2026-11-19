"""Delay prediction for new PO lines."""

from datetime import timedelta

import numpy as np

from app.ai_core import predict_delays
from app.config import INDEX
from app.history import build_features, load_history
from app.utils import status


def predict(rows):
    if rows.empty:
        return []
    history = load_history()
    queries = build_features(rows)
    queries[INDEX] = "Q" + rows.index.astype(str)
    delays = np.rint(predict_delays(history, queries)).astype(int)
    # A delivery cannot arrive before the PO exists: cap early predictions at the PO date.
    lead = queries["LeadTimeDays"].to_numpy()
    capped = (lead >= 0) & (delays < -lead)
    delays = np.where(capped, -lead, delays)

    known_suppliers, known_materials = set(history["Supplier"]), set(history["Material"])
    known_types = set(history["PurchaseOrderType"])
    output = []
    for (_, row), (_, f), delay, cap in zip(rows.iterrows(), queries.iterrows(), delays, capped):
        notes = ["capped at PO date"] if cap else []
        if f["Supplier"] not in known_suppliers:
            notes.append("new supplier")
        if f["Material"] not in known_materials:
            notes.append("material not in history")
        if row["PurchaseOrderType"] not in known_types:
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
