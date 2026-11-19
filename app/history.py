"""Historical training data (CSV) and the shared feature logic."""

import pandas as pd

from app.config import CSV_PATH, INDEX, MAX_ABS_DELAY, TARGET, log
from app.utils import norm_key


def build_features(df):
    """Same feature logic for CSV rows and SAP rows."""
    out = pd.DataFrame(index=df.index)
    out["Supplier"] = df["Supplier"].map(norm_key)
    out["Material"] = df["Material"].map(norm_key)
    for col in ("PurchasingOrganization", "PurchasingGroup", "Plant", "OrderQuantityUnit", "MaterialGroup"):
        out[col] = df[col].fillna("").astype(str).str.strip().str.upper()
    planned = pd.to_datetime(df["ScheduleLineDeliveryDate"])
    out["OrderQuantity"] = pd.to_numeric(df["OrderQuantity"], errors="coerce").round(3)
    # PurchaseOrderDate = real order date. CreationDate is when the PO was entered in S/4
    # (bulk-loaded POs get the load date, which made lead times negative).
    out["LeadTimeDays"] = (planned - pd.to_datetime(df["PurchaseOrderDate"])).dt.days
    out["PlannedMonth"] = planned.dt.month
    out["PlannedDayOfWeek"] = planned.dt.day_of_week
    return out


def load_history():
    text_cols = ["PurchaseOrder", "PurchaseOrderItem", "ScheduleLine", "CompanyCode", "Supplier", "Material",
                 "PurchasingOrganization", "PurchasingGroup", "Plant", "PurchaseOrderType", "OrderQuantityUnit",
                 "MaterialGroup"]
    df = pd.read_csv(CSV_PATH, dtype={c: str for c in text_cols},
                     parse_dates=["PurchaseOrderDate", "CreationDate", "ScheduleLineDeliveryDate",
                                  "ActualGoodsReceiptDate"])
    mismatch = int(((df["ActualGoodsReceiptDate"] - df["ScheduleLineDeliveryDate"]).dt.days != df[TARGET]).sum())
    log.info("History rows: %s | missing values: %s | DelayDays/date mismatches: %s",
             len(df), int(df.isna().sum().sum()), mismatch)
    df = df.dropna(subset=["Supplier", "ScheduleLineDeliveryDate", "PurchaseOrderDate", TARGET])
    outliers = df[TARGET].abs() > MAX_ABS_DELAY
    if outliers.any():
        log.info("Skipping %s history rows with |DelayDays| > %s", int(outliers.sum()), MAX_ABS_DELAY)
        df = df[~outliers]
    history = build_features(df)
    history[INDEX] = "H" + df["PurchaseOrder"] + "-" + df["PurchaseOrderItem"] + "-" + df["ScheduleLine"]
    history["PurchaseOrderDate"] = df["PurchaseOrderDate"]
    history["PurchaseOrderType"] = df["PurchaseOrderType"]
    history[TARGET] = df[TARGET]
    return history.sort_values("PurchaseOrderDate").reset_index(drop=True)
