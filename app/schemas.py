"""Request models for the API."""

from datetime import date

from pydantic import BaseModel, model_validator


class PredictRequest(BaseModel):
    purchase_orders: list[str] | None = None   # 10-digit SAP PO numbers, exact format, e.g. "4500001234"
    created_from: date | None = None           # ISO yyyy-mm-dd
    created_to: date | None = None             # ISO yyyy-mm-dd
    level: str = "schedule_line"               # schedule_line | item
    max_pos: int = 100                         # newest N POs (SAP-RPT-1 handles 128 rows per call)

    @model_validator(mode="after")
    def check_level(self):
        if self.level not in ("schedule_line", "item"):
            raise ValueError("level must be 'schedule_line' or 'item'")
        return self


class POLine(BaseModel):
    """One PO item / schedule line given directly in the request (no S/4 lookup)."""
    PurchaseOrder: str = "TEST"
    PurchaseOrderItem: str = "10"
    ScheduleLine: str = "1"
    Supplier: str
    Material: str = ""
    MaterialGroup: str = ""
    PurchasingOrganization: str = ""
    PurchasingGroup: str = ""
    PurchaseOrderType: str = "NB"
    Plant: str = ""
    OrderQuantity: float
    OrderQuantityUnit: str = ""
    CreationDate: date
    PurchaseOrderDate: date | None = None   # order date; defaults to CreationDate
    ScheduleLineDeliveryDate: date          # planned delivery date


class PredictDataRequest(BaseModel):
    items: list[POLine]
