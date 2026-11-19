"""Small shared helpers."""

import re
from datetime import date, datetime, timezone


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
