# src/utils/receipts_repo.py
from __future__ import annotations

from typing import Any, Dict, Optional, Iterable, List
from datetime import datetime, date
from uuid import UUID
from decimal import Decimal
import dataclasses, json, re
from collections import deque

try:
    from dateutil import parser as dtparser
except Exception:
    dtparser = None

try:
    from langchain.schema import BaseMessage  # type: ignore
except Exception:
    BaseMessage = tuple()  # type: ignore


# ---------- JSON safety ----------
def to_json_safe(obj: Any):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseMessage):
        return {
            "type": getattr(obj, "type", obj.__class__.__name__),
            "content": getattr(obj, "content", None),
            "name": getattr(obj, "name", None),
            "additional_kwargs": to_json_safe(getattr(obj, "additional_kwargs", None)),
        }
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(x) for x in obj]
    if dataclasses.is_dataclass(obj):
        return to_json_safe(dataclasses.asdict(obj))
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return to_json_safe(obj.dict())
        except Exception:
            pass
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ---------- helpers ----------
def _coalesce(*vals):
    for v in vals:
        if v is not None and str(v).strip().lower() not in ("", "null", "none"):
            return v
    return None

def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None

def _parse_datetime(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    if dtparser:
        try:
            return dtparser.parse(str(val)).isoformat()
        except Exception:
            pass
    m = re.match(r"^\s*(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})", str(val))
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T00:00:00"
    return None

def _dict_ci_get(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    if not isinstance(d, dict):
        return None
    lower_map = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        lk = k.lower()
        if lk in lower_map:
            return d[lower_map[lk]]
    return None

def _deep_find(obj: Any, keys: Iterable[str]) -> Optional[Any]:
    if obj is None:
        return None
    keyset = {k.lower() for k in keys}
    q = deque([obj])
    while q:
        cur = q.popleft()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if str(k).lower() in keyset:
                    return v
            for v in cur.values():
                if isinstance(v, (dict, list, tuple)):
                    q.append(v)
        elif isinstance(cur, (list, tuple)):
            for v in cur:
                if isinstance(v, (dict, list, tuple)):
                    q.append(v)
    return None


# ---------- normalization ----------
def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # locate root (receipt_data / detected_fields), even if nested
    root = None
    if isinstance(payload, dict):
        root = _dict_ci_get(payload, ["receipt_data"]) or _dict_ci_get(payload, ["detected_fields"])
        if root is None:
            wrapper = _dict_ci_get(payload, ["result", "output", "data", "response"])
            if isinstance(wrapper, dict):
                root = (
                    _dict_ci_get(wrapper, ["receipt_data"]) or
                    _dict_ci_get(wrapper, ["detected_fields"]) or
                    wrapper
                )
    if root is None:
        root = _deep_find(payload, ["receipt_data", "detected_fields"]) or payload
    if not isinstance(root, dict):
        root = {}

    items = _dict_ci_get(root, ["items"]) or _deep_find(root, ["items"])
    # header fields
    store_name = _coalesce(_dict_ci_get(root, ["store_name", "merchant", "store", "retailer", "vendor", "shop", "chain"]))
    store_address = _coalesce(_dict_ci_get(root, ["address", "store_address", "location"]))
    # if missing, fall back to first item's store/address (your items carry these)
    if not store_name and isinstance(items, list) and items:
        store_name = _coalesce(items[0].get("store"))
    if not store_address and isinstance(items, list) and items:
        store_address = _coalesce(items[0].get("address"))

    purchase_dt = _parse_datetime(_coalesce(_dict_ci_get(root, ["date", "purchase_date", "datetime", "timestamp", "purchase_datetime"])))
    subtotal    = _to_float(_coalesce(_dict_ci_get(root, ["subtotal", "sub_total", "sub-total", "sub total"])))
    tax         = _to_float(_coalesce(_dict_ci_get(root, ["tax", "sales_tax", "vat", "gst"])))
    total       = _to_float(_coalesce(_dict_ci_get(root, ["total", "grand_total", "amount_due", "amount"])))
    currency    = _coalesce(_dict_ci_get(root, ["currency"]), "USD")
    raw_text    = _coalesce(_dict_ci_get(payload if isinstance(payload, dict) else {}, ["raw_text"]), _dict_ci_get(root, ["raw_text"]))

    return {
        "store_name": store_name,
        "store_address": store_address,
        "purchase_datetime": purchase_dt,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "currency": currency,
        "raw_text": raw_text,
        "items": items,
        "raw_json": payload,
    }


# ---------- public repo ops ----------
def upsert_receipts_dtl(supabase, payload: Dict[str, Any], *, receipt_file_id: Optional[str], visitor_id: Optional[str]):
    safe_payload = to_json_safe(payload)
    fields = _normalize_payload(safe_payload)

    row = {
        "visitor_id":        visitor_id,
        "receipt_file_id":   receipt_file_id,
        "store_name":        fields["store_name"],
        "store_address":     fields["store_address"],  # will retry as "address" if needed
        "purchase_datetime": fields["purchase_datetime"],
        "subtotal":          fields["subtotal"],
        "tax":               fields["tax"],
        "total":             fields["total"],
        "currency":          fields["currency"],
        "raw_text":          fields["raw_text"],
        "items":             to_json_safe(fields["items"]),
        "raw_json":          safe_payload,
    }

    # first attempt (schema with store_address)
    try:
        return supabase.table("receipts_dtl").upsert(row, on_conflict="receipt_file_id").execute()
    except Exception as e:
        # If their table actually has "address" instead of "store_address", retry
        if 'store_address' in str(e) and 'column' in str(e).lower():
            row_fixed = dict(row)
            row_fixed["address"] = row_fixed.pop("store_address")
            return supabase.table("receipts_dtl").upsert(row_fixed, on_conflict="receipt_file_id").execute()
        raise


def upsert_receipt_items(supabase, *, receipt_id: str, items: Optional[List[Dict[str, Any]]]):
    """Insert/Upsert one row per item."""
    if not receipt_id or not items or not isinstance(items, list):
        return None

    rows = []
    for i, it in enumerate(items, start=1):
        name = _coalesce(it.get("name"), it.get("item"), it.get("description"))
        qty  = _to_float(_coalesce(it.get("quantity"), it.get("qty"), 1)) or 1.0
        price = _to_float(_coalesce(it.get("price"), it.get("unit_price"), it.get("cost"), it.get("amount")))
        line_total = _to_float(_coalesce(it.get("line_total"), (price * qty) if (price is not None) else None))
        rows.append({
            "receipt_id": receipt_id,
            "line_no": i,
            "name": name,
            "quantity": qty,
            "unit_price": price,
            "line_total": line_total,
            "raw": to_json_safe(it),
        })

    # relies on a unique constraint (receipt_id, line_no)
    return supabase.table("receipt_items").upsert(rows, on_conflict="receipt_id,line_no").execute()


def save_receipt_and_items(supabase, payload: Dict[str, Any], *, receipt_file_id: Optional[str], visitor_id: Optional[str]):
    """Convenience: upsert header then items. Returns (header_res, items_res)."""
    hdr = upsert_receipts_dtl(supabase, payload, receipt_file_id=receipt_file_id, visitor_id=visitor_id)

    # try to get new receipt id
    receipt_id = None
    if getattr(hdr, "data", None):
        receipt_id = hdr.data[0].get("id")

    if not receipt_id and receipt_file_id:
        try:
            sel = supabase.table("receipts_dtl").select("id").eq("receipt_file_id", receipt_file_id).limit(1).execute()
            if getattr(sel, "data", None):
                receipt_id = sel.data[0]["id"]
        except Exception:
            pass

    items_res = None
    if receipt_id:
        # pull items from normalized view of payload
        items = _normalize_payload(to_json_safe(payload))["items"]
        items_res = upsert_receipt_items(supabase, receipt_id=receipt_id, items=items)

    return hdr, items_res


def ensure_receipt_file_id(supabase, *, receipt_file_id: Optional[str], filename: Optional[str]) -> Optional[str]:
    if receipt_file_id:
        return receipt_file_id
    if not filename:
        return None
    try:
        res = supabase.table("receipt_files").select("id").eq("filename", filename).limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]["id"]
    except Exception:
        pass
    return None
