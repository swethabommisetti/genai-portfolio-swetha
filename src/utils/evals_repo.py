from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime
from utils.supabase_utils import get_supabase_client


# -----------------------------
# Writes
# -----------------------------
def save_manual_scores(
    *,
    receipt_file_id: str,
    store_correct: bool | None,
    total_correct: bool | None,
    date_correct: bool | None,
    overall_score: int,
    comment: str | None,
    evaluator_email: str | None,
) -> Dict[str, Any]:
    sb = get_supabase_client()
    row = {
        "receipt_file_id": receipt_file_id,
        "store_correct": store_correct,
        "total_correct": total_correct,
        "date_correct": date_correct,
        "overall_score": overall_score,
        "comment": comment,
        "evaluator_email": evaluator_email,
        "created_at": datetime.utcnow().isoformat(),
    }
    res = sb.table("evaluations").insert(row).execute()
    return {"ok": True, "inserted": getattr(res, "data", [])}


def save_error_tags(
    *,
    receipt_file_id: str,
    tags: List[str],
    comment: str | None,
    evaluator_email: str | None,
) -> Dict[str, Any]:
    sb = get_supabase_client()
    rows = [
        {
            "receipt_file_id": receipt_file_id,
            "tag": t,
            "comment": comment,
            "evaluator_email": evaluator_email,
            "created_at": datetime.utcnow().isoformat(),
        }
        for t in tags
    ]
    if not rows:
        return {"ok": False, "error": "no-tags"}
    res = sb.table("evaluation_errors").insert(rows).execute()
    return {"ok": True, "inserted": getattr(res, "data", [])}


# -----------------------------
# Reads / dashboards
# -----------------------------
def list_recent_receipts(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent receipts from receipts_dtl without assuming specific columns exist."""
    sb = get_supabase_client()
    # Try to order by purchase_datetime if it exists; otherwise fall back to id
    try:
        q = (
            sb.table("receipts_dtl")
            .select("*")
            .order("purchase_datetime", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception:
        q = (
            sb.table("receipts_dtl")
            .select("*")
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
    return getattr(q, "data", []) or []


def metrics_summary() -> Dict[str, Any]:
    sb = get_supabase_client()
    out: Dict[str, Any] = {"counts": {}, "accuracy": {}}
    try:
        out["counts"]["receipts"] = (
            sb.table("receipts_dtl").select("id", count="exact").execute().count
        ) or 0
        out["counts"]["evaluations"] = (
            sb.table("evaluations").select("id", count="exact").execute().count
        ) or 0
        out["counts"]["errors"] = (
            sb.table("evaluation_errors").select("id", count="exact").execute().count
        ) or 0
    except Exception:
        pass
    try:
        ev = sb.table("evaluations").select("store_correct, total_correct, date_correct").execute()
        rows = getattr(ev, "data", []) or []

        def pct(field: str) -> float:
            vals = [r.get(field) for r in rows if r.get(field) is not None]
            return round(100.0 * sum(1 for v in vals if v) / max(1, len(vals)), 1) if vals else 0.0

        out["accuracy"] = {
            "store_name_accuracy_%": pct("store_correct"),
            "total_accuracy_%": pct("total_correct"),
            "date_accuracy_%": pct("date_correct"),
        }
    except Exception:
        pass
    return out


# -----------------------------
# Assets
# -----------------------------
def receipt_image_url(receipt_file_id: str, *, expires: int = 600) -> Optional[str]:
    """Return a short-lived signed URL for the receipt image."""
    sb = get_supabase_client()
    try:
        resp = (
            sb.table("receipt_files")
            .select("filename,bucket_name")
            .eq("id", receipt_file_id)
            .limit(1)
            .execute()
        )
        row = (getattr(resp, "data", None) or [{}])[0]
        filename = row.get("filename")
        bucket = row.get("bucket_name")
        if not filename or not bucket:
            return None

        signed = sb.storage.from_(bucket).create_signed_url(filename, expires)
        return (
            signed.get("signedURL")
            or signed.get("signed_url")
            or signed.get("url")
            or None
        )
    except Exception:
        return None


# -----------------------------
# Header + Items (robust parser)
# -----------------------------
def get_extracted_values(*, receipt_file_id: str | None = None, dtl_id: str | None = None) -> dict:
    """
    Load header + items for a selected receipt.

    Priority:
      1) Read header from receipts_dtl (by id or receipt_file_id) and, if present,
         use its JSONB 'items' column (string or array) -> normalize.
      2) Fallback: read row-per-item from receipt_items by receipt_file_id -> normalize.

    Returns: {"header": {...}, "items": [...], "totals": {"grand_total": float | None}}
    """
    import json
    sb = get_supabase_client()
    out: Dict[str, Any] = {"header": {}, "items": [], "totals": {}}

    # ---- HEADER ----
    header = None
    try:
        if dtl_id:
            r = sb.table("receipts_dtl").select("*").eq("id", dtl_id).limit(1).execute()
            header = (getattr(r, "data", []) or [None])[0]
        if header is None and receipt_file_id:
            r = (
                sb.table("receipts_dtl")
                .select("*")
                .eq("receipt_file_id", receipt_file_id)
                .order("id", desc=True)
                .limit(1)
                .execute()
            )
            header = (getattr(r, "data", []) or [None])[0]
    except Exception:
        header = None

    if not header:
        return out

    out["header"] = header
    rid = header.get("receipt_file_id") or receipt_file_id

    # ---- Try items JSONB on receipts_dtl ----
    raw_items = header.get("items")
    if raw_items:
        if isinstance(raw_items, str):
            try:
                raw_items = json.loads(raw_items)
            except Exception:
                raw_items = []
    items = []
    if isinstance(raw_items, list):
        items = raw_items

    # ---- Fallback: receipt_items by receipt_file_id ----
    if not items and rid:
        try:
            q = (
                sb.table("receipt_items")
                .select("*")
                .eq("receipt_file_id", rid)
                .order("line_no", desc=False)
                .execute()
            )
            items = getattr(q, "data", []) or []
        except Exception:
            items = []

    # ---- Normalize for display ----
    norm: List[Dict[str, Any]] = []
    grand_total = 0.0

    def fnum(v):
        try:
            return float(str(v).replace("$", "")) if v is not None else None
        except Exception:
            return None

    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("item_name") or "—"
        qty  = it.get("qty", it.get("quantity"))
        unit = it.get("unit_price", it.get("price"))
        line = it.get("line_total", it.get("cost", it.get("total")))
        store = it.get("store")
        addr  = it.get("address")
        curr  = it.get("currency")

        qty_f  = fnum(qty)
        unit_f = fnum(unit)
        line_f = fnum(line)
        if line_f is None and qty_f is not None and unit_f is not None:
            line_f = qty_f * unit_f
        if line_f is not None:
            grand_total += line_f

        norm.append({
            "line_no": it.get("line_no") or it.get("line") or it.get("seq"),
            "name": name,
            "qty": qty_f if qty_f is not None else qty,
            "unit_price": unit_f if unit_f is not None else unit,
            "line_total": line_f if line_f is not None else line,
            "store": store,
            "address": addr,
            "currency": curr,
        })

    out["items"] = norm
    out["totals"] = {"grand_total": round(grand_total, 2) if norm else None}
    return out
