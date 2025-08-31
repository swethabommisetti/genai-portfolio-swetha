# src/pages/18_Evaluator — Prompt Accuracy.py
import math
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
from utils.supabase_utils import get_supabase_client
import utils.evals_repo as evals  # reuse your helpers

st.set_page_config(page_title="Evaluator — Prompt Accuracy (Headers)", layout="wide")
st.title("🎯 Evaluator — Prompt Accuracy (Headers)")
st.markdown(      """
Evaluates Header between Model and Gold dataset Created
        """)

TOLERANCE = 0.05  # $ cents tolerance for totals

supabase = get_supabase_client()

@st.cache_data(ttl=60)
def load_current_gold(limit:int=300) -> List[Dict[str, Any]]:
    """
    Pull the SCD-2 current gold rows and join to receipts_dtl for file linkage.
    Returns list of dicts with: receipt_id, gold fields, and receipt_file_id.
    """
    gold_rows = (
        supabase.table("receipts_gold")
        .select("receipt_id, store_name_gold, total_gold, purchase_date_gold")
        .eq("curr_rec_ind", True)
        .order("rec_eff_start", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )
    if not gold_rows:
        return []

    ids = [r["receipt_id"] for r in gold_rows if r.get("receipt_id")]
    if not ids:
        return []

    dtl = (
        supabase.table("receipts_dtl")
        .select("id, receipt_file_id")
        .in_("id", ids)
        .execute()
        .data or []
    )
    idx = {d["id"]: d for d in dtl}
    out = []
    for g in gold_rows:
        rid = g["receipt_id"]
        rf = idx.get(rid, {})
        out.append({
            "receipt_id": rid,
            "receipt_file_id": rf.get("receipt_file_id"),
            "store_name_gold": g.get("store_name_gold"),
            "total_gold": g.get("total_gold"),
            "purchase_date_gold": g.get("purchase_date_gold"),
        })
    return out

def normalize_date_str(x) -> Optional[str]:
    if not x:
        return None
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace("Z","")).date().isoformat()
        except Exception:
            return x[:10] if len(x) >= 10 else x
    return None

def to_float_safe(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def evaluate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a row with gold + receipt ids, fetch model output and compute pass/fail per field.
    """
    rid = row.get("receipt_id")
    rf_id = row.get("receipt_file_id")
    if not (rid and rf_id):
        return {**row, "eval_status": "missing_link"}

    data = evals.get_extracted_values(receipt_file_id=rf_id, dtl_id=rid) or {}
    hdr = data.get("header") or {}
    totals = data.get("totals") or {}

    # predictions
    pred_store = (hdr.get("store_name") or "") or None
    pred_total = to_float_safe(hdr.get("total"))
    if pred_total is None:
        pred_total = to_float_safe(totals.get("grand_total"))
    pred_date_raw = hdr.get("purchase_datetime")
    pred_date = normalize_date_str(pred_date_raw)

    # golds
    gold_store = (row.get("store_name_gold") or "") or None
    gold_total = to_float_safe(row.get("total_gold"))
    gold_date = normalize_date_str(row.get("purchase_date_gold"))

    # comparisons
    store_pass = None
    if gold_store or pred_store:
        store_pass = ( (gold_store or "").strip().casefold() ==
                       (pred_store or "").strip().casefold() )

    total_pass = None
    delta = None
    if gold_total is not None and pred_total is not None:
        delta = abs(pred_total - gold_total)
        total_pass = (delta <= TOLERANCE)

    date_pass = None
    if gold_date or pred_date:
        date_pass = (gold_date == pred_date)

    overall = all([
        x is True for x in (store_pass, total_pass, date_pass)
    ]) if any(v is not None for v in (store_pass, total_pass, date_pass)) else None

    return {
        "receipt_id": rid,
        "receipt_file_id": rf_id,
        "gold_store": gold_store,
        "pred_store": pred_store,
        "store_pass": store_pass,
        "gold_total": gold_total,
        "pred_total": pred_total,
        "total_delta": delta,
        "total_pass": total_pass,
        "gold_date": gold_date,
        "pred_date": pred_date,
        "date_pass": date_pass,
        "overall_pass": overall,
    }

# ---------- UI ----------
gold_rows = load_current_gold(limit=300)
if not gold_rows:
    st.info("No current gold rows found. Use the Manual Scoring page to create SCD-2 gold first.")
    st.stop()

with st.status("Evaluating against current gold…", expanded=False):
    results = [evaluate_row(r) for r in gold_rows]

df = pd.DataFrame(results)

# KPIs
def rate(col):
    s = df[col].dropna()
    return (s.mean() * 100.0) if len(s) else None

store_acc = rate("store_pass")
total_acc = rate("total_pass")
date_acc  = rate("date_pass")
overall   = rate("overall_pass")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Store match", f"{store_acc:.1f}%" if store_acc is not None else "—")
k2.metric("Total within ±$0.05", f"{total_acc:.1f}%" if total_acc is not None else "—")
k3.metric("Date match", f"{date_acc:.1f}%" if date_acc is not None else "—")
k4.metric("Overall (all 3)", f"{overall:.1f}%" if overall is not None else "—")

st.divider()

# ---- Failures table (compact) ----
fails = df[(df["overall_pass"] == False) | df["overall_pass"].isna()].copy()
st.subheader("❌ Mismatches / Needs review")

def _status(b):
    return "PASS" if b is True else ("FAIL" if b is False else "—")

if fails.empty:
    st.success("All evaluated receipts passed (for fields present).")
else:
    # Map booleans -> PASS/FAIL/—
    fails_show = fails.copy()
    for col in ["store_pass", "total_pass", "date_pass", "overall_pass"]:
        fails_show[col] = fails_show[col].map(_status)

    view_cols = [
        "receipt_id","receipt_file_id",
        "gold_store","pred_store","store_pass",
        "gold_total","pred_total","total_delta","total_pass",
        "gold_date","pred_date","date_pass",
        "overall_pass"
    ]

    st.dataframe(
        fails_show[view_cols],
        use_container_width=True,
        hide_index=True
    )

    # ——— Row-by-row expanders with receipt preview + side-by-side details
    st.markdown("### Inspect each mismatch")
    for _, row in fails.iterrows():  # use original 'fails' booleans for expander logic
        rid = row["receipt_id"]; rfid = row["receipt_file_id"]
        label = f"{rid}  •  file:{rfid}  •  store:{row.get('pred_store') or '—'}"
        with st.expander(label, expanded=False):
            left, right = st.columns([5, 7], gap="large")
            with left:
                url = evals.receipt_image_url(rfid)
                if url:
                    st.image(url, use_container_width=True)
                    st.caption(f"receipt_file_id: `{rfid}`")
                else:
                    st.info("Could not generate a signed URL for this image.")

            with right:
                st.markdown("**Header comparison**")
                def badge(ok, txt):
                    if ok is True:   return f"✅ {txt} — PASS"
                    if ok is False:  return f"🟡 {txt} — FAIL"
                    return f"— {txt}"

                store_ok = row.get("store_pass")
                total_ok = row.get("total_pass")
                date_ok  = row.get("date_pass")
                delta    = row.get("total_delta")

                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Gold")
                    st.write(f"Store: {row.get('gold_store') or '—'}")
                    st.write(f"Total: {row.get('gold_total') if row.get('gold_total') is not None else '—'}")
                    st.write(f"Date:  {row.get('gold_date') or '—'}")
                with c2:
                    st.caption("Prediction")
                    st.write(badge(store_ok, f"Store: {row.get('pred_store') or '—'}"))
                    pred_total = row.get("pred_total")
                    total_txt = (f"{pred_total}  (Δ={delta:.2f})"
                                 if (pred_total is not None and delta is not None)
                                 else f"{pred_total if pred_total is not None else '—'}")
                    st.write(badge(total_ok, f"Total: {total_txt}"))
                    st.write(badge(date_ok,  f"Date:  {row.get('pred_date') or '—'}"))

                st.caption("Legend: ✅ PASS • 🟡 FAIL • — not available")
