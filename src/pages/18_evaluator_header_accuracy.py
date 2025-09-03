# src/pages/18_evaluator_header_accuracy.py
# Evaluator — Prompt Accuracy (Headers) for your schema:
#   Gold: receipts_gold (current rows: curr_rec_ind = true)
#   Pred: receipts_dtl (normalized extraction you store after upload)
#   Files: receipt_files (to render image via signed URL)

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from utils.supabase_utils import get_supabase_client

st.set_page_config(page_title="Evaluator — Prompt Accuracy (Headers)", layout="wide")
st.title("🎯 Evaluator — Prompt Accuracy (Headers)")
st.caption("Evaluates header fields between **Model (receipts_dtl)** and **Gold (receipts_gold)**.")

# ---------------- Tunables ----------------
TOTAL_TOLERANCE = 5.00  # pass if |pred - gold| <= $0.05
DROPDOWN_LIMIT = 600    # how many latest gold rows to offer in the dropdown
SIGNED_URL_TTL = 900    # seconds (15 min)
# ------------------------------------------

supabase = get_supabase_client()

# ---------------- Supabase helpers ----------------
@st.cache_data(ttl=90)
def _fetch_current_gold(limit: int = DROPDOWN_LIMIT) -> pd.DataFrame:
    """
    Pull current rows from receipts_gold and join to receipts_dtl to get receipt_file_id
    and the predicted header fields we want to compare.
    """
    gold = (
        supabase.table("receipts_gold")
        .select("id,receipt_id,store_name_gold,total_gold,purchase_date_gold,rec_eff_start")
        .eq("curr_rec_ind", True)
        .order("rec_eff_start", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )
    if not gold:
        return pd.DataFrame()

    gdf = pd.DataFrame(gold)

    # Join to receipts_dtl for predictions + file linkage
    receipt_ids = [r for r in gdf["receipt_id"].tolist() if r]
    if not receipt_ids:
        return gdf.assign(
            receipt_file_id=None, store_name=None, total=None, purchase_datetime=None
        )

    dtl = (
        supabase.table("receipts_dtl")
        .select("id,receipt_file_id,store_name,total,purchase_datetime")
        .in_("id", receipt_ids)
        .execute()
        .data or []
    )
    ddf = pd.DataFrame(dtl) if dtl else pd.DataFrame(columns=["id","receipt_file_id","store_name","total","purchase_datetime"])
    ddf.rename(columns={"id": "receipt_id"}, inplace=True)

    merged = gdf.merge(ddf, on="receipt_id", how="left")
    return merged

@st.cache_data(ttl=300)
def _file_row(receipt_file_id: str) -> Optional[Dict]:
    try:
        res = (
            supabase.table("receipt_files")
            .select("filename,bucket_name")
            .eq("id", receipt_file_id)
            .limit(1)
            .execute()
        )
        if getattr(res, "data", None):
            return res.data[0]
    except Exception:
        pass
    return None

def _signed_url(receipt_file_id: Optional[str]) -> Optional[str]:
    if not receipt_file_id:
        return None
    row = _file_row(receipt_file_id)
    if not row:
        return None
    path = row.get("filename")
    bucket = row.get("bucket_name")
    if not (path and bucket):
        return None
    try:
        signed = supabase.storage.from_(bucket).create_signed_url(path=path, expires_in=SIGNED_URL_TTL)
        if isinstance(signed, dict):
            return signed.get("signedURL") or signed.get("signed_url") or signed.get("url")
        return (
            getattr(signed, "signedURL", None)
            or getattr(signed, "signed_url", None)
            or getattr(signed, "url", None)
        )
    except Exception:
        return None

# ---------------- Eval logic ----------------
def _norm_date_only(x) -> Optional[str]:
    """Return YYYY-MM-DD for comparisons."""
    if not x:
        return None
    if isinstance(x, date):
        return x.isoformat()
    s = str(x).strip()
    try:
        return datetime.fromisoformat(s.replace("Z", "")).date().isoformat()
    except Exception:
        return s[:10] if len(s) >= 10 else s

@dataclass
class EvalRow:
    receipt_id: str
    receipt_file_id: Optional[str]
    gold_store: Optional[str]
    pred_store: Optional[str]
    store_pass: Optional[bool]
    gold_total: Optional[float]
    pred_total: Optional[float]
    total_delta: Optional[float]
    total_pass: Optional[bool]
    gold_date: Optional[str]
    pred_date: Optional[str]
    date_pass: Optional[bool]
    overall_pass: Optional[bool]

def _to_float(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None

def evaluate_rows(df: pd.DataFrame, only_receipt_id: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if only_receipt_id is not None:
        df = df[df["receipt_id"] == only_receipt_id]

    rows: List[EvalRow] = []
    for _, r in df.iterrows():
        rid = r.get("receipt_id")
        rfid = r.get("receipt_file_id")

        # golds
        gold_store = r.get("store_name_gold")
        gold_total = _to_float(r.get("total_gold"))
        gold_date = _norm_date_only(r.get("purchase_date_gold"))

        # preds from receipts_dtl
        pred_store = r.get("store_name")
        pred_total = _to_float(r.get("total"))
        pred_date = _norm_date_only(r.get("purchase_datetime"))

        # store pass
        store_pass = None
        if (gold_store or pred_store):
            store_pass = ( (str(gold_store or "").strip().casefold()) ==
                           (str(pred_store or "").strip().casefold()) )

        # total pass / delta
        total_pass = None
        total_delta = None
        if gold_total is not None and pred_total is not None:
            total_delta = round(abs(pred_total - gold_total), 2)
            total_pass = (total_delta <= TOTAL_TOLERANCE)

        # date pass (date-only compare)
        date_pass = None
        if (gold_date or pred_date):
            date_pass = (gold_date == pred_date)

        overall_pass = None
        touched = [store_pass, total_pass, date_pass]
        if any(v is not None for v in touched):
            overall_pass = all(v is True for v in touched)

        rows.append(
            EvalRow(
                receipt_id=str(rid),
                receipt_file_id=rfid,
                gold_store=gold_store,
                pred_store=pred_store,
                store_pass=store_pass,
                gold_total=gold_total,
                pred_total=pred_total,
                total_delta=total_delta,
                total_pass=total_pass,
                gold_date=gold_date,
                pred_date=pred_date,
                date_pass=date_pass,
                overall_pass=overall_pass,
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])

def _pct(col: str, df: pd.DataFrame) -> Optional[float]:
    s = df[col].dropna()
    return round(float(s.mean() * 100.0), 1) if len(s) else None

def _nan_to_none(v):
    return None if pd.isna(v) else v

def _write_results_to_db(results_df: pd.DataFrame, scope: str):
    if results_df.empty:
        return 0

    def _nan_to_none(v):
        return None if pd.isna(v) else v

    payload = []
    for _, r in results_df.iterrows():
        payload.append({
            "source_kind":      "personal",                   # ← NEW: required by your table
            "receipt_id":       _nan_to_none(r.get("receipt_id")),
            "receipt_file_id":  _nan_to_none(r.get("receipt_file_id")),
            "gold_store":       _nan_to_none(r.get("gold_store")),
            "pred_store":       _nan_to_none(r.get("pred_store")),
            "store_pass":       _nan_to_none(r.get("store_pass")),
            "gold_total":       _nan_to_none(r.get("gold_total")),
            "pred_total":       _nan_to_none(r.get("pred_total")),
            "total_delta":      _nan_to_none(r.get("total_delta")),
            "total_pass":       _nan_to_none(r.get("total_pass")),
            "gold_date":        _nan_to_none(r.get("gold_date")),
            "pred_date":        _nan_to_none(r.get("pred_date")),
            "date_pass":        _nan_to_none(r.get("date_pass")),
            "overall_pass":     _nan_to_none(r.get("overall_pass")),
            "eval_scope":       scope,
            "total_tolerance":  TOTAL_TOLERANCE,
        })

    supabase.table("receipts_header_eval_results").insert(payload).execute()
    return len(payload)

# ---------------- UI: top controls ----------------
gold_df = _fetch_current_gold(DROPDOWN_LIMIT)

with st.container(border=True):
    st.write("✅ Evaluating against **current** gold (receipts_gold.curr_rec_ind = true).")

if gold_df.empty:
    st.info("No current gold rows found. Create gold labels first on your ‘Gold Dataset’ page.")
    st.stop()

def _label_for_row(r) -> str:
    note = r.get("store_name_gold") or "—"
    d = r.get("purchase_date_gold")
    d_txt = _norm_date_only(d) or "—"
    return f"{str(r.get('receipt_id'))[:8]}…  •  {note}  •  {d_txt}"

options = [(None, "All receipts")] + [
    (str(r["receipt_id"]), _label_for_row(r)) for _, r in gold_df.iterrows()
]

idx_map = {i: rid for i, (rid, _) in enumerate(options)}
label_map = {i: lbl for i, (_, lbl) in enumerate(options)}

selection = st.selectbox(
    "Pick a receipt",
    options=list(idx_map.keys()),
    format_func=lambda i: label_map[i],
    index=0,
)
selected_receipt_id = idx_map[selection]

run = st.button("Run", type="primary")

# remember last selection (allow None => evaluate ALL)
if "hdr_eval_last" not in st.session_state:
    st.session_state.hdr_eval_last = None
if run:
    st.session_state.hdr_eval_last = selected_receipt_id

if "hdr_eval_last" not in st.session_state:
    st.stop()

active_receipt_id = st.session_state.hdr_eval_last
if "hdr_eval_ran" not in st.session_state:
    st.session_state.hdr_eval_ran = False

if active_receipt_id is None:
    scope_txt = "all"
else:
    scope_txt = "single"

if not run and not st.session_state.hdr_eval_ran:
    st.info("Choose a receipt (or leave **All receipts**) and click **Run**.")
    st.stop()

# ---------------- Execute evaluation ----------------
with st.spinner("Running header accuracy evaluation…"):
    results_df = evaluate_rows(gold_df, only_receipt_id=active_receipt_id)

st.session_state.hdr_eval_ran = True

if results_df.empty:
    st.info("No rows to evaluate for this selection.")
    st.stop()

# Write results to DB
try:
    inserted = _write_results_to_db(results_df, scope=scope_txt)
    st.success(f"Saved {inserted} row(s) to `public.receipts_header_eval_results`.")
except Exception as e:
    st.error(f"DB insert failed: {e}")

# KPIs
k_store = _pct("store_pass", results_df)
k_total = _pct("total_pass", results_df)
k_date  = _pct("date_pass",  results_df)
k_over  = _pct("overall_pass", results_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Store match", f"{k_store:.1f}%" if k_store is not None else "—")
c2.metric(f"Total within ±${TOTAL_TOLERANCE:.2f}", f"{k_total:.1f}%" if k_total is not None else "—")
c3.metric("Date match", f"{k_date:.1f}%" if k_date is not None else "—")
c4.metric("Overall (all 3)", f"{k_over:.1f}%" if k_over is not None else "—")

st.divider()

# ---------------- Mismatches table ----------------
st.subheader("❌ Mismatches / Needs review")
fails = results_df[(results_df["overall_pass"] == False) | results_df["overall_pass"].isna()].copy()

def _status(v: Optional[bool]) -> str:
    return "PASS" if v is True else ("FAIL" if v is False else "—")

if fails.empty:
    st.success("All evaluated rows passed (for the fields present).")
else:
    show = fails.copy()
    for col in ["store_pass", "total_pass", "date_pass", "overall_pass"]:
        show[col] = show[col].map(_status)

    cols = [
        "receipt_id","receipt_file_id",
        "gold_store","pred_store","store_pass",
        "gold_total","pred_total","total_delta","total_pass",
        "gold_date","pred_date","date_pass",
        "overall_pass",
    ]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)

    st.markdown("### Inspect each mismatch")
    for _, row in fails.iterrows():
        rid = row["receipt_id"]; rfid = row.get("receipt_file_id")
        label = f"{str(rid)[:8]}…  •  file:{str(rfid)[:8]+'…' if rfid else '—'}  •  store:{row.get('pred_store') or '—'}"
        with st.expander(label, expanded=False):
            left, right = st.columns([5, 7], gap="large")
            with left:
                url = _signed_url(rfid)
                if url:
                    st.image(url, use_container_width=True)
                else:
                    st.info("No image available for this receipt.")
                st.caption(f"receipt_file_id: `{rfid or '—'}`")

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

                cA, cB = st.columns(2)
                with cA:
                    st.caption("Gold")
                    st.write(f"Store: {row.get('gold_store') or '—'}")
                    st.write(f"Total: {row.get('gold_total') if row.get('gold_total') is not None else '—'}")
                    st.write(f"Date:  {row.get('gold_date') or '—'}")
                with cB:
                    st.caption("Prediction (receipts_dtl)")
                    st.write(badge(store_ok, f"Store: {row.get('pred_store') or '—'}"))
                    pt = row.get("pred_total")
                    total_txt = (f"{pt}  (Δ={delta:.2f})"
                                 if (pt is not None and delta is not None)
                                 else f"{pt if pt is not None else '—'}")
                    st.write(badge(total_ok, f"Total: {total_txt}"))
                    st.write(badge(date_ok,  f"Date:  {row.get('pred_date') or '—'}"))

                st.caption("Legend: ✅ PASS • 🟡 FAIL • — not available")
