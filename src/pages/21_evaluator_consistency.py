# src/pages/21_evaluator_consistency.py
import os
import io
import math
import tempfile
from datetime import datetime, date
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st

from utils.supabase_utils import get_supabase_client
import utils.evals_repo as evals  # for receipt_image_url()

# Your LangGraph agent (same import pattern you used elsewhere)
try:
    from agents.receipt_extractor.agent import app as receipt_graph
except Exception:
    receipt_graph = None

# -------------------------- Page setup --------------------------
st.set_page_config(page_title="Evaluator — Consistency", layout="wide")
st.title("🎯 Evaluator — Consistency (single receipt)")

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "genai-analytics-bucket")
TOLERANCE = 0.05  # dollars: how close totals must be to count as same

def sb():
    return get_supabase_client()

# -------------------------- Helpers -----------------------------
def _download(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def _bytes_to_tmp_jpg(data: bytes) -> str:
    fd, path = tempfile.mkstemp(prefix="consistency_", suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

def _norm_date(x) -> Optional[str]:
    if not x:
        return None
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, str):
        # try ISO first; otherwise fallback to yyyy-mm-dd prefix
        try:
            return datetime.fromisoformat(x.replace("Z", "")).date().isoformat()
        except Exception:
            return x[:10] if len(x) >= 10 else x
    return None

def _to_float(v) -> Optional[float]:
    try:
        if v in (None, "", "None"):
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _extract_fields_from_result(res: Any) -> Dict[str, Any]:
    """
    Your agent returns a dict with receipt_data.header/totals (or header/totals).
    This normalizes to a small dict for downstream scoring.
    """
    header = {}
    totals = {}
    if isinstance(res, dict):
        header = res.get("receipt_data", {}).get("header") or res.get("header") or {}
        totals = res.get("receipt_data", {}).get("totals") or res.get("totals") or {}

    pred_store = (header.get("store_name") or None)
    pred_total = _to_float(header.get("total"))
    if pred_total is None:
        pred_total = _to_float(totals.get("grand_total"))
    pred_date = _norm_date(header.get("purchase_datetime"))

    return {
        "store_name": pred_store,
        "total": pred_total,
        "purchase_datetime": pred_date,
        "raw": res,
    }

def _invoke_agent_on_image_url(image_url: str) -> Dict[str, Any]:
    if receipt_graph is None:
        return {"error": "agent-not-imported"}
    b = _download(image_url)
    if not b:
        return {"error": "download-failed"}

    tmp = _bytes_to_tmp_jpg(b)
    try:
        res = receipt_graph.invoke({"image_path": tmp})
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

    return _extract_fields_from_result(res)

def _mode_or_none(values: List[Any]) -> Optional[Any]:
    """Mode (most common non-None); tie breaks by first seen."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    counts = {}
    order = []
    for v in vals:
        k = v if not isinstance(v, str) else v.strip().casefold()
        if k not in counts:
            counts[k] = 0
            order.append(k)
        counts[k] += 1
    best = max(order, key=lambda k: counts[k])
    # return original value “as seen” (case preserved where possible)
    for v in values:
        kk = v if not isinstance(v, str) else v.strip().casefold()
        if kk == best:
            return v
    return None

def _percent_equal(values: List[Any], pivot: Any, *, numeric_tolerance: Optional[float] = None) -> float:
    if pivot is None or not values:
        return 0.0
    same = 0
    total = 0
    for v in values:
        total += 1
        if isinstance(pivot, (int, float)) and isinstance(v, (int, float)) and numeric_tolerance is not None:
            if v is not None and abs(float(v) - float(pivot)) <= numeric_tolerance:
                same += 1
        elif isinstance(pivot, str) and isinstance(v, str):
            if pivot.strip().casefold() == v.strip().casefold():
                same += 1
        else:
            if v == pivot:
                same += 1
    return (same / total) * 100.0

# ----------------------- Data: personal receipts -------------------
@st.cache_data(ttl=60)
def list_personal_recent(limit: int = 100) -> List[Dict[str, Any]]:
    rows = (
        sb()
        .table("receipts_dtl")
        .select("id, receipt_file_id, created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )
    out = []
    for r in rows:
        rid = r.get("id")
        out.append(
            {
                "receipt_id": str(rid),
                "receipt_file_id": r.get("receipt_file_id"),
                "label": f"#{rid} • file:{r.get('receipt_file_id')}",
            }
        )
    return out

# -------------------------- UI: pick receipt -----------------------
rows = list_personal_recent(limit=100)
if not rows:
    st.info("No personal receipts found yet. Run the extractor first.")
    st.stop()

choice = st.selectbox("Pick a receipt", options=rows, format_func=lambda r: r["label"])
rid = choice["receipt_id"]
file_id = choice["receipt_file_id"]
signed_url = evals.receipt_image_url(file_id)

left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("Receipt image")
    if signed_url:
        st.image(signed_url, use_container_width=True)
    else:
        st.warning("Could not generate a signed URL for this image.")

with right:
    st.subheader("Experiment setup")
    runs = st.slider("Number of repeated runs", min_value=3, max_value=20, value=8, step=1)
    st.caption("We’ll run your agent multiple times on the exact same image and measure how often the predicted fields agree.")

    run_btn = st.button("Run consistency test", type="primary", use_container_width=True)

# -------------------------- Execute -------------------------------
if not run_btn:
    st.stop()

if not signed_url:
    st.error("No image URL available; cannot run.")
    st.stop()

st.info(f"Running {runs} repetitions on receipt #{rid}…")
prog = st.progress(0.0)

preds: List[Dict[str, Any]] = []
for i in range(runs):
    preds.append(_invoke_agent_on_image_url(signed_url))
    prog.progress((i + 1) / runs)

prog.empty()

# -------------------------- Results table -------------------------
df = pd.DataFrame(
    [
        {
            "run": i + 1,
            "store": p.get("store_name"),
            "total": p.get("total"),
            "date": p.get("purchase_datetime"),
            "error": p.get("error"),
        }
        for i, p in enumerate(preds)
    ]
)

with st.expander("Raw per-run results", expanded=False):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ------------------------ Consistency metrics ----------------------
stores = df["store"].tolist()
totals = df["total"].tolist()
dates  = df["date"].tolist()

mode_store = _mode_or_none(stores)
mode_total = _mode_or_none([t for t in totals if t is not None])  # numeric
mode_date  = _mode_or_none(dates)

store_consistency = _percent_equal(stores, mode_store, numeric_tolerance=None)
total_consistency = _percent_equal(totals, mode_total, numeric_tolerance=TOLERANCE) if mode_total is not None else 0.0
date_consistency  = _percent_equal(dates, mode_date, numeric_tolerance=None)

# Overall = mean of available field consistencies
avail = [store_consistency, date_consistency] + ([total_consistency] if mode_total is not None else [])
overall_consistency = sum(avail) / len(avail) if avail else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Store consistency", f"{store_consistency:.1f}%")
k2.metric(f"Total consistency (±${TOLERANCE:.02f})", f"{total_consistency:.1f}%")
k3.metric("Date consistency", f"{date_consistency:.1f}%")
k4.metric("Overall", f"{overall_consistency:.1f}%")

st.divider()

# ----------------------- Quick visual diff ------------------------
st.subheader("What the model is *usually* saying")
c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Store (mode)")
    st.write(mode_store or "—")
with c2:
    st.caption(f"Total (mode within ±${TOLERANCE:.02f})")
    st.write(f"{mode_total:.2f}" if isinstance(mode_total, (int, float)) else (mode_total or "—"))
with c3:
    st.caption("Date (mode)")
    st.write(mode_date or "—")

# Highlight mismatches vs mode
disp = df.copy()
def _fmt_total(x):
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

disp["store_ok"] = disp["store"].apply(lambda s: (s is not None and mode_store is not None and s.strip().casefold() == mode_store.strip().casefold()))
disp["total_ok"] = disp["total"].apply(lambda t: (isinstance(mode_total, (int, float)) and isinstance(t, (int, float)) and abs(t - mode_total) <= TOLERANCE))
disp["date_ok"]  = disp["date"].apply(lambda d: (d is not None and mode_date is not None and d == mode_date))

pretty = disp[["run", "store", "store_ok", "total", "total_ok", "date", "date_ok", "error"]].copy()
pretty["total"] = pretty["total"].map(_fmt_total)
pretty["store_ok"] = pretty["store_ok"].map(lambda b: "✔︎" if b else "✖︎")
pretty["total_ok"] = pretty["total_ok"].map(lambda b: "✔︎" if b else "✖︎")
pretty["date_ok"]  = pretty["date_ok"].map(lambda b: "✔︎" if b else "✖︎")

st.caption("Per-run agreement with the run-mode")
st.dataframe(pretty, use_container_width=True, hide_index=True)

st.caption(
    "This page checks how stable your agent’s header extraction is for a **single** receipt. "
    "If you see low consistency, consider prompt tightening, deterministic settings, or tool/vision tweaks."
)
