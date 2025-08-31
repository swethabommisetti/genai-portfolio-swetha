# src/pages/21_evaluator_latency.py
import os
import io
import time
import math
import random
import tempfile
from datetime import datetime, date
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st

from utils.supabase_utils import get_supabase_client
import utils.evals_repo as evals  # for personal receipt -> signed URL

# Your LangGraph agent app (used to run predictions on images)
try:
    from agents.receipt_extractor.agent import app as receipt_graph
except Exception:
    receipt_graph = None

st.set_page_config(page_title="Evaluator — Latency & Stability", layout="wide")
st.title("⏱️ Evaluator — Latency & Stability")

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "genai-analytics-bucket")

# -------------------------------------------------------------------
# Supabase helpers
# -------------------------------------------------------------------
def sb():
    return get_supabase_client()

def make_signed_url(storage_path: str, ttl: int = 3600) -> Optional[str]:
    try:
        resp = sb().storage.from_(SUPABASE_BUCKET).create_signed_url(storage_path, ttl)
        return resp.get("signedURL")
    except Exception:
        return None

# -------------------------------------------------------------------
# Data access (same contracts as other pages)
# -------------------------------------------------------------------
@st.cache_data(ttl=60)
def list_public(dataset: str, limit: int = 2000) -> List[Dict[str, Any]]:
    table = {
        "sroie": "Receiptscanner_gold_public_dataset_sroie",
        "expressexpense": "Receiptscanner_gold_public_dataset_expressexpense",
    }[dataset]
    rows = (
        sb().table(table)
        .select("id, source_id, image_storage_path")
        .limit(limit)
        .execute()
        .data or []
    )
    out = []
    for r in rows:
        rid = r.get("source_id") or r.get("id")
        out.append({
            "receipt_id": str(rid),
            "image_path": r.get("image_storage_path"),
            "label": f"{rid} • {r.get('image_storage_path')}",
        })
    return out

@st.cache_data(ttl=60)
def list_personal(limit: int = 2000) -> List[Dict[str, Any]]:
    dtl = (
        sb().table("receipts_dtl")
        .select("id, receipt_file_id")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )
    out = []
    for d in dtl:
        rid = d.get("id")
        out.append({
            "receipt_id": str(rid),
            "receipt_file_id": d.get("receipt_file_id"),
            "label": f"{rid} • file:{d.get('receipt_file_id')}",
        })
    return out

def original_signed_url_public(row: Dict[str, Any]) -> Optional[str]:
    return make_signed_url(row["image_path"])

def original_signed_url_personal(row: Dict[str, Any]) -> Optional[str]:
    return evals.receipt_image_url(row["receipt_file_id"])

# -------------------------------------------------------------------
# Agent invocation helpers
# -------------------------------------------------------------------
def download_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def to_temp_file(b: bytes, suffix=".jpg") -> str:
    fd, path = tempfile.mkstemp(prefix="lat_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(b)
    return path

def normalize_header(res: Any) -> Dict[str, Any]:
    """
    Make a minimal success signal from agent output.
    We consider success if at least one of: store | total | date is present.
    """
    if not isinstance(res, dict):
        return {"store_name": None, "total": None, "purchase_datetime": None}

    # The agent sometimes stores under receipt_data.{header,totals}
    header = res.get("receipt_data", {}).get("header") or res.get("header") or {}
    totals = res.get("receipt_data", {}).get("totals") or res.get("totals") or {}

    store = header.get("store_name") or None

    total = header.get("total")
    if total is None:
        total = totals.get("grand_total")
    try:
        total = float(total) if total not in (None, "", "None") else None
    except Exception:
        total = None

    dt = header.get("purchase_datetime") or None
    # Best-effort ISO normalization
    if isinstance(dt, date):
        dt = dt.isoformat()
    elif isinstance(dt, str) and len(dt) >= 10:
        dt = dt[:10]
    else:
        dt = None

    return {"store_name": store, "total": total, "purchase_datetime": dt}

def run_once(image_url: str) -> Dict[str, Any]:
    """
    Time the agent on a single image URL.
    Returns: { ok: bool, error: str|None, latency_ms: float, summary: dict }
    """
    if not image_url:
        return {"ok": False, "error": "no-url", "latency_ms": None, "summary": {}}
    if receipt_graph is None:
        return {"ok": False, "error": "agent-not-imported", "latency_ms": None, "summary": {}}

    b = download_bytes(image_url)
    if not b:
        return {"ok": False, "error": "download-failed", "latency_ms": None, "summary": {}}

    path = to_temp_file(b, suffix=os.path.splitext(image_url.split("?")[0])[1] or ".jpg")
    try:
        t0 = time.perf_counter()
        res = receipt_graph.invoke({"image_path": path})
        t1 = time.perf_counter()
    except Exception as e:
        return {"ok": False, "error": f"agent-error: {e}", "latency_ms": None, "summary": {}}
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

    summary = normalize_header(res)
    ok = any([summary.get("store_name"), summary.get("total"), summary.get("purchase_datetime")])
    return {"ok": ok, "error": None if ok else "empty-header", "latency_ms": (t1 - t0) * 1000.0, "summary": summary}

# -------------------------------------------------------------------
# UI Controls
# -------------------------------------------------------------------
c_src, c_ds, c_n = st.columns([2, 2, 1])

with c_src:
    src = st.radio("Dataset source", ["Public", "Personal"], horizontal=True)

dataset = None
rows: List[Dict[str, Any]] = []
make_url = None

with c_ds:
    if src == "Public":
        dataset = st.radio("Public dataset", ["expressexpense", "sroie"], horizontal=True)
        rows = list_public(dataset)
        make_url = original_signed_url_public
    else:
        st.caption("Using your uploaded receipts.")
        rows = list_personal()
        make_url = original_signed_url_personal

with c_n:
    sample_n = st.number_input("Sample size", min_value=5, max_value=500, value=50, step=5)

if not rows:
    st.info("No receipts found for the selected source. Ingest/upload some first.")
    st.stop()

# Sample uniformly at random (stable enough for demo)
if len(rows) > sample_n:
    random.seed(42)
    rows = random.sample(rows, int(sample_n))

st.caption(f"Ready to evaluate **{len(rows)}** receipts.")

run_btn = st.button("Run latency benchmark", type="primary", use_container_width=True)
if not run_btn:
    st.stop()

# -------------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------------
results = []
prog = st.progress(0.0, text="Running…")

for i, r in enumerate(rows, start=1):
    rid = r["receipt_id"]
    url = make_url(r)
    one = run_once(url)
    results.append({
        "receipt_id": rid,
        "url": url,
        "ok": one["ok"],
        "error": one["error"],
        "latency_ms": one["latency_ms"],
        "store": (one["summary"] or {}).get("store_name"),
        "total": (one["summary"] or {}).get("total"),
        "date": (one["summary"] or {}).get("purchase_datetime"),
    })
    prog.progress(i / len(rows), text=f"Running… {i}/{len(rows)}")

prog.empty()

df = pd.DataFrame(results)

# -------------------------------------------------------------------
# KPIs
# -------------------------------------------------------------------
def pct(x):
    return float(x) * 100.0

def pctl(series, q):
    s = series.dropna()
    if len(s) == 0:
        return None
    return float(s.quantile(q))

lat_ok = df["latency_ms"].dropna()
p50 = pctl(df["latency_ms"], 0.50)
p95 = pctl(df["latency_ms"], 0.95)
mean = float(lat_ok.mean()) if len(lat_ok) else None
success = pct((df["ok"] == True).mean()) if len(df) else 0.0
errors = df["error"].dropna().value_counts().to_dict()

k1, k2, k3, k4 = st.columns(4)
k1.metric("p50 latency", f"{p50:.0f} ms" if p50 is not None else "—")
k2.metric("p95 latency", f"{p95:.0f} ms" if p95 is not None else "—")
k3.metric("mean latency", f"{mean:.0f} ms" if mean is not None else "—")
k4.metric("success rate", f"{success:.1f} %")

st.divider()

# Optional latency histogram
if len(lat_ok):
    st.caption("Latency distribution (ms)")
    st.bar_chart(lat_ok, height=180, use_container_width=True)

# Error breakdown
if errors:
    st.caption("Errors")
    err_view = pd.DataFrame([{"error": k, "count": v} for k, v in errors.items()])
    st.dataframe(err_view, use_container_width=True, hide_index=True)

st.subheader("Per-receipt results")
pretty = df.copy()
pretty["ok"] = pretty["ok"].map(lambda b: "PASS" if b else "FAIL")
pretty["latency_ms"] = pretty["latency_ms"].map(lambda v: f"{v:.0f}" if pd.notnull(v) else "—")
st.dataframe(
    pretty[["receipt_id", "ok", "latency_ms", "store", "total", "date", "error"]],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "This benchmark measures end-to-end time spent in your agent (download + local temp save + graph.invoke). "
    "Use this to compare models, prompt variants, or runtime settings."
)
