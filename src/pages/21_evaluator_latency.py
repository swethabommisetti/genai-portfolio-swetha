# src/pages/21_evaluator_latency.py
from __future__ import annotations

import os
import io
import time
import math
import tempfile
from datetime import datetime, date
from typing import Dict, Any, List, Optional

import json
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
st.title("⏱️ Evaluator — Latency & Stability (Single Receipt)")

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "genai-analytics-bucket")
AGENT_VERSION   = os.getenv("AGENT_VERSION", "app")
MODEL_NAME      = os.getenv("MODEL_NAME")      # optional
MODEL_VERSION   = os.getenv("MODEL_VERSION")   # optional

# -------------------------------------------------------------------
# Supabase helpers
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def sb():
    return get_supabase_client()

def make_signed_url(storage_path: str, ttl: int = 3600) -> Optional[str]:
    try:
        resp = sb().storage.from_(SUPABASE_BUCKET).create_signed_url(storage_path, ttl)
        if isinstance(resp, dict):
            return resp.get("signedURL") or resp.get("signed_url") or resp.get("url")
        return getattr(resp, "signedURL", None) or getattr(resp, "signed_url", None) or getattr(resp, "url", None)
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
# Agent invocation helpers (with detailed timings)
# -------------------------------------------------------------------
def _ext_from_url(url: str) -> str:
    base = url.split("?", 1)[0]
    _, dot, ext = base.rpartition(".")
    ext = (ext or "").lower()
    return "png" if ext == "png" else "jpg"

def _mime_from_ext(ext: str) -> str:
    return "image/png" if ext == "png" else "image/jpeg"

def download_bytes_with_timing(url: str) -> Dict[str, Any]:
    """
    Returns: { ok, error, bytes, download_ms }
    """
    t0 = time.perf_counter()
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        b = r.content
        t1 = time.perf_counter()
        return {"ok": True, "error": None, "bytes": b, "download_ms": (t1 - t0) * 1000.0}
    except Exception as e:
        t1 = time.perf_counter()
        return {"ok": False, "error": f"download-error: {e}", "bytes": None, "download_ms": (t1 - t0) * 1000.0}

def write_temp(b: bytes, suffix: str) -> Dict[str, Any]:
    """
    Returns: { path, write_ms }
    """
    t0 = time.perf_counter()
    fd, path = tempfile.mkstemp(prefix="lat_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(b)
    t1 = time.perf_counter()
    return {"path": path, "write_ms": (t1 - t0) * 1000.0}

def invoke_agent(path: str) -> Dict[str, Any]:
    """
    Returns: { ok, error, res, inference_ms }
    """
    if receipt_graph is None:
        return {"ok": False, "error": "agent-not-imported", "res": None, "inference_ms": None}
    t0 = time.perf_counter()
    try:
        res = receipt_graph.invoke({"image_path": path})
    except Exception as e:
        return {"ok": False, "error": f"agent-error: {e}", "res": None, "inference_ms": None}
    t1 = time.perf_counter()
    return {"ok": True, "error": None, "res": res, "inference_ms": (t1 - t0) * 1000.0}

def normalize_header(res: Any) -> Dict[str, Any]:
    """
    Minimal success signal: True if any of store|total|date present.
    """
    if not isinstance(res, dict):
        return {"store_name": None, "total": None, "purchase_datetime": None}

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
    if isinstance(dt, date):
        dt = dt.isoformat()
    elif isinstance(dt, str) and len(dt) >= 10:
        dt = dt[:10]
    else:
        dt = None

    return {"store_name": store, "total": total, "purchase_datetime": dt}

def run_once(image_url: str) -> Dict[str, Any]:
    """
    Returns detailed timings & summary:
    {
      ok, status, error,
      image_bytes, image_mime,
      download_ms, preprocess_ms, inference_ms, postprocess_ms, total_ms,
      summary: {store_name,total,purchase_datetime}
    }
    """
    if not image_url:
        return {"ok": False, "status": "error", "error": "no-url"}

    # download
    dl = download_bytes_with_timing(image_url)
    if not dl["ok"]:
        return {
            "ok": False, "status": "error", "error": dl["error"],
            "image_bytes": None, "image_mime": None,
            "download_ms": dl["download_ms"], "preprocess_ms": None, "inference_ms": None,
            "postprocess_ms": None, "total_ms": dl["download_ms"], "summary": {}
        }

    b = dl["bytes"]
    ext = _ext_from_url(image_url)
    mime = _mime_from_ext(ext)
    image_bytes = len(b) if isinstance(b, (bytes, bytearray)) else None

    # write temp (preprocess)
    wt = write_temp(b, suffix=f".{ext}")

    # invoke agent
    inv = invoke_agent(wt["path"])

    # cleanup
    try:
        os.remove(wt["path"])
    except Exception:
        pass

    if not inv["ok"]:
        total_ms = (dl["download_ms"] or 0) + (wt["write_ms"] or 0) + (inv["inference_ms"] or 0)
        return {
            "ok": False, "status": "error", "error": inv["error"],
            "image_bytes": image_bytes, "image_mime": mime,
            "download_ms": dl["download_ms"], "preprocess_ms": wt["write_ms"],
            "inference_ms": inv["inference_ms"], "postprocess_ms": None,
            "total_ms": total_ms, "summary": {}
        }

    summary = normalize_header(inv["res"])
    total_ms = (dl["download_ms"] or 0) + (wt["write_ms"] or 0) + (inv["inference_ms"] or 0)
    ok = any([summary.get("store_name"), summary.get("total"), summary.get("purchase_datetime")])

    return {
        "ok": ok, "status": "ok" if ok else "error", "error": None if ok else "empty-header",
        "image_bytes": image_bytes, "image_mime": mime,
        "download_ms": dl["download_ms"], "preprocess_ms": wt["write_ms"],
        "inference_ms": inv["inference_ms"], "postprocess_ms": None,
        "total_ms": total_ms, "summary": summary
    }

# -------------------------------------------------------------------
# UI Controls — pick ONE receipt and N trials
# -------------------------------------------------------------------
c_src, c_ds, c_trials = st.columns([2, 2, 1])

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

with c_trials:
    trials = st.number_input("Trials", min_value=3, max_value=200, value=20, step=1)

if not rows:
    st.info("No receipts found for the selected source. Ingest/upload some first.")
    st.stop()

choice = st.selectbox("Pick a receipt", options=rows, format_func=lambda r: r["label"])

# Build URL + linkage fields
if src == "Public":
    source_kind = "public"
    url = make_url(choice)
    receipt_id = None
    receipt_file_id = None
    original_public_path = choice.get("image_path")
else:
    source_kind = "personal"
    url = make_url(choice)
    receipt_id = choice.get("receipt_id")
    receipt_file_id = choice.get("receipt_file_id")
    original_public_path = None

# ---- NEW: show the selected receipt immediately ----
st.subheader("Preview")
if url:
    st.image(url, use_container_width=True, caption=f"Receipt preview • source={source_kind}")
else:
    st.warning("Could not create a signed URL for this receipt.")

st.caption(f"Selected: `{choice['label']}`")

run_btn = st.button("Run latency benchmark", type="primary", use_container_width=True)
if not run_btn:
    st.stop()

# -------------------------------------------------------------------
# Run evaluation (N trials)
# -------------------------------------------------------------------
results = []
prog = st.progress(0.0, text="Running…")

for i in range(1, int(trials) + 1):
    one = run_once(url)
    results.append({
        "trial": i,
        "ok": one.get("ok"),
        "status": one.get("status"),
        "error": one.get("error"),
        "latency_ms": one.get("total_ms"),
        "download_ms": one.get("download_ms"),
        "preprocess_ms": one.get("preprocess_ms"),
        "inference_ms": one.get("inference_ms"),
        "postprocess_ms": one.get("postprocess_ms"),
        "store": (one.get("summary") or {}).get("store_name"),
        "total": (one.get("summary") or {}).get("total"),
        "date": (one.get("summary") or {}).get("purchase_datetime"),
        "image_bytes": one.get("image_bytes"),
        "image_mime": one.get("image_mime"),
    })
    prog.progress(i / trials, text=f"Running… {i}/{trials}")

prog.empty()

df = pd.DataFrame(results)

# -------------------------------------------------------------------
# Write to DB — receipts_latency_eval_results (one row per trial)
# -------------------------------------------------------------------
rows_to_insert = []
for r in results:
    rows_to_insert.append({
        "source_kind": source_kind,                 # 'personal' | 'public'
        "receipt_id": receipt_id,                   # uuid | None
        "receipt_file_id": receipt_file_id,         # uuid | None
        "original_public_path": original_public_path,

        "agent_version": AGENT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "request_id": None,                         # if you have one from your stack

        "image_bytes": r["image_bytes"],
        "image_mime": r["image_mime"],

        "download_ms": r["download_ms"],
        "preprocess_ms": r["preprocess_ms"],
        "inference_ms": r["inference_ms"],
        "postprocess_ms": r["postprocess_ms"],
        "total_ms": r["latency_ms"],

        "status": r["status"],
        "error": r["error"],
        "extra": {
            "trial": r["trial"],
            "store": r["store"],
            "total": r["total"],
            "date":  r["date"]
        },  # jsonb
    })

try:
    ins = sb().table("receipts_latency_eval_results").insert(rows_to_insert).execute()
    st.success(f"Saved {len(rows_to_insert)} trial rows to receipts_latency_eval_results.")
except Exception as e:
    st.error(f"DB insert failed: {e}")

# -------------------------------------------------------------------
# KPIs for the chosen receipt
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
success = pct((df["status"] == "ok").mean()) if len(df) else 0.0
errors = df["error"].dropna().value_counts().to_dict()

k1, k2, k3, k4 = st.columns(4)
k1.metric("p50 latency", f"{p50:.0f} ms" if p50 is not None else "—")
k2.metric("p95 latency", f"{p95:.0f} ms" if p95 is not None else "—")
k3.metric("mean latency", f"{mean:.0f} ms" if mean is not None else "—")
k4.metric("success rate", f"{success:.1f} %")

st.divider()

# Optional latency histogram (across trials)
if len(lat_ok):
    st.caption("Latency distribution (ms) across trials")
    st.bar_chart(lat_ok, height=180, use_container_width=True)

# Error breakdown
if errors:
    st.caption("Errors")
    err_view = pd.DataFrame([{"error": k, "count": v} for k, v in errors.items()])
    st.dataframe(err_view, use_container_width=True, hide_index=True)

st.subheader("Per-trial results")
pretty = df.copy()
pretty["ok"] = pretty["status"].map(lambda s: "PASS" if s == "ok" else "FAIL")
pretty["latency_ms"] = pretty["latency_ms"].map(lambda v: f"{v:.0f}" if pd.notnull(v) else "—")
pretty["download_ms"] = pretty["download_ms"].map(lambda v: f"{v:.0f}" if pd.notnull(v) else "—")
pretty["preprocess_ms"] = pretty["preprocess_ms"].map(lambda v: f"{v:.0f}" if pd.notnull(v) else "—")
pretty["inference_ms"] = pretty["inference_ms"].map(lambda v: f"{v:.0f}" if pd.notnull(v) else "—")
st.dataframe(
    pretty[["trial", "ok", "latency_ms", "download_ms", "preprocess_ms", "inference_ms", "store", "total", "date", "error"]],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Measures end-to-end time: download + temp write (preprocess) + graph.invoke. "
    "Each trial is recorded in `public.receipts_latency_eval_results`."
)
