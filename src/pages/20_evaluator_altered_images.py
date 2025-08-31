# src/pages/20_evaluator_pertubed_images.py
import os
import io
import math
import tempfile
from datetime import datetime, date
from typing import Dict, Any, List, Optional

import requests
import streamlit as st
import pandas as pd

from utils.supabase_utils import get_supabase_client
import utils.evals_repo as evals

# Your LangGraph agent app (used to run predictions on images)
try:
    from agents.receipt_extractor.agent import app as receipt_graph
except Exception:
    receipt_graph = None

st.set_page_config(page_title="Evaluator — Perturbed Images (Robustness)", layout="wide")
st.title("🧪 Evaluator — Altered Images (Robustness)")

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "genai-analytics-bucket")
TOLERANCE = 0.05  # $ tolerance for totals

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

def list_storage(path_prefix: str) -> List[Dict[str, Any]]:
    """List objects directly under a folder (no recursion)."""
    try:
        items = sb().storage.from_(SUPABASE_BUCKET).list(path_prefix)
        return items or []
    except Exception:
        return []

# -------------------------------------------------------------------
# Data access (gold + originals)
# -------------------------------------------------------------------
@st.cache_data(ttl=60)
def list_public(dataset: str, limit: int = 1000) -> List[Dict[str, Any]]:
    table = {
        "sroie": "Receiptscanner_gold_public_dataset_sroie",
        "expressexpense": "Receiptscanner_gold_public_dataset_expressexpense",
    }[dataset]
    rows = (
        sb().table(table)
        .select("id, source_id, image_storage_path, store_name_gold, total_gold, purchase_date_gold")
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
            "gold_store": r.get("store_name_gold"),
            "gold_total": r.get("total_gold"),
            "gold_date": r.get("purchase_date_gold"),
            "label": f"{rid} • {r.get('image_storage_path')}",
        })
    return out

@st.cache_data(ttl=60)
def list_personal(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Personal receipts + current gold.
    """
    dtl = (
        sb().table("receipts_dtl")
        .select("id, receipt_file_id")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )
    idx = {d["id"]: d for d in dtl}

    gold = (
        sb().table("receipts_gold")
        .select("receipt_id, store_name_gold, total_gold, purchase_date_gold, curr_rec_ind")
        .eq("curr_rec_ind", True)
        .limit(limit * 2)
        .execute()
        .data or []
    )
    out = []
    for g in gold:
        rid = g.get("receipt_id")
        d = idx.get(rid)
        if not d:
            continue
        out.append({
            "receipt_id": str(rid),
            "receipt_file_id": d.get("receipt_file_id"),
            "gold_store": g.get("store_name_gold"),
            "gold_total": g.get("total_gold"),
            "gold_date": g.get("purchase_date_gold"),
            "label": f"{rid} • file:{d.get('receipt_file_id')}",
        })
    return out

def original_signed_url_public(row: Dict[str, Any]) -> Optional[str]:
    return make_signed_url(row["image_path"])

def original_signed_url_personal(row: Dict[str, Any]) -> Optional[str]:
    return evals.receipt_image_url(row["receipt_file_id"])

# -------------------------------------------------------------------
# Perturbed files (FLAT FOLDER): pertubed_images/
#   expected names: {receipt_id}_pertubed_{ptype}.jpg|jpeg|png
# -------------------------------------------------------------------
PERTURBED_FOLDER = "pertubed_images"

def list_perturbed_ids_for_type(ptype: str) -> Dict[str, str]:
    """
    Return a map {receipt_id: filename} for files in pertubed_images/
    that match the chosen perturbation type and allowed extensions.
    """
    files = list_storage(PERTURBED_FOLDER)
    rid_to_file: Dict[str, str] = {}
    endings = [
        f"_pertubed_{ptype}.jpg",
        f"_pertubed_{ptype}.jpeg",
        f"_pertubed_{ptype}.png",
    ]
    for f in files:
        name = (f.get("name") or "").strip()
        if not name or name.startswith("_"):  # skip healthcheck, etc
            continue
        for suffix in endings:
            if name.lower().endswith(suffix):
                rid = name[: -len(suffix)]
                rid_to_file[rid] = name
                break
    return rid_to_file

def signed_url_for_perturbed(filename: str) -> Optional[str]:
    path = f"{PERTURBED_FOLDER}/{filename}"
    return make_signed_url(path)

# -------------------------------------------------------------------
# Prediction helpers
# -------------------------------------------------------------------
def download_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def to_temp_file(b: bytes, suffix=".jpg") -> str:
    fd, path = tempfile.mkstemp(prefix="robust_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(b)
    return path

def norm_date(x) -> Optional[str]:
    if not x:
        return None
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace("Z", "")).date().isoformat()
        except Exception:
            return x[:10] if len(x) >= 10 else x
    return None

def to_float(v) -> Optional[float]:
    try:
        if v in (None, "", "None"):
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def run_agent_on_image_url(image_url: str) -> Dict[str, Any]:
    if receipt_graph is None:
        return {"error": "agent-not-imported"}

    b = download_bytes(image_url)
    if not b:
        return {"error": "download-failed"}

    suffix = ".jpg"  # or use original extension if you’re tracking it
    path = to_temp_file(b, suffix=suffix)

    try:
        try:
            res = receipt_graph.invoke({"image_path": path})
        except Exception as e:
            return {"error": "invoke-exception", "exception": str(e)}

        # Log what actually came back
        if res is None:
            return {"error": "invoke-returned-none"}

        # If it isn’t a dict, report the type and repr
        res=receipt_graph.invoke({"image_path":path})

        if not isinstance(res, dict):
            return {"error": "unexpected-response-type", "type": str(type(res)), "repr": repr(res)}
        
        # If it is a dict, inspect keys so we don’t crash
        # Try both shapes you’ve used before:
        header = (
            (res.get("receipt_data") or {}).get("header") or
            res.get("header") or
            {}
        )
        totals = (
            (res.get("receipt_data") or {}).get("totals") or
            res.get("totals") or
            {}
        )

        pred_store = header.get("store_name")
        pred_total = header.get("total") or totals.get("grand_total")
        pred_date  = header.get("purchase_datetime")

        return {
            "store_name": pred_store,
            "total": pred_total,
            "purchase_datetime": pred_date,
            "raw": {"keys": list(res.keys())}  # small, safe echo for debugging
        }

    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def score_vs_gold(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    g_store = (gold.get("gold_store") or None)
    g_total = to_float(gold.get("gold_total"))
    g_date  = norm_date(gold.get("gold_date"))

    p_store = (pred.get("store_name") or None)
    p_total = to_float(pred.get("total"))
    p_date  = norm_date(pred.get("purchase_datetime"))

    store_ok = None
    if g_store or p_store:
        store_ok = ((g_store or "").strip().casefold() == (p_store or "").strip().casefold())

    total_ok = None
    delta = None
    if g_total is not None and p_total is not None:
        delta = abs(p_total - g_total)
        total_ok = (delta <= TOLERANCE)

    date_ok = None
    if g_date or p_date:
        date_ok = (g_date == p_date)

    overall = None
    avail = [x for x in (store_ok, total_ok, date_ok) if x is not None]
    if avail:
        overall = all(avail)

    return {
        "pred_store": p_store, "gold_store": g_store, "store_pass": store_ok,
        "pred_total": p_total, "gold_total": g_total, "total_delta": delta, "total_pass": total_ok,
        "pred_date": p_date, "gold_date": g_date, "date_pass": date_ok,
        "overall_pass": overall
    }

# -------------------------------------------------------------------
# UI controls
# -------------------------------------------------------------------
c_src, c_ptype = st.columns([2, 2])

with c_src:
    src = st.radio("Dataset source", ["Public", "Personal"], horizontal=True)

dataset = None
rows: List[Dict[str, Any]] = []
make_orig_url = None

if src == "Public":
    dataset = st.radio("Public dataset", ["expressexpense", "sroie"], horizontal=True, key="rob_ds")
    rows = list_public(dataset)
    make_orig_url = original_signed_url_public
else:
    rows = list_personal()
    make_orig_url = original_signed_url_personal

with c_ptype:
    ptype = st.selectbox(
        "Perturbation type",
        ["rotate_fixed", "blur(WIP)", "brightness(WIP)", "contrast(WIP)", "rotate(WIP)", "crop(WIP)", "noise(WIP)"],
        index=0,
        help="Files are read from bucket folder 'pertubed_images/'."
    )

if not rows:
    st.info("No receipts found for the selected source. Add gold or ingest a public dataset first.")
    st.stop()

# Which receipts have a perturbed file available (in flat folder)
rid_to_file = list_perturbed_ids_for_type(ptype)
eligible = [r for r in rows if r["receipt_id"] in rid_to_file]

if not eligible:
    st.warning(
        f"No perturbed images found in '{PERTURBED_FOLDER}' for type “{ptype}”. "
        f"Expected filenames like '{{receipt_id}}_pertubed_{ptype}.jpg|png'."
    )
    st.stop()

# (soft) cap to keep runs reasonable
eligible = eligible[:300]
st.caption(f"Evaluating {len(eligible)} receipts with available perturbed images (type: {ptype}).")

run_btn = st.button("Run robustness evaluation", type="primary", use_container_width=True)
if not run_btn:
    st.stop()

# -------------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------------
baseline_results = []
perturbed_results = []

prog = st.progress(0.0, text="Running predictions…")

for i, r in enumerate(eligible, start=1):
    rid = r["receipt_id"]
    gold = {"gold_store": r.get("gold_store"), "gold_total": r.get("gold_total"), "gold_date": r.get("gold_date")}

    # original
    orig_url = make_orig_url(r)
    base_pred = run_agent_on_image_url(orig_url) if orig_url else {"error": "no-original-url"}
    base_score = score_vs_gold(base_pred, gold)
    baseline_results.append({"receipt_id": rid, "original_url": orig_url, **base_score})

    # perturbed (from flat folder)
    pert_file = rid_to_file.get(rid)
    pert_url = signed_url_for_perturbed(pert_file) if pert_file else None
    pert_pred = run_agent_on_image_url(pert_url) if pert_url else {"error": "no-perturbed-url"}
    pert_score = score_vs_gold(pert_pred, gold)
    perturbed_results.append({"receipt_id": rid, "perturbed_url": pert_url, **pert_score})

    prog.progress(i / len(eligible), text=f"Running predictions… {i}/{len(eligible)}")

prog.empty()

base_df = pd.DataFrame(baseline_results)
pert_df = pd.DataFrame(perturbed_results)

def acc(df, col):
    s = df[col].dropna()
    return float(s.mean()) * 100.0 if len(s) else None

# KPIs
b_store = acc(base_df, "store_pass")
p_store = acc(pert_df, "store_pass")
b_total = acc(base_df, "total_pass")
p_total = acc(pert_df, "total_pass")
b_date  = acc(base_df, "date_pass")
p_date  = acc(pert_df, "date_pass")
b_over  = acc(base_df, "overall_pass")
p_over  = acc(pert_df, "overall_pass")

k1, k2, k3 = st.columns(3)
k1.metric("Store match", f"{(p_store or 0):.1f}%", delta=f"{((p_store or 0)-(b_store or 0)):.1f} pp")
k2.metric("Total within ±$0.05", f"{(p_total or 0):.1f}%", delta=f"{((p_total or 0)-(b_total or 0)):.1f} pp")
k3.metric("Date match", f"{(p_date or 0):.1f}%", delta=f"{((p_date or 0)-(b_date or 0)):.1f} pp")

k4 = st.columns(1)[0]
k4.metric("Overall (all 3)", f"{(p_over or 0):.1f}%", delta=f"{((p_over or 0)-(b_over or 0)):.1f} pp")

st.divider()

# -------------------------------------------------------------------
# Failures / inspection
# -------------------------------------------------------------------
st.subheader("🔎 Inspect differences")

merged = pd.merge(
    base_df[["receipt_id", "store_pass", "total_pass", "date_pass", "overall_pass"]],
    pert_df[["receipt_id", "store_pass", "total_pass", "date_pass", "overall_pass"]],
    on="receipt_id",
    suffixes=("_base", "_pert")
)

def worsened(row) -> bool:
    for col in ["store_pass", "total_pass", "date_pass", "overall_pass"]:
        b = row[f"{col}_base"]
        p = row[f"{col}_pert"]
        if b is True and p is False:
            return True
    return False

merged["worsened"] = merged.apply(worsened, axis=1)
show = merged[(merged["worsened"]) | (merged["overall_pass_pert"] == False)].copy()

def label_bool(b):
    return "PASS" if b is True else ("FAIL" if b is False else "—")

if show.empty:
    st.success("No degradations detected for the selected perturbed set.")
else:
    view = show.copy()
    for c in ["store_pass_base","store_pass_pert","total_pass_base","total_pass_pert",
              "date_pass_base","date_pass_pert","overall_pass_base","overall_pass_pert"]:
        view[c] = view[c].map(label_bool)
    st.dataframe(view[["receipt_id",
                       "store_pass_base","store_pass_pert",
                       "total_pass_base","total_pass_pert",
                       "date_pass_base","date_pass_pert",
                       "overall_pass_base","overall_pass_pert"]],
                 use_container_width=True, hide_index=True)

    st.markdown("### Receipt details")

    bmap = {r["receipt_id"]: r for _, r in base_df.iterrows()}
    pmap = {r["receipt_id"]: r for _, r in pert_df.iterrows()}
    orig_url_map = {r["receipt_id"]: r["original_url"] for _, r in base_df.iterrows()}
    pert_url_map = {r["receipt_id"]: r["perturbed_url"] for _, r in pert_df.iterrows()}

    for _, row in show.iterrows():
        rid = row["receipt_id"]
        with st.expander(f"{rid}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Original")
                url = orig_url_map.get(rid)
                if url:
                    st.image(url, use_container_width=True)
                else:
                    st.info("No original URL.")
            with c2:
                st.caption(f"Perturbed — {ptype}")
                urlp = pert_url_map.get(rid)
                if urlp:
                    st.image(urlp, use_container_width=True)
                else:
                    st.info("No perturbed URL.")

            b = bmap.get(rid, {})
            p = pmap.get(rid, {})

            def badge(ok, txt):
                if ok is True:   return f"✅ {txt} — PASS"
                if ok is False:  return f"🟡 {txt} — FAIL"
                return f"— {txt}"

            st.markdown("**Header comparison vs gold**")
            c3, c4 = st.columns(2)
            with c3:
                st.caption("Baseline prediction")
                st.write(badge(b.get("store_pass"), f"Store: {b.get('gold_store') or '—'} vs {b.get('pred_store') or '—'}"))
                tot_txt = f"{b.get('pred_total')}  (Δ={b.get('total_delta'):.2f})" if b.get("total_delta") is not None else f"{b.get('pred_total')}"
                st.write(badge(b.get("total_pass"), f"Total: {b.get('gold_total')} vs {tot_txt}"))
                st.write(badge(b.get("date_pass"),  f"Date:  {b.get('gold_date')} vs {b.get('pred_date')}"))
            with c4:
                st.caption("Perturbed prediction")
                st.write(badge(p.get("store_pass"), f"Store: {p.get('gold_store') or '—'} vs {p.get('pred_store') or '—'}"))
                tot_txt2 = f"{p.get('pred_total')}  (Δ={p.get('total_delta'):.2f})" if p.get("total_delta") is not None else f"{p.get('pred_total')}"
                st.write(badge(p.get("total_pass"), f"Total: {p.get('gold_total')} vs {tot_txt2}"))
                st.write(badge(p.get("date_pass"),  f"Date:  {p.get('gold_date')} vs {p.get('pred_date')}"))

st.caption(
    "This evaluator compares **baseline vs perturbed** performance against gold labels. "
    f"Perturbed samples are read from Supabase Storage under "
    f"`{SUPABASE_BUCKET}/pertubed_images/{{receipt_id}}_pertubed_{{type}}.(jpg|jpeg|png)`."
)
