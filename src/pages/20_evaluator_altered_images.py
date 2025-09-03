# src/pages/20_evaluator_altered_images.py
# Evaluator — Perturbed Images (single-receipt)
# - First dropdown lists receipt_id from public.receipt_perturbations
# - Shows Original & Perturbed images side-by-side
# - "Robustness Awareness" button runs the agent on both and writes results to
#   public.receipts_robustness_eval_results

from __future__ import annotations

import os
import io
import re
import json
import math
import hashlib
import tempfile
from datetime import datetime, date
from typing import Dict, Any, List, Optional

import requests
import streamlit as st
import pandas as pd

from utils.supabase_utils import get_supabase_client
import utils.evals_repo as evals  # used for personal original image URL

# Your agent (optional; page still loads if not importable)
try:
    from agents.receipt_extractor.agent import app as receipt_graph
except Exception:
    receipt_graph = None

st.set_page_config(page_title="Evaluator — Perturbed (Single)", layout="wide")
st.title("🧪 Evaluator — Perturbed Images (Single Receipt)")

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "genai-analytics-bucket")
AGENT_VERSION = os.getenv("AGENT_VERSION", "app")
TOLERANCE = 0.05   # dollar tolerance for total

# ───────────────────────── Supabase helpers ─────────────────────────
@st.cache_resource(show_spinner=False)
def sb():
    return get_supabase_client()

def signed_url(bucket: str, storage_path: str, ttl: int = 3600) -> Optional[str]:
    """
    Allows per-row bucket (perturbed_bucket) while defaulting to env bucket.
    """
    try:
        resp = sb().storage.from_(bucket).create_signed_url(storage_path, ttl)
        if isinstance(resp, dict):
            return resp.get("signedURL") or resp.get("signed_url") or resp.get("url")
        return getattr(resp, "signedURL", None) or getattr(resp, "signed_url", None) or getattr(resp, "url", None)
    except Exception:
        return None

# ──────────────────────── Data access helpers ───────────────────────
@st.cache_data(ttl=60)
def list_receipt_ids_from_mappings(limit: int = 1000) -> List[str]:
    """
    Return distinct receipt ids from receipt_perturbations (personal only).
    For public rows (no receipt_id), we still include a pseudo-id based on original_public_path.
    """
    rows = (
        sb()
        .table("receipt_perturbations")
        .select("source_kind, original_receipt_id, original_public_path, insrt_dttm")
        .order("insrt_dttm", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )
    ids: List[str] = []
    seen = set()
    for r in rows:
        rid = r.get("original_receipt_id")
        if rid:
            if rid not in seen:
                seen.add(rid)
                ids.append(rid)
        else:
            # Use basename of public path as a pseudo-id (still traceable)
            op = r.get("original_public_path") or ""
            name = op.split("/")[-1] if op else None
            if name and name not in seen:
                seen.add(name)
                ids.append(name)
    return ids

@st.cache_data(ttl=60)
def mappings_for_receipt_id(receipt_or_name: str) -> List[Dict[str, Any]]:
    """
    Fetch all mapping rows for a given id (handles both personal and public pseudo-id)
    """
    # Try personal first (uuid receipt_id)
    rows = (
        sb().table("receipt_perturbations")
        .select("id, source_kind, original_receipt_id, original_receipt_file_id, original_public_path, "
                "perturbed_storage_path, perturbed_bucket, perturbed_receipt_file_id, "
                "perturb_type, params, insrt_dttm")
        .eq("original_receipt_id", receipt_or_name)
        .order("insrt_dttm", desc=True)
        .limit(50)
        .execute()
        .data or []
    )
    if rows:
        return rows

    # Else, public by filename
    # We match by basename of original_public_path to keep selection simple
    all_rows = (
        sb().table("receipt_perturbations")
        .select("id, source_kind, original_receipt_id, original_receipt_file_id, original_public_path, "
                "perturbed_storage_path, perturbed_bucket, perturbed_receipt_file_id, "
                "perturb_type, params, insrt_dttm")
        .is_("original_receipt_id", None)
        .order("insrt_dttm", desc=True)
        .limit(200)
        .execute()
        .data or []
    )
    want = []
    for r in all_rows:
        op = r.get("original_public_path") or ""
        name = op.split("/")[-1] if op else ""
        if name == receipt_or_name:
            want.append(r)
    return want

@st.cache_data(ttl=60)
def fetch_current_gold_for_personal(receipt_id: str) -> Optional[Dict[str, Any]]:
    try:
        rows = (
            sb()
            .table("receipts_gold")
            .select("receipt_id, store_name_gold, total_gold, purchase_date_gold, curr_rec_ind")
            .eq("receipt_id", receipt_id)
            .eq("curr_rec_ind", True)
            .limit(1)
            .execute()
            .data or []
        )
        return rows[0] if rows else None
    except Exception:
        return None

# ───────────────────────── Image helpers ─────────────────────────
def download_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def temp_path_for_bytes(b: bytes, suffix=".jpg") -> str:
    fd, path = tempfile.mkstemp(prefix="robust_one_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(b)
    return path

def normalize_ext_from_url(url: str) -> str:
    base = url.split("?", 1)[0]
    m = re.search(r"\.([A-Za-z0-9]+)$", base)
    ext = (m.group(1) if m else "").lower()
    if ext in ("jpg", "jpeg"): return "jpg"
    if ext == "png": return "png"
    return "jpg"

# ───────────────────────── Agent helpers ─────────────────────────
def run_agent_on_image_url(image_url: str) -> Dict[str, Any]:
    if not image_url:
        return {"error": "no-url"}
    if receipt_graph is None:
        return {"error": "agent-not-imported"}

    b = download_bytes(image_url)
    if not b:
        return {"error": "download-failed"}

    # Use extension for temp if possible
    suffix = "." + normalize_ext_from_url(image_url)
    path = temp_path_for_bytes(b, suffix=suffix)
    try:
        try:
            res = receipt_graph.invoke({"image_path": path})
        except Exception as e:
            return {"error": "invoke-exception", "exception": str(e)}

        if res is None:
            return {"error": "invoke-returned-none"}

        # Normalize response shapes you’ve used before
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
            "raw": {"keys": list(res.keys())}
        }

    finally:
        try: os.remove(path)
        except Exception: pass

# ───────────────────────── Eval helpers ─────────────────────────
def norm_date(x) -> Optional[str]:
    if not x: return None
    if isinstance(x, date): return x.isoformat()
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace("Z", "")).date().isoformat()
        except Exception:
            return x[:10] if len(x) >= 10 else x
    return None

def to_float(v) -> Optional[float]:
    try:
        if v in (None, "", "None"): return None
        f = float(v); return f if math.isfinite(f) else None
    except Exception:
        return None

def compare_to_gold(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    g_store = (gold.get("store_name_gold") or None)
    g_total = to_float(gold.get("total_gold"))
    g_date  = norm_date(gold.get("purchase_date_gold"))

    p_store = (pred.get("store_name") or None)
    p_total = to_float(pred.get("total"))
    p_date  = norm_date(pred.get("purchase_datetime"))

    store_ok = None
    if g_store or p_store:
        store_ok = ((g_store or "").strip().casefold() == (p_store or "").strip().casefold())

    total_ok = None; delta = None
    if g_total is not None and p_total is not None:
        delta = abs(p_total - g_total)
        total_ok = (delta <= TOLERANCE)

    date_ok = None
    if g_date or p_date:
        date_ok = (g_date == p_date)

    overall = None
    avail = [x for x in (store_ok, total_ok, date_ok) if x is not None]
    if avail: overall = all(avail)

    return {
        "gold_store": g_store, "pred_store": p_store, "store_pass": store_ok,
        "gold_total": g_total, "pred_total": p_total, "total_delta": delta, "total_pass": total_ok,
        "gold_date": g_date,  "pred_date": p_date,  "date_pass": date_ok,
        "overall_pass": overall,
    }

def compare_baseline_vs_perturbed(baseline: Dict[str, Any], perturbed: Dict[str, Any]) -> Dict[str, Any]:
    # Soft comparison for self-consistency (no gold):
    b_store = (baseline.get("store_name") or "").strip().casefold()
    p_store = (perturbed.get("store_name") or "").strip().casefold()
    store_same = (b_store == p_store) if (b_store or p_store) else None

    bt = to_float(baseline.get("total"))
    pt = to_float(perturbed.get("total"))
    total_same = None
    total_delta = None
    if bt is not None and pt is not None:
        total_delta = abs(bt - pt)
        total_same = (total_delta <= TOLERANCE)

    bd = norm_date(baseline.get("purchase_datetime"))
    pd_ = norm_date(perturbed.get("purchase_datetime"))
    date_same = (bd == pd_) if (bd or pd_) else None

    overall = None
    avail = [x for x in (store_same, total_same, date_same) if x is not None]
    if avail: overall = all(avail)

    return {
        "store_same": store_same,
        "total_same": total_same,
        "total_delta": total_delta,
        "date_same": date_same,
        "overall_same": overall
    }

# ───────────────────────── UI: pick receipt ─────────────────────────
ids = list_receipt_ids_from_mappings()
if not ids:
    st.info("No entries in public.receipt_perturbations yet. Create perturbed images first.")
    st.stop()

rid = st.selectbox("Select receipt_id (from mappings)", ids, index=0)

# gather all variants (latest first)
rows = mappings_for_receipt_id(rid)
if not rows:
    st.warning("No mapping rows found for that receipt.")
    st.stop()

# choose variant (if only one, this selector is hidden)
def _variant_label(r: Dict[str, Any]) -> str:
    ptype = r.get("perturb_type") or "unknown"
    params = r.get("params") or {}
    if isinstance(params, dict) and "angle" in params:
        return f"{ptype} (angle={params['angle']})"
    return ptype

variant = rows[0] if len(rows) == 1 else st.selectbox(
    "Variant",
    rows,
    format_func=_variant_label
)

# build URLs
source_kind = variant.get("source_kind")
orig_url: Optional[str] = None

if source_kind == "personal" and variant.get("original_receipt_file_id"):
    orig_url = evals.receipt_image_url(variant["original_receipt_file_id"])
elif source_kind == "public" and variant.get("original_public_path"):
    # Use the default bucket for original public images (your datasets live there)
    orig_url = signed_url(SUPABASE_BUCKET, variant["original_public_path"])
else:
    orig_url = None

pert_bucket = variant.get("perturbed_bucket") or SUPABASE_BUCKET
pert_path = variant.get("perturbed_storage_path")
pert_url = signed_url(pert_bucket, pert_path) if pert_path else None

# ───────────────────────── Show images ─────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Original")
    if orig_url: st.image(orig_url, use_container_width=True)
    else:        st.info("No original URL available for this row.")

with c2:
    ptype = variant.get("perturb_type") or "perturbed"
    st.markdown(f"### Perturbed — {ptype}")
    if pert_url: st.image(pert_url, use_container_width=True)
    else:        st.info("No perturbed URL available for this row.")

st.divider()

# ───────────────── Robustness Awareness (run test) ─────────────────
run_btn = st.button("🛡️ Robustness Awareness", type="primary", use_container_width=True)

if run_btn:
    if not (orig_url and pert_url):
        st.error("Missing URLs for original or perturbed image.")
        st.stop()

    with st.status("Running predictions on both images…", expanded=True) as s:
        base_pred = run_agent_on_image_url(orig_url)
        st.write({"baseline_pred": {k: base_pred.get(k) for k in ("store_name","total","purchase_datetime","error")}})

        pert_pred = run_agent_on_image_url(pert_url)
        st.write({"perturbed_pred": {k: pert_pred.get(k) for k in ("store_name","total","purchase_datetime","error")}})

        # Self-consistency
        self_cmp = compare_baseline_vs_perturbed(base_pred, pert_pred)
        st.write({"self_consistency": self_cmp})

        # Vs-gold (only for personal)
        gold_row = None
        if source_kind == "personal" and variant.get("original_receipt_id"):
            gold_row = fetch_current_gold_for_personal(variant["original_receipt_id"])

        if gold_row:
            base_vs_gold = compare_to_gold(base_pred, gold_row)
            pert_vs_gold = compare_to_gold(pert_pred, gold_row)
        else:
            base_vs_gold = None
            pert_vs_gold = None

        s.update(label="Saving results…", state="running")

        # ───────────── Persist results to receipts_robustness_eval_results ─────────────
        try:
            prms = variant.get("params")
            if isinstance(prms, (dict, list)):
                prms_hash = hashlib.sha1(json.dumps(prms, sort_keys=True).encode("utf-8")).hexdigest()
            else:
                prms_hash = hashlib.sha1(str(prms).encode("utf-8")).hexdigest() if prms is not None else None

            row = {
                "source_kind": source_kind,                              # 'personal' | 'public'
                "receipt_id": variant.get("original_receipt_id"),        # None for public
                "receipt_file_id": variant.get("original_receipt_file_id"),
                "original_public_path": variant.get("original_public_path"),

                "perturb_type": variant.get("perturb_type"),
                "params": prms,
                "params_hash": prms_hash,
                "perturbed_storage_path": pert_path,
                "perturbed_bucket": pert_bucket,
                "perturbed_receipt_file_id": variant.get("perturbed_receipt_file_id"),

                "agent_version": AGENT_VERSION,

                # Baseline (original)
                "base_store": base_pred.get("store_name"),
                "base_total": to_float(base_pred.get("total")),
                "base_purchase_date": norm_date(base_pred.get("purchase_datetime")),
                "base_error": base_pred.get("error"),

                # Perturbed
                "pert_store": pert_pred.get("store_name"),
                "pert_total": to_float(pert_pred.get("total")),
                "pert_purchase_date": norm_date(pert_pred.get("purchase_datetime")),
                "pert_error": pert_pred.get("error"),

                # Self-consistency
                "self_store_same": self_cmp.get("store_same"),
                "self_total_same": self_cmp.get("total_same"),
                "self_total_delta": self_cmp.get("total_delta"),
                "self_date_same": self_cmp.get("date_same"),
                "self_overall_same": self_cmp.get("overall_same"),

                # small debug payload
                "extra": {
                    "baseline_keys": list((base_pred or {}).keys()),
                    "perturbed_keys": list((pert_pred or {}).keys())
                }
            }

            if gold_row:
                # make sure gold types align to table
                row.update({
                    "gold_store": gold_row.get("store_name_gold"),
                    "gold_total": gold_row.get("total_gold"),
                    "gold_date":  norm_date(gold_row.get("purchase_date_gold")),
                })
                row.update({
                    "base_store_pass":   base_vs_gold.get("store_pass"),
                    "base_total_pass":   base_vs_gold.get("total_pass"),
                    "base_total_delta":  base_vs_gold.get("total_delta"),
                    "base_date_pass":    base_vs_gold.get("date_pass"),
                    "base_overall_pass": base_vs_gold.get("overall_pass"),
                })
                row.update({
                    "pert_store_pass":   pert_vs_gold.get("store_pass"),
                    "pert_total_pass":   pert_vs_gold.get("total_pass"),
                    "pert_total_delta":  pert_vs_gold.get("total_delta"),
                    "pert_date_pass":    pert_vs_gold.get("date_pass"),
                    "pert_overall_pass": pert_vs_gold.get("overall_pass"),
                })

            ins = sb().table("receipts_robustness_eval_results").insert(row).execute()
            s.update(label="Done.", state="complete")
            st.success("Results saved to receipts_robustness_eval_results")
            st.caption("Inserted row (truncated fields shown):")
            st.write({
                "id": (ins.data[0]["id"] if getattr(ins, "data", None) else None),
                "source_kind": row["source_kind"],
                "receipt_id": row["receipt_id"],
                "perturb_type": row["perturb_type"],
                "params": row["params"],
                "self_overall_same": row["self_overall_same"],
            })

        except Exception as e:
            s.update(label="Save failed.", state="error")
            st.error(f"DB insert failed: {e}")

    # KPIs / badges
    st.markdown("### Results")

    def badge_bool(b, txt_ok="PASS", txt_fail="FAIL"):
        if b is True:  return f"✅ {txt_ok}"
        if b is False: return f"🟡 {txt_fail}"
        return "—"

    csc1, csc2, csc3, csc4 = st.columns(4)
    csc1.metric("Store same", badge_bool(self_cmp["store_same"]))
    csc2.metric("Total within ±$0.05", badge_bool(self_cmp["total_same"]))
    csc3.metric("Date same", badge_bool(self_cmp["date_same"]))
    csc4.metric("Overall", badge_bool(self_cmp["overall_same"]))

    if gold_row:
        st.markdown("#### Vs Gold (personal)")

        def label_vs_gold(d: Dict[str, Any]) -> List[str]:
            out = []
            out.append(f"Store — {badge_bool(d['store_pass'])}: {d['gold_store']} vs {d['pred_store']}")
            if d["total_delta"] is not None:
                out.append(f"Total — {badge_bool(d['total_pass'])}: {d['gold_total']} vs {d['pred_total']} (Δ={d['total_delta']:.2f})")
            else:
                out.append(f"Total — {badge_bool(d['total_pass'])}: {d['gold_total']} vs {d['pred_total']}")
            out.append(f"Date — {badge_bool(d['date_pass'])}: {d['gold_date']} vs {d['pred_date']}")
            out.append(f"Overall — {badge_bool(d['overall_pass'])}")
            return out

        base_vs_gold = compare_to_gold(base_pred, gold_row)
        pert_vs_gold = compare_to_gold(pert_pred, gold_row)
        btxt = label_vs_gold(base_vs_gold)
        ptxt = label_vs_gold(pert_vs_gold)

        vg1, vg2 = st.columns(2)
        with vg1:
            st.caption("Baseline vs Gold")
            for line in btxt: st.write(line)
        with vg2:
            st.caption("Perturbed vs Gold")
            for line in ptxt: st.write(line)

st.caption(
    "Pick a receipt from `public.receipt_perturbations`. "
    "This tool runs your extractor on both the original and the chosen perturbed image, "
    "reports self-consistency, (for personal) vs gold, and persists results."
)
