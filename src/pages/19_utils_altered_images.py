# src/pages/19_utils_altered_images.py
# Utilities — Altered (Perturbed) Images
# - Auto-generate preview on Rotate dropdown change (no extra click)
# - Show ONLY the rotated image
# - SAVE button below the rotated image writes to Storage + DB link rows

from __future__ import annotations

import io
import os
import re
import hashlib
from typing import Optional, Dict, Any, List

import streamlit as st
import requests
from PIL import Image

from utils.supabase_utils import get_supabase_client
import utils.evals_repo as evals  # used to generate signed URL for personal receipts

st.set_page_config(page_title="Utilities — Altered (Perturbed) Images", layout="wide")
st.title("🧪 Utilities — Altered (Perturbed) Images")

# ---------------- Tunables ----------------
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "genai-analytics-bucket")
PERTURB_ROOT = "pertubed_images"           # keep spelling to match your bucket
SIGNED_URL_TTL = 900                       # seconds
MAX_SIDE = 1800                            # resize long edge before upload
JPEG_QUALITY = 88
# -----------------------------------------


# ---------------- Supabase helpers ----------------
@st.cache_resource(show_spinner=False)
def sb():
    return get_supabase_client()

def storage_signed_url(path: str, ttl: int = SIGNED_URL_TTL) -> Optional[str]:
    try:
        resp = sb().storage.from_(SUPABASE_BUCKET).create_signed_url(path, ttl)
        if isinstance(resp, dict):
            return resp.get("signedURL") or resp.get("signed_url") or resp.get("url")
        return getattr(resp, "signedURL", None) or getattr(resp, "signed_url", None) or getattr(resp, "url", None)
    except Exception:
        return None

def storage_upload(path: str, data: bytes, content_type: str, upsert: bool = True) -> Dict[str, Any]:
    """Normalize upload to {ok, path, error}."""
    try:
        resp = sb().storage.from_(SUPABASE_BUCKET).upload(
            path,
            data,
            file_options={"content-type": content_type, "upsert": "true" if upsert else "false"},
        )
        err = getattr(resp, "error", None)
        if isinstance(resp, dict):
            err = err or resp.get("error")
        if err:
            msg = getattr(err, "message", None) or getattr(err, "error", None) or str(err)
            return {"ok": False, "path": path, "error": msg}
        return {"ok": True, "path": path, "error": None}
    except Exception as e:
        return {"ok": False, "path": path, "error": f"{e.__class__.__name__}: {e}"}


# ---------------- Image helpers ----------------
def load_image_from_url(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except Exception:
        return None

def normalize_ext_from_url(url: str) -> str:
    base = url.split("?", 1)[0]
    m = re.search(r"\.([A-Za-z0-9]+)$", base)
    ext = (m.group(1) if m else "").lower()
    if ext in ("jpg", "jpeg"):
        return "jpg"
    if ext == "png":
        return "png"
    return "jpg"

def pil_to_bytes(img: Image.Image, ext: str = "jpg", quality: int = JPEG_QUALITY) -> bytes:
    bio = io.BytesIO()
    if ext == "png":
        if img.mode == "P":
            img = img.convert("RGBA")
        img.save(bio, format="PNG", optimize=True)
    else:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(bio, format="JPEG", quality=quality, optimize=True)
    return bio.getvalue()

def compress_for_upload(img: Image.Image, ext: str, max_side: int = MAX_SIDE) -> bytes:
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
    return pil_to_bytes(img, ext=ext, quality=JPEG_QUALITY)

def content_type_for_ext(ext: str) -> str:
    return "image/png" if ext == "png" else "image/jpeg"


# ---------------- Data sources ----------------
@st.cache_data(ttl=60)
def list_public(dataset: str, limit: int = 400) -> List[Dict[str, Any]]:
    table = {
        "sroie": "Receiptscanner_gold_public_dataset_sroie",
        "expressexpense": "Receiptscanner_gold_public_dataset_expressexpense",
        "retailus": "Receiptscanner_gold_public_dataset_retailus",
        "cord": "Receiptscanner_gold_public_dataset_cord",
        "hitl": "Receiptscanner_gold_public_dataset_hitl",
        "synthetic": "Receiptscanner_gold_public_dataset_synthetic",
    }.get(dataset, "Receiptscanner_gold_public_dataset_sroie")
    rows = sb().table(table).select("id, source_id, image_storage_path").limit(limit).execute().data or []
    out = []
    for r in rows:
        rid = r.get("source_id") or r.get("id")
        out.append({"receipt_id": str(rid), "image_path": r.get("image_storage_path"), "label": f"{rid} • {r.get('image_storage_path')}"})
    return out

@st.cache_data(ttl=60)
def list_personal(limit: int = 200) -> List[Dict[str, Any]]:
    rows = sb().table("receipts_dtl").select("id, receipt_file_id").order("created_at", desc=True).limit(limit).execute().data or []
    return [{"receipt_id": str(r["id"]), "receipt_file_id": r.get("receipt_file_id"), "label": f"{r['id']} • file:{r.get('receipt_file_id')}"} for r in rows]


# ---------------- Perturbation ----------------
def rotate_fixed(img: Image.Image, degrees: int) -> Image.Image:
    deg = degrees % 360
    if deg == 0:
        return img.copy()
    if deg == 90:
        return img.transpose(Image.ROTATE_270)  # PIL is CCW
    if deg == 180:
        return img.transpose(Image.ROTATE_180)
    if deg == 270:
        return img.transpose(Image.ROTATE_90)
    return img.rotate(deg, expand=True, resample=Image.NEAREST)


# ---------------- UI ----------------
left, right = st.columns([4, 6], gap="large")

# ===== Left: source & selection =====
with left:
    st.subheader("Source")
    src = st.radio("Choose source", ["Public", "Personal"], horizontal=True)

    original_image: Optional[Image.Image] = None
    ext = "jpg"
    choice = None

    if src == "Public":
        dataset = st.radio("Public dataset", ["sroie", "expressexpense", "retailus", "cord", "hitl", "synthetic"], horizontal=True)
        rows = list_public(dataset)
        if rows:
            choice = st.selectbox("Pick a receipt", options=rows, format_func=lambda r: r["label"])
            signed = storage_signed_url(choice["image_path"])
            if signed:
                with st.status("Loading image…", expanded=False):
                    original_image = load_image_from_url(signed)
                if original_image:
                    ext = normalize_ext_from_url(signed)
            else:
                st.error("Could not create a signed URL for that image.")
        else:
            st.info("No rows in that dataset.")
    else:
        rows = list_personal()
        if rows:
            choice = st.selectbox("Pick a personal receipt", options=rows, format_func=lambda r: r["label"])
            signed = evals.receipt_image_url(choice["receipt_file_id"])
            if signed:
                with st.status("Loading image…", expanded=False):
                    original_image = load_image_from_url(signed)
                if original_image:
                    ext = normalize_ext_from_url(signed)
            else:
                st.error("Could not create a signed URL for that personal image.")
        else:
            st.info("No personal receipts yet. Upload/scan one first.")

# ===== Right: rotate (auto) + SAVE =====
with right:
    st.subheader("Rotate & Save")

    if original_image is None:
        st.info("Select a source receipt on the left.")
        st.stop()

    # 1) Pick angle — auto-generate preview (no extra click)
    angle = st.selectbox("Rotate (degrees)", [0, 90, 180, 270], index=0)

    # 2) Auto-generate rotated image on every selection change
    rotated = rotate_fixed(original_image, int(angle))

    # 3) Show ONLY rotated image
    st.image(rotated, use_container_width=True, caption=f"rotate_fixed ({angle}°)")

    # 4) Persist bytes & meta so SAVE works on the rerun
    sid = (choice.get("receipt_id") or choice.get("image_path") or "unknown")
    filename = f"{sid}_pertubed_rotate_fixed_{angle}.{ext}"
    storage_path = f"{PERTURB_ROOT}/{filename}"
    content_type = content_type_for_ext(ext)

    st.session_state["pert_img_bytes"] = compress_for_upload(rotated, ext=ext, max_side=MAX_SIDE)
    st.session_state["save_meta"] = {
        "sid": sid,
        "storage_path": storage_path,
        "content_type": content_type,
        "source_kind": ("personal" if src == "Personal" else "public"),
        "receipt_id": choice.get("receipt_id"),
        "receipt_file_id": choice.get("receipt_file_id"),
        "public_path": choice.get("image_path"),
        "angle": int(angle),
        "ext": ext,
    }

    st.caption(f"Will save to: `{SUPABASE_BUCKET}/{storage_path}`")

    # 5) SAVE button below the rotated image
    save_key = f"SAVE::{sid}::{angle}"
    save_click = st.button("💾 SAVE", key=save_key, type="primary", use_container_width=True)

    if save_click:
        data = st.session_state.get("pert_img_bytes")
        meta = st.session_state.get("save_meta") or {}

        if not data or not meta:
            st.error("Missing preview bytes. Change the angle once and try again.")
            st.stop()

        with st.status("Saving to Supabase…", expanded=True) as s:
            # Dedupe key (same original + same angle)
            perturb_type = "rotate_fixed"
            params = {"angle": meta["angle"]}
            params_hash = hashlib.sha1(f"{perturb_type}|angle={meta['angle']}|{sid}".encode("utf-8")).hexdigest()

            # Optional: skip if mapping exists
            try:
                q = sb().table("receipt_perturbations").select("id, perturbed_storage_path")
                if meta["source_kind"] == "personal":
                    q = q.eq("source_kind", "personal").eq("original_receipt_id", meta["receipt_id"])
                else:
                    q = q.eq("source_kind", "public").eq("original_public_path", meta["public_path"])
                existing = q.eq("perturb_type", perturb_type).eq("params_hash", params_hash).limit(1).execute().data or []
            except Exception:
                existing = []

            if existing:
                s.update(label="Already existed — not re-uploading.", state="complete")
                st.warning("This perturbation already exists for this original.")
                st.code(existing[0], language="json")
                st.stop()

            # 1) Upload
            up = storage_upload(meta["storage_path"], data, content_type=meta["content_type"], upsert=True)
            st.write({"upload_result": up})
            if not up.get("ok"):
                s.update(label="Upload failed.", state="error")
                st.error(f"Upload failed → {up.get('error')}")
                st.stop()

            # 2) Create/find receipt_files row
            filename_only = meta["storage_path"].split("/", 1)[-1]
            rf = sb().table("receipt_files").select("id").eq("filename", filename_only).limit(1).execute().data or []
            perturbed_receipt_file_id = rf[0]["id"] if rf else None
            if not perturbed_receipt_file_id:
                ins = sb().table("receipt_files").insert({
                    "filename": filename_only,
                    "original_filename": filename_only,
                    "bucket_name": SUPABASE_BUCKET,
                    "mime_type": meta["content_type"],
                    "visitor_id": None,
                }).execute()
                if ins.data:
                    perturbed_receipt_file_id = ins.data[0]["id"]
            st.write({"receipt_files.id": perturbed_receipt_file_id})

            # 3) Insert mapping row
            mapping_row = {
                "source_kind": meta["source_kind"],
                "perturb_type": perturb_type,
                "params": params,
                "params_hash": params_hash,
                "perturbed_receipt_file_id": perturbed_receipt_file_id,
                "perturbed_storage_path": meta["storage_path"],
                "perturbed_bucket": SUPABASE_BUCKET,
            }
            if meta["source_kind"] == "personal":
                mapping_row["original_receipt_id"] = meta["receipt_id"]
                mapping_row["original_receipt_file_id"] = meta["receipt_file_id"]
            else:
                mapping_row["original_public_path"] = meta["public_path"]

            ins_map = sb().table("receipt_perturbations").insert(mapping_row).execute()
            st.write({"receipt_perturbations": getattr(ins_map, "data", None) or ins_map})

            s.update(label="Saved.", state="complete")
            url = storage_signed_url(meta["storage_path"])
            if url:
                st.success("Saved ✅")
                st.caption("Signed URL")
                st.code(url, language="text")
