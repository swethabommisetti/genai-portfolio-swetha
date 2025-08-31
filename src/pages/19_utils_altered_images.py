# src/pages/19_utils_pertubed.py
import io
import os
import re
from typing import Optional, Dict, Any, List

import streamlit as st
import requests
from PIL import Image
import numpy as np

from utils.supabase_utils import get_supabase_client
import utils.evals_repo as evals  # personal receipt -> signed URL

st.set_page_config(page_title="Utilities — Create Perturbed Images", layout="wide")
st.title("🧪 Utilities — Create Perturbed Images")

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "genai-analytics-bucket")
PERTURB_ROOT = "pertubed_images"   # (keep your spelling)

# ----------------------------- helpers -----------------------------
def supabase() -> Any:
    return get_supabase_client()

def to_signed_url_from_path(storage_path: str, ttl: int = 3600) -> Optional[str]:
    try:
        resp = supabase().storage.from_(SUPABASE_BUCKET).create_signed_url(storage_path, ttl)
        return resp.get("signedURL") or resp.get("signed_url")
    except Exception:
        return None

def load_image_from_url(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=20)
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

def pil_to_bytes(img: Image.Image, ext: str = "jpg", quality: int = 90) -> bytes:
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

def compress_for_upload(img: Image.Image, ext: str, max_side: int = 1800) -> bytes:
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
    return pil_to_bytes(img, ext=ext, quality=88)

def content_type_for_ext(ext: str) -> str:
    return "image/png" if ext == "png" else "image/jpeg"

def upload_bytes_to_storage(path: str, data: bytes, content_type: str, upsert: bool = True) -> Dict[str, Any]:
    try:
        resp = supabase().storage.from_(SUPABASE_BUCKET).upload(
            path,
            data,
            file_options={"content-type": content_type, "x-upsert": "true" if upsert else "false"},
        )
        return resp or {"ok": True, "path": path}
    except Exception as e:
        return {"ok": False, "error": str(e), "path": path}

# ---------------------------- data loads ----------------------------
@st.cache_data(ttl=60)
def list_public(dataset: str, limit: int = 300) -> List[Dict[str, Any]]:
    table = {
        "sroie": "Receiptscanner_gold_public_dataset_sroie",
        "expressexpense": "Receiptscanner_gold_public_dataset_expressexpense",
    }[dataset]
    rows = (
        supabase()
        .table(table)
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
def list_personal(limit: int = 200) -> List[Dict[str, Any]]:
    dtl = (
        supabase()
        .table("receipts_dtl")
        .select("id, receipt_file_id")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )
    out = []
    for d in dtl:
        out.append({
            "receipt_id": str(d.get("id")),
            "receipt_file_id": d.get("receipt_file_id"),
            "label": f"{d.get('id')} • file:{d.get('receipt_file_id')}",
        })
    return out

# ----------------------- perturbations -----------------------
def p_rotate_fixed(img: Image.Image, degrees: int) -> Image.Image:
    """Fast 0/90/180/270 rotation using transpose (no resampling blur)."""
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

# WIP placeholders (for the dropdown only)
WIP_TYPES = [
    "blur (WIP)",
    "brightness (WIP)",
    "contrast (WIP)",
    "rotate (free-angle) (WIP)",
    "crop (WIP)",
    "noise (WIP)",
]

# ------------------------------- UI --------------------------------
left, right = st.columns([5, 7], gap="large")

# ===== Left: source and original preview =====
with left:
    st.subheader("Source")
    src = st.radio("Choose source", ["Public", "Personal"], horizontal=True)

    # values for right pane
    original_image: Optional[Image.Image] = None
    original_label: Optional[str] = None
    ext: str = "jpg"
    source_ok = False

    if src == "Public":
        dataset_name = st.radio("Public dataset", ["sroie", "expressexpense"], horizontal=True)
        rows = list_public(dataset_name)
        if not rows:
            st.warning("No rows found in the selected public dataset table. Ingest some first.")
        else:
            choice = st.selectbox("Pick a receipt", options=rows, format_func=lambda r: r["label"])
            original_label = choice["receipt_id"]
            signed = to_signed_url_from_path(choice["image_path"])
            if not signed:
                st.error("Failed to obtain a signed URL for the selected public image.")
            else:
                with st.status("Loading image…", expanded=False):
                    original_image = load_image_from_url(signed)
                if not original_image:
                    st.error("Could not load the image.")
                else:
                    ext = normalize_ext_from_url(signed)
                    source_ok = True

    else:  # Personal
        rows = list_personal()
        if not rows:
            st.info("No personal receipts yet. Upload/scan first.")
        else:
            choice = st.selectbox("Pick a personal receipt", options=rows, format_func=lambda r: r["label"])
            original_label = choice["receipt_id"]
            signed = evals.receipt_image_url(choice["receipt_file_id"])
            if not signed:
                st.error("Could not generate a signed URL for this image.")
            else:
                with st.status("Loading image…", expanded=False):
                    original_image = load_image_from_url(signed)
                if not original_image:
                    st.error("Could not load the image.")
                else:
                    ext = normalize_ext_from_url(signed)
                    source_ok = True

    st.divider()
    if original_image is not None:
        st.caption(f"Original (detected format: **{ext.upper()}**)")
        preview_img = original_image.convert("RGB") if original_image.mode in ("P", "LA") else original_image
        st.image(preview_img, use_container_width=True)

# ===== Right: perturb and save =====
with right:
    st.subheader("Perturbation")

    # Default is the new fixed-rotation; show other choices as WIP
    type_labels = ["rotate_fixed (recommended)"] + WIP_TYPES
    type_choice = st.selectbox("Type", type_labels, index=0)

    if type_choice != "rotate_fixed (recommended)":
        st.info("This perturbation is WIP. Please use **rotate_fixed (recommended)** for now.")
        st.stop()

    # Fixed rotation UI
    degrees = st.selectbox("Pick angle", [0, 90, 180, 270], index=0)
    st.caption(f"Value: {degrees}°")

    gen_btn = st.button("🔧 Generate preview", type="primary", use_container_width=True, disabled=not source_ok)

    perturbed: Optional[Image.Image] = None
    if gen_btn and original_image is not None:
        with st.spinner("Applying rotation…"):
            perturbed = p_rotate_fixed(original_image, int(degrees))

    if perturbed is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original")
            st.image(original_image, use_container_width=True)
        with c2:
            st.caption(f"Perturbed — rotate_fixed ({degrees}°)")
            st.image(perturbed, use_container_width=True)

        st.divider()

        # build storage path; keep your naming convention
        filename = f"{original_label}_pertubed_rotate_fixed.{ext}"
        storage_path = f"{PERTURB_ROOT}/{filename}"
        content_type = content_type_for_ext(ext)

        colA, colB = st.columns(2)
        with colA:
            save_btn = st.button("💾 Save perturbed image to Supabase", use_container_width=True)
        with colB:
            test_btn = st.button("🔬 Write healthcheck file", use_container_width=True)

        if save_btn:
            with st.status("Uploading to Supabase…", expanded=True) as s:
                data = compress_for_upload(perturbed, ext=ext, max_side=1800)
                st.write("Bucket:", SUPABASE_BUCKET)
                st.write("Path:", storage_path)
                st.write("Bytes:", len(data))
                res = upload_bytes_to_storage(storage_path, data, content_type=content_type, upsert=True)
                st.write("Upload response:", res)
                if res.get("ok") is False or res.get("error"):
                    s.update(label="Upload failed.", state="error")
                    st.error(f"Upload failed: {res}")
                else:
                    s.update(label="Upload complete.", state="complete")
                    st.success(f"Saved: {storage_path}")
                    signed_new = to_signed_url_from_path(storage_path)
                    if signed_new:
                        st.caption("Signed URL (temporary)")
                        st.write(signed_new)

        if test_btn:
            path = f"{PERTURB_ROOT}/_healthcheck.txt"
            res = upload_bytes_to_storage(path, b"hello", content_type="text/plain", upsert=True)
            st.write("Healthcheck upload:", res)

st.divider()
st.caption(
    f"Files are saved to Supabase Storage under "
    f"`{SUPABASE_BUCKET}/{PERTURB_ROOT}/{{receipt_id}}_pertubed_{{type}}.{{ext}}` "
    "(PNG stays PNG; JPG/JPEG stays JPG)."
)
