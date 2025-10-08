# src/pages/19_utils_altered_images.py
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
import utils.evals_repo as evals  # used to fetch a signed url for personal images

# ───────────────────────── Page meta ─────────────────────────
st.set_page_config(page_title="Utilities — Altered (Perturbed) Images", layout="wide")
st.title("🧪 Utilities — Altered (Perturbed) Images")

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "genai-analytics-bucket")
PERTURB_ROOT = "pertubed_images"   # keep spelling as used in your bucket
SIGNED_URL_TTL = 900
MAX_SIDE = 1800
JPEG_QUALITY = 88

# ─────────────────────── Supabase helpers ───────────────────────
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
    """
    Normalize upload result to {ok, path, error}.
    IMPORTANT: use 'upsert' (not 'x-upsert') for supabase-py.
    """
    try:
        resp = sb().storage.from_(SUPABASE_BUCKET).upload(
            path,
            data,
            file_options={"content-type": content_type, "upsert": "true" if upsert else "false"},
        )
        # supabase-py returns object with .data / .error OR a dict
        err = getattr(resp, "error", None)
        if isinstance(resp, dict):
            err = err or resp.get("error")
        if err:
            msg = getattr(err, "message", None) or getattr(err, "error", None) or str(err)
            return {"ok": False, "path": path, "error": msg}
        return {"ok": True, "path": path, "error": None}
    except Exception as e:
        return {"ok": False, "path": path, "error": f"{e.__class__.__name__}: {e}"}

def storage_list(prefix: str) -> List[Dict[str, Any]]:
    try:
        items = sb().storage.from_(SUPABASE_BUCKET).list(prefix)
        if isinstance(items, list):
            return [i if isinstance(i, dict) else dict(i) for i in items]
        return []
    except Exception:
        return []

# ─────────────────────── Image helpers ───────────────────────
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

# ─────────────────────── Data sources ───────────────────────
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

# ───────────────────── Perturbations ─────────────────────
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

# ─────────────────────────── UI ───────────────────────────
left, right = st.columns([5, 7], gap="large")

# ===== Left: source and original preview =====
with left:
    st.subheader("Source")
    src = st.radio("Choose source", ["Public", "Personal"], horizontal=True)

    original_image: Optional[Image.Image] = None
    ext = "jpg"
    source_ok = False
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
                    source_ok = True
            else:
                st.error("Could not create a signed URL for that image.")
        else:
            st.warning("No rows in that dataset.")

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
                    source_ok = True
            else:
                st.error("Could not create a signed URL for that personal image.")
        else:
            st.info("No personal receipts yet.")

    st.divider()
    if original_image is not None:
        st.caption(f"Original (detected format: **{ext.upper()}**)")
        preview = original_image.convert("RGB") if original_image.mode in ("P", "LA") else original_image
        st.image(preview, use_container_width=True)

# ===== Right: perturb & save =====
with right:
    st.subheader("Perturbation")
    angle = st.selectbox("Rotate (fixed angles)", [0, 90, 180, 270], index=0)
    gen_btn = st.button("🔧 Generate preview", type="primary", use_container_width=True, disabled=not source_ok)

    perturbed: Optional[Image.Image] = None
    if gen_btn and original_image is not None:
        with st.spinner("Applying rotation…"):
            perturbed = rotate_fixed(original_image, int(angle))

    if perturbed is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original")
            st.image(original_image, use_container_width=True)
        with c2:
            st.caption(f"Perturbed — rotate_fixed ({angle}°)")
            st.image(perturbed, use_container_width=True)

        st.divider()

        # Build storage path & filename (same convention as your earlier files)
        original_label = choice["receipt_id"]
        filename = f"{original_label}_pertubed_rotate_fixed.{ext}"
        storage_path = f"{PERTURB_ROOT}/{filename}"
        content_type = content_type_for_ext(ext)

        with st.container(border=True):
            st.write("**Preflight**")
            st.write("Bucket:", SUPABASE_BUCKET)
            st.write("Path:", storage_path)

        colA, colB, colC = st.columns([2, 1, 1])
        with colA:
            save_btn = st.button("💾 Save + DB Link", use_container_width=True)
        with colB:
            list_btn = st.button("🧪 List folder", use_container_width=True)
        with colC:
            view_btn = st.button("🔗 Get signed URL", use_container_width=True)

        if list_btn:
            items = storage_list(PERTURB_ROOT)
            if not items:
                st.warning(f"No objects found under `{PERTURB_ROOT}/`.")
            else:
                st.success(f"Found {len(items)} object(s) under `{PERTURB_ROOT}/`.")
                for it in items:
                    st.write("•", it.get("name") or it)

        if view_btn:
            url = storage_signed_url(storage_path)
            if url:
                st.code(url, language="text")
            else:
                st.info("No object yet at that path (or signed URL failed).")

        if save_btn:

            progress = st.progress(0, text="Preparing upload…")
            try:
                # Dedupe key for mapping table
                perturb_type = "rotate_fixed"
                params = {"angle": int(angle)}
                params_hash = hashlib.sha1(f"{perturb_type}|angle={int(angle)}|{original_label}".encode("utf-8")).hexdigest()

                # Check if mapping already exists
                try:
                    q = sb().table("receipt_perturbations").select("id, perturbed_storage_path")
                    if src == "Personal":
                        q = q.eq("source_kind", "personal").eq("original_receipt_id", choice["receipt_id"])
                    else:
                        q = q.eq("source_kind", "public").eq("original_public_path", choice["image_path"])
                    exists = q.eq("perturb_type", perturb_type).eq("params_hash", params_hash).limit(1).execute().data or []
                except Exception:
                    exists = []

                if exists:
                    progress.progress(100, text="Already existed")
                    st.warning("This perturbation already exists for this original:")
                    st.code(exists[0], language="json")
                    st.stop()

                # 1) bytes
                data = compress_for_upload(perturbed, ext=ext, max_side=MAX_SIDE)
                st.write("Bytes:", len(data))
                progress.progress(30, text="Uploading…")

                # 2) upload (correct 'upsert' key)
                up = storage_upload(storage_path, data, content_type=content_type, upsert=True)
                st.write("Upload:", up)
                if not up.get("ok"):
                    progress.progress(100, text="Upload failed")
                    st.error(f"Upload failed → {up.get('error')}")
                    st.stop()

                progress.progress(60, text="Ensuring receipt_files row…")

                # 3) create/find receipt_files record for perturbed image
                filename_only = storage_path.split("/", 1)[-1]
                rf = sb().table("receipt_files").select("id").eq("filename", filename_only).limit(1).execute().data or []
                perturbed_receipt_file_id = rf[0]["id"] if rf else None
                if not perturbed_receipt_file_id:
                    ins = sb().table("receipt_files").insert({
                        "filename": filename_only,
                        "original_filename": filename_only,
                        "bucket_name": SUPABASE_BUCKET,
                        "mime_type": content_type,
                        "visitor_id": None,
                    }).execute()
                    if ins.data:
                        perturbed_receipt_file_id = ins.data[0]["id"]
                st.write("receipt_files.id:", perturbed_receipt_file_id)

                progress.progress(85, text="Inserting mapping…")

                # 4) insert mapping row
                mapping_row = {
                    "source_kind": "personal" if src == "Personal" else "public",
                    "perturb_type": perturb_type,
                    "params": params,
                    "params_hash": params_hash,
                    "perturbed_receipt_file_id": perturbed_receipt_file_id,
                    "perturbed_storage_path": storage_path,
                    "perturbed_bucket": SUPABASE_BUCKET,
                }
                if src == "Personal":
                    mapping_row["original_receipt_id"] = choice["receipt_id"]
                    mapping_row["original_receipt_file_id"] = choice["receipt_file_id"]
                else:
                    mapping_row["original_public_path"] = choice["image_path"]

                ins_map = sb().table("receipt_perturbations").insert(mapping_row).execute()
                st.write("receipt_perturbations insert:", getattr(ins_map, "data", None) or ins_map)
                st.success("Linked original ↔ perturbed.")

                progress.progress(100, text="Verifying…")

                # 5) show signed URL to prove object exists
                url = storage_signed_url(storage_path)
                if url:
                    st.caption("Signed URL")
                    st.code(url, language="text")

            except Exception as e:
                progress.progress(100, text="Failed")
                st.error(f"Unexpected error: {e}")
                st.stop()

st.divider()
st.caption(
    f"Files save to `{SUPABASE_BUCKET}/{PERTURB_ROOT}/{{receipt_id}}_pertubed_rotate_fixed.{{ext}}`, "
    f"then a row is added to `receipt_files` and `receipt_perturbations`."
)
