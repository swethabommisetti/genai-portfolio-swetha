# src/pages/17_utils_home.py
import os
import io
import csv
import zipfile
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import streamlit as st
from utils.supabase_utils import get_supabase_client

st.set_page_config(page_title="Utilities — Gold Dataset Ingest", layout="wide")
st.title("🧰 Utilities — Gold Dataset Ingest")
st.caption("- Upload your own Gold Dataset \n"
           "- Best Viewed in Web")

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
TABLE_PREFIX = "Receiptscanner_gold_public_dataset_"
BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "genai-analytics-bucket")
MAX_ROWS_PER_DATASET = 1000  # soft cap to protect free tier

# All datasets (keep code for all 6)
ALL_DATASETS: List[Dict] = [
    {"key": "sroie",          "label": "SROIE",           "blurb": "ICDAR receipt IE",          "license": "research"},
    {"key": "cord",           "label": "CORD",            "blurb": "Consolidated receipts",     "license": "research"},
    {"key": "expressexpense", "label": "ExpressExpense",  "blurb": "Sample receipt images",     "license": "MIT"},
    {"key": "hitl",           "label": "Humans in the Loop", "blurb": "Free receipt OCR set",  "license": "CC0"},
    {"key": "retailus",       "label": "Retail (US)",     "blurb": "US store receipts",         "license": "varies"},
    {"key": "synthetic",      "label": "Synthetic",       "blurb": "Programmatically generated","license": "internal"},
]

# 🔒 Show only these two in the UI (keep others hidden but intact)
ENABLED_KEYS = {"sroie", "expressexpense"}
DATASETS = [d for d in ALL_DATASETS if d["key"] in ENABLED_KEYS]

supabase = get_supabase_client()

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _sha256(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _table_name(dataset_key: str) -> str:
    return f"{TABLE_PREFIX}{dataset_key}"

def _ensure_table_exists(dataset_key: str) -> bool:
    """Assumes you already ran the DDL. Warn if missing."""
    table = _table_name(dataset_key)
    try:
        supabase.table(table).select("id").limit(1).execute()
        return True
    except Exception as e:
        st.error(f"Table `{table}` not found. Did you run the SQL DDL? • {e}")
        return False

def _storage_upload(path: str, content: bytes, mime: Optional[str] = None) -> bool:
    """Upload to Supabase Storage. Returns True on success/exists."""
    try:
        supabase.storage.from_(BUCKET).upload(
            path, content,
            {"content-type": mime or "application/octet-stream", "upsert": False}
        )
        return True
    except Exception as e:
        # treat "already exists" as success for idempotency
        if "exists" in str(e).lower() or "duplicate" in str(e).lower():
            return True
        return False

def _storage_signed_url(path: str, seconds: int = 3600) -> Optional[str]:
    try:
        res = supabase.storage.from_(BUCKET).create_signed_url(path, seconds)
        return res.get("signedURL") or res.get("signed_url")
    except Exception:
        return None

def _parse_labels_csv(zf: zipfile.ZipFile) -> Dict[str, Dict]:
    """
    Optional labels.csv inside the zip (UTF-8):
    filename,store_name_gold,total_gold,purchase_date_gold,source_id,source_url,license
    """
    out: Dict[str, Dict] = {}
    for name in zf.namelist():
        if name.split("/")[-1].lower() == "labels.csv":
            with zf.open(name) as f:
                text = f.read().decode("utf-8", errors="ignore")
                rdr = csv.DictReader(io.StringIO(text))
                for row in rdr:
                    fname = (row.get("filename") or "").strip()
                    if not fname:
                        continue
                    rec = {
                        "store_name_gold": (row.get("store_name_gold") or "").strip() or None,
                        "total_gold": None,
                        "purchase_date_gold": (row.get("purchase_date_gold") or "").strip() or None,
                        "source_id": (row.get("source_id") or "").strip() or None,
                        "source_url": (row.get("source_url") or "").strip() or None,
                        "license": (row.get("license") or "").strip() or None,
                    }
                    try:
                        if row.get("total_gold") not in (None, "", "null"):
                            rec["total_gold"] = float(row["total_gold"])
                    except Exception:
                        rec["total_gold"] = None
                    out[fname] = rec
            break
    return out

def _current_count(table: str) -> int:
    try:
        res = supabase.table(table).select("id", count="exact").execute()
        return res.count or 0
    except Exception:
        return 0

def _ingest_zip(dataset_key: str, zbytes: bytes, status_slot) -> Tuple[int, int, str]:
    """
    Ingest a zip for a dataset.
    Returns: (inserted_rows, skipped_rows, batch_id)
    """
    if not _ensure_table_exists(dataset_key):
        return (0, 0, "")

    table = _table_name(dataset_key)
    already = _current_count(table)
    remaining = max(0, MAX_ROWS_PER_DATASET - already)
    if remaining == 0:
        st.warning(f"Row limit reached ({MAX_ROWS_PER_DATASET}) for `{table}`. No rows ingested.")
        return (0, 0, "")

    batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = f"{TABLE_PREFIX}{dataset_key}_{batch_id}"

    zf = zipfile.ZipFile(io.BytesIO(zbytes))
    labels = _parse_labels_csv(zf)

    # Count image files for progress
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
    image_names = [n for n in zf.namelist() if n.split("/")[-1].lower().endswith(IMAGE_EXTS)]
    total_imgs = len(image_names)
    if total_imgs == 0:
        st.warning("No images found in zip.")
        return (0, 0, "")

    progress = status_slot.progress(0, text="Starting upload…")
    inserted = 0
    skipped = 0
    to_insert = []

    for idx, name in enumerate(image_names, start=1):
        if len(to_insert) >= remaining:
            break
        base = name.split("/")[-1]
        img_bytes = zf.read(name)
        checksum = _sha256(img_bytes)
        storage_path = f"{prefix}/{base}"

        # Upload file
        ok = _storage_upload(storage_path, img_bytes)
        if not ok:
            skipped += 1
        else:
            lab = labels.get(base) or {}
            row = {
                "dataset": dataset_key,
                "batch_id": batch_id,
                "source_id": lab.get("source_id") or base,
                "checksum": checksum,
                "image_storage_path": storage_path,
                "store_name_gold": lab.get("store_name_gold"),
                "total_gold": lab.get("total_gold"),
                "purchase_date_gold": lab.get("purchase_date_gold"),
                "items_gold": None,
                "source_url": lab.get("source_url"),
                "license": lab.get("license"),
            }
            to_insert.append(row)

        # update progress
        pct = int((idx / max(total_imgs, 1)) * 100)
        progress.progress(min(pct, 100), text=f"Uploading… {idx}/{total_imgs}")

    # Insert in chunks
    CHUNK = 500
    for i in range(0, len(to_insert), CHUNK):
        chunk = to_insert[i:i+CHUNK]
        try:
            supabase.table(table).insert(chunk).execute()
            inserted += len(chunk)
        except Exception:
            for r in chunk:
                try:
                    supabase.table(table).insert(r).execute()
                    inserted += 1
                except Exception:
                    skipped += 1

    progress.progress(100, text="Done.")
    return (inserted, skipped, batch_id)

def _delete_dataset(dataset_key: str, scope: str) -> Tuple[int, int]:
    """
    Deletes DB rows and storage objects.
    scope: 'all' or 'last'
    Returns: (rows_deleted, objects_deleted)
    """
    table = _table_name(dataset_key)
    rows = supabase.table(table).select("batch_id, image_storage_path").execute().data or []
    if not rows:
        return (0, 0)

    objects_deleted = 0
    if scope == "last":
        last_batch = max(r["batch_id"] for r in rows if r.get("batch_id"))
        batch_rows = [r for r in rows if r.get("batch_id") == last_batch]
        supabase.table(table).delete().eq("batch_id", last_batch).execute()
        for r in batch_rows:
            try:
                supabase.storage.from_(BUCKET).remove([r["image_storage_path"]])
                objects_deleted += 1
            except Exception:
                pass
        return (len(batch_rows), objects_deleted)

    # scope == "all"
    supabase.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    prefixes = set([r["image_storage_path"].split("/")[0] for r in rows if r.get("image_storage_path")])
    for folder in list(prefixes):
        try:
            listing = supabase.storage.from_(BUCKET).list(path=folder)
            paths = [f"{folder}/{obj['name']}" for obj in listing]
            if paths:
                supabase.storage.from_(BUCKET).remove(paths)
                objects_deleted += len(paths)
        except Exception:
            pass

    return (len(rows), objects_deleted)

def _verify_dataset(dataset_key: str, sample_n: int = 5) -> Dict:
    table = _table_name(dataset_key)
    info = {"count": 0, "ok": 0, "fail": 0}
    try:
        res = supabase.table(table).select("id", count="exact").execute()
        info["count"] = res.count or 0
    except Exception as e:
        st.error(f"Count failed: {e}")
        return info

    rows = supabase.table(table).select("image_storage_path").limit(sample_n).execute().data or []
    for r in rows:
        url = _storage_signed_url(r["image_storage_path"], 120)
        if url:
            info["ok"] += 1
        else:
            info["fail"] += 1
    return info

def _view_dataset(dataset_key: str, limit: int = 24):
    table = _table_name(dataset_key)
    rows = supabase.table(table).select(
        "dataset,batch_id,source_id,image_storage_path,store_name_gold,total_gold,purchase_date_gold,ingested_at"
    ).order("ingested_at", desc=True).limit(limit).execute().data or []
    return rows

# ------------------------------------------------------------
# UI — Only 2 quadrants (SROIE & ExpressExpense)
# ------------------------------------------------------------
def render_quadrant(container, ds: Dict):
    with container:
        st.markdown(
            f"### {ds['label']}  \n"
            f"<small>{ds['blurb']} • license: {ds['license']}</small>",
            unsafe_allow_html=True
        )
        st.caption(f"Table: `{_table_name(ds['key'])}` — Bucket: `{BUCKET}`")
        st.divider()

        controls = st.columns([1,1,1,1])

        zip_file = st.file_uploader(
            f"Upload .zip for {ds['label']} (images + optional labels.csv)",
            type=["zip"],
            key=f"zip_{ds['key']}"
        )

        # Upload (with live status)
        with controls[0]:
            if st.button("Upload", key=f"btn_upload_{ds['key']}", use_container_width=True):
                if not zip_file:
                    st.warning("Choose a .zip file first.")
                else:
                    with st.status("Ingesting…", expanded=True) as status:
                        prog = st.empty()
                        inserted, skipped, batch_id = _ingest_zip(ds["key"], zip_file.read(), prog)
                        st.write(f"Batch: `{batch_id or '—'}`")
                        st.write(f"Inserted: **{inserted}**, Skipped: **{skipped}**")
                        status.update(label="Ingest complete", state="complete")

        # Delete last batch
        with controls[1]:
            if st.button("Delete Last Batch", key=f"btn_del_last_{ds['key']}", use_container_width=True):
                nrows, nobjs = _delete_dataset(ds["key"], scope="last")
                st.warning(f"Deleted last batch • Rows: {nrows} • Objects: {nobjs}")

        # Delete all
        with controls[2]:
            if st.button("Delete ALL", key=f"btn_del_all_{ds['key']}", use_container_width=True):
                nrows, nobjs = _delete_dataset(ds["key"], scope="all")
                st.error(f"Deleted ALL • Rows: {nrows} • Objects: {nobjs}")

        # Verify
        with controls[3]:
            if st.button("Verify", key=f"btn_verify_{ds['key']}", use_container_width=True):
                info = _verify_dataset(ds["key"])
                st.info(f"Rows: {info['count']} • Signed URLs OK: {info['ok']} • Fail: {info['fail']}")

        # View grid
        if st.button("View Latest", key=f"btn_view_{ds['key']}"):
            rows = _view_dataset(ds["key"], limit=24)
            if not rows:
                st.info("No rows to show yet.")
            else:
                ncols = 4
                cols = st.columns(ncols)
                for i, r in enumerate(rows):
                    col = cols[i % ncols]
                    with col:
                        url = _storage_signed_url(r["image_storage_path"], 300)
                        if url:
                            st.image(url, use_container_width=True)
                        st.caption(
                            f"`{r.get('source_id')}`  \n"
                            f"Store: {r.get('store_name_gold') or '—'} • "
                            f"Total: {r.get('total_gold') if r.get('total_gold') is not None else '—'}  \n"
                            f"Date: {r.get('purchase_date_gold') or '—'}  \n"
                            f"Batch: `{r.get('batch_id')}`  \n"
                            f"Path: `{r.get('image_storage_path')}`  \n"
                            f"At: {r.get('ingested_at')}"
                        )

# One row, two quadrants
row = st.columns(2, gap="large")
render_quadrant(row[0], DATASETS[0])  # SROIE
render_quadrant(row[1], DATASETS[1])  # ExpressExpense

st.markdown("---")
st.caption(
    "Tip: Include a `labels.csv` in your .zip to pre-fill gold fields. "
    "Format: filename,store_name_gold,total_gold,purchase_date_gold,source_id,source_url,license. "
    f"Each dataset table is capped at {MAX_ROWS_PER_DATASET} rows to stay within the free tier."
)
