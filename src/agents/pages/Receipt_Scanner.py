import streamlit as st
from datetime import datetime
import re

from utils.supabase_utils import get_supabase_client
from utils.tracking import log_once_per_page


def run_receipt_scanner():
    # ---------------------------------
    # 1. Track visit once per session
    # ---------------------------------
    if "user_email" in st.session_state and st.session_state["user_email"]:
        log_once_per_page("Receipt Scanner")

    st.title("📸 Receipt Scanner Agent")
    st.write("Upload a receipt image to get suggestions based on purchase history.")

    # ---------------------------------
    # 2. Supabase client
    # ---------------------------------
    supabase = get_supabase_client()

    # ---------------------------------
    # 3. File Upload UI
    # ---------------------------------
    uploaded = st.file_uploader("Upload your receipt", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, caption="🖼️ Preview: Uploaded Receipt", use_column_width=True)

        # ---------------------------------
        # 4. Generate safe filename
        # ---------------------------------
        user_email_raw = st.session_state.get("user_email", "anonymous")
        user_email_clean = user_email_raw.replace("@", "_at_").replace(".", "_dot_")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        original_filename_raw = uploaded.name or "receipt"
        original_name_no_ext = re.sub(r'\.[^.]+$', '', original_filename_raw)
        original_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name_no_ext).lower()

        filename = f"{original_name_clean}_{user_email_clean}_{timestamp}.jpg"
        bucket_name = "genai-analytics-bucket"
        mime_type = uploaded.type
        visitor_id = st.session_state.get("visitor_id")

        # ---------------------------------
        # 5. Upload to Supabase bucket
        # ---------------------------------
        file_bytes = uploaded.read()
        upload_res = supabase.storage.from_(bucket_name).upload(
            path=filename,
            file=file_bytes,
            file_options={
                "content-type": mime_type,
                "x-upsert": "false"
            }
        )

        if isinstance(upload_res, dict) and upload_res.get("error") is None:
            st.success("✅ File uploaded to Supabase!")

            # ---------------------------------
            # 6. Insert metadata into table (with debug)
            # ---------------------------------
            try:
                supabase.table("receipt_files").insert({
                    "filename": filename,
                    "original_filename": original_filename_raw,
                    "bucket_name": bucket_name,
                    "mime_type": mime_type,
                    "visitor_id": visitor_id
                }).execute()
            except Exception as e:
                st.warning(f"⚠️ Could not save receipt metadata: {e}")

            # ---------------------------------
            # 7. Display signed URL
            # ---------------------------------
            signed_url = supabase.storage.from_(bucket_name).create_signed_url(
                path=filename,
                expires_in=600  # 10 minutes
            )

            st.image(signed_url, caption="🔐 Secure Preview (valid for 10 min)", use_column_width=True)
        else:
            error_msg = upload_res.get("error", {}).get("message", "Unknown error")
            st.error(f"❌ Upload failed: {error_msg}")
