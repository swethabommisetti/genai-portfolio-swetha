import streamlit as st
from datetime import datetime
import io
import re

from utils.tracking import log_page_visit
from utils.supabase_utils import get_supabase_client

# ---------------------------------
# 1. Track page visit
# ---------------------------------
log_page_visit("Receipt Scanner")

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
    # 4. Filename generation
    # ---------------------------------
    user_email_raw = st.session_state.get("user_email", "anonymous")
    user_email_clean = user_email_raw.replace("@", "_at_").replace(".", "_dot_")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    original_filename_raw = uploaded.name if uploaded.name else "receipt"
    original_name_no_ext = re.sub(r'\.[^.]+$', '', original_filename_raw)  # remove file extension
    original_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name_no_ext).lower()

    filename = f"{original_name_clean}_{user_email_clean}_{timestamp}.jpg"
    bucket_name = "genai-analytics-bucket"
    mime_type = uploaded.type
    visitor_id = st.session_state.get("visitor_id")

    # ---------------------------------
    # 5. Upload to Supabase Storage
    # ---------------------------------
    file_bytes = uploaded.read()
        upload_res = supabase.storage.from_(bucket_name).upload(
        path=filename,
        file=io.BytesIO(file_bytes),
        file_options={
            "content-type": mime_type,
            "x-upsert": "false"  # prevent overwriting
        }
    )

    if upload_res.status_code == 200:
        st.success("✅ File uploaded to Supabase!")

        # ---------------------------------
        # 6. Insert metadata into receipt_files table
        # ---------------------------------
        supabase.table("receipt_files").insert({
            "filename": filename,
            "original_filename": original_filename_raw,
            "bucket_name": bucket_name,
            "mime_type": mime_type,
            "visitor_id": visitor_id
        }).execute()

        # ---------------------------------
        # 7. Generate signed URL for preview
        # ---------------------------------
        signed_url = supabase.storage.from_(bucket_name).create_signed_url(
            path=filename,
            expires_in=600  # 10 minutes
        )
        st.image(signed_url, caption="🔐 Secure Preview (valid for 10 min)", use_column_width=True)

    else:
        st.error("❌ Upload failed. Check Supabase Storage permissions or filename format.")
