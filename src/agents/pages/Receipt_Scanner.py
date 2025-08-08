import streamlit as st
from utils.tracking import log_page_visit
from supabase import create_client
from datetime import datetime
import os
import io
import re

# ---------------------------------
# 1. Track page visit
# ---------------------------------
log_page_visit("Receipt Scanner")

st.title("📸 Receipt Scanner Agent")
st.write("Upload a receipt image to get suggestions based on purchase history.")

# ---------------------------------
# 2. Supabase client setup
# ---------------------------------
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

# ---------------------------------
# 3. File Upload UI
# ---------------------------------
uploaded = st.file_uploader("Upload your receipt", type=["jpg", "jpeg", "png"])

if uploaded:
    # Step 4. Show preview
    st.image(uploaded, caption="🖼️ Preview: Uploaded Receipt", use_column_width=True)

    # Step 5. Clean & build filename
    user_email_raw = st.session_state.get("user_email", "anonymous")
    user_email_clean = user_email_raw.replace("@", "_at_").replace(".", "_dot_")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    original_filename_raw = uploaded.name if uploaded.name else "receipt"
    original_name_no_ext = re.sub(r'\.[^.]+$', '', original_filename_raw)  # remove extension
    original_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name_no_ext).lower()

    filename = f"{original_name_clean}_{user_email_clean}_{timestamp}.jpg"
    bucket_name = "genai-analytics-bucket"
    mime_type = uploaded.type
    visitor_id = st.session_state.get("visitor_id")

    # Step 6. Upload to Supabase Storage
    file_bytes = uploaded.read()
    upload_res = supabase.storage.from_(bucket_name).upload(
        path=filename,
        file=io.BytesIO(file_bytes),
        file_options={
            "content-type": mime_type,
            "x-upsert": "false"
        }
    )

    # Step 7. Insert metadata into receipt_files table
    if upload_res.status_code == 200:
        st.success("✅ File uploaded to Supabase!")

        supabase.table("receipt_files").insert({
            "filename": filename,
            "original_filename": original_filename_raw,
            "bucket_name": bucket_name,
            "mime_type": mime_type,
            "visitor_id": visitor_id
        }).execute()

        # Step 8. Generate signed URL (10 min expiry)
        signed_url = supabase.storage.from_(bucket_name).create_signed_url(
            path=filename,
            expires_in=600
        )
        st.image(signed_url, caption="🔐 Secure Preview (valid 10 minutes)", use_column_width=True)

    else:
        st.error("❌ Upload failed. Please check file format or Supabase setup.")
