import streamlit as st
from datetime import datetime
import re

from utils.receipts_repo import (
    upsert_receipts_dtl,
    ensure_receipt_file_id,
    save_receipt_and_items,
)
from utils.supabase_utils import get_supabase_client
from utils.tracking import log_once_per_page
from utils.email_utils import prompt_for_optional_email

# IMPORTANT: do NOT import your agent at module import time — that caused the crash
# from agents.receipt_extractor.agent import extract_receipt_values


def run_receipt_scanner():
    # ---------------------------------
    # 0. Optional email capture
    # ---------------------------------
    st.title("📸 Receipt Scanner Agent")
    st.write("Upload a receipt image to get suggestions based on purchase history.")

    # Email input + submit button (added)
    col1, col2 = st.columns([3, 1])
    with col1:
        email_value = st.text_input(
            "Email (optional)",
            value=st.session_state.get("user_email", ""),
            placeholder="you@example.com",
        )
    with col2:
        if st.button("Submit Email"):
            if email_value and "@" in email_value:
                st.session_state["user_email"] = email_value.strip()
                st.success("Email saved for this session.")
            elif email_value:
                st.warning("Please enter a valid email.")

    # ---------------------------------
    # 1. Track visit once per session
    # ---------------------------------
    # log_once_per_page("Receipt Scanner")

    # ---------------------------------
    # 2. Supabase client
    # ---------------------------------
    supabase = get_supabase_client()

    # ---------------------------------
    # 3. File Upload UI
    # ---------------------------------
    uploaded = st.file_uploader("Upload your receipt", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = uploaded.getvalue()
        st.image(file_bytes, caption="🖼️ Preview: Uploaded Receipt", use_column_width=True)

        # ---------------------------------
        # 4. Generate safe filename
        # ---------------------------------
        user_email_raw = st.session_state.get("user_email", "anonymous")
        user_email_clean = user_email_raw.replace("@", "_at_").replace(".", "_dot_")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        original_filename_raw = uploaded.name or "receipt"
        original_name_no_ext = re.sub(r"\.[^.]+$", "", original_filename_raw)
        original_name_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", original_name_no_ext).lower()

        filename = f"{original_name_clean}_{user_email_clean}_{timestamp}.jpg"
        bucket_name = "genai-analytics-bucket"
        mime_type = uploaded.type
        visitor_id = st.session_state.get("visitor_id")

        # ---------------------------------
        # 5. Upload to Supabase bucket
        # ---------------------------------
        upload_res = supabase.storage.from_(bucket_name).upload(
            path=filename,
            file=file_bytes,
            file_options={"content-type": mime_type, "x-upsert": "false"},
        )

        if not hasattr(upload_res, "error"):
            st.success("✅ File uploaded to Supabase!")

            # ---------------------------------
            # 6. Insert metadata into table (capture id)
            # ---------------------------------
            receipt_file_id = None
            try:
                insert_res = (
                    supabase.table("receipt_files")
                    .insert(
                        {
                            "filename": filename,
                            "original_filename": original_filename_raw,
                            "bucket_name": bucket_name,
                            "mime_type": mime_type,
                            "visitor_id": visitor_id,
                        }
                    )
                    .execute()
                )

                if getattr(insert_res, "data", None):
                    receipt_file_id = insert_res.data[0].get("id")
            except Exception as e:
                st.warning(f"⚠️ Could not save receipt metadata: {e}")

            # Fallback lookup by filename if id not returned
            if receipt_file_id is None:
                try:
                    sel = (
                        supabase.table("receipt_files")
                        .select("id")
                        .eq("filename", filename)
                        .limit(1)
                        .execute()
                    )
                    if getattr(sel, "data", None):
                        receipt_file_id = sel.data[0]["id"]
                except Exception:
                    pass

            # ---------------------------------
            # 7. Create signed URL and extract the URL string
            # ---------------------------------
            signed_url = None
            try:
                signed_res = supabase.storage.from_(bucket_name).create_signed_url(
                    path=filename, expires_in=600  # 10 minutes
                )
                if isinstance(signed_res, dict):
                    signed_url = (
                        signed_res.get("signedURL")
                        or signed_res.get("signed_url")
                        or signed_res.get("url")
                    )
                else:
                    signed_url = (
                        getattr(signed_res, "signedURL", None)
                        or getattr(signed_res, "signed_url", None)
                        or getattr(signed_res, "url", None)
                    )
            except Exception as e:
                st.warning(f"⚠️ Could not create signed URL: {e}")

            # ---------------------------------
            # 8. Extract Values button (calls your agent)
            # ---------------------------------
            st.divider()
            if st.button("🔎 Extract Values", type="primary"):
                with st.spinner("Running receipt extractor…"):
                    # LAZY IMPORT happens here — avoids the startup import crash
                    try:
                        from agents.receipt_extractor.agent import (
                            extract_receipt_values,
                        )
                    except Exception as imp_err:
                        st.error(
                            "Importing the extractor failed. "
                            "This usually means a dependency version mismatch "
                            "(pydantic / langchain / langchain_groq)."
                        )
                        st.exception(imp_err)
                        st.stop()

                    try:
                        result = extract_receipt_values(
                            file_url=signed_url,
                            file_bytes=None if signed_url else file_bytes,
                            filename=filename,
                            mime_type=mime_type,
                            user_email=st.session_state.get("user_email"),
                            visitor_id=visitor_id,
                            bucket=bucket_name,
                        )
                        st.success("✅ Extraction complete.")
                        st.json(
                            result
                            if result is not None
                            else {"status": "ok", "details": "No result payload returned."}
                        )

                        # === Persist to receipts_dtl ===
                        try:
                            rid = ensure_receipt_file_id(
                                supabase,
                                receipt_file_id=receipt_file_id,
                                filename=filename,
                            )
                            if rid is None:
                                st.warning(
                                    "⚠️ Could not determine receipt_file_id; skipping receipts_dtl insert."
                                )
                            else:
                                # upsert_receipts_dtl(...)  # if you still need it
                                save_receipt_and_items(
                                    supabase,
                                    result,
                                    receipt_file_id=rid,
                                    visitor_id=visitor_id,
                                )
                                st.success("🗂️ Receipt details saved to receipts_dtl.")
                        except Exception as db_e:
                            st.warning(f"⚠️ Could not save receipt details: {db_e}")

                    except Exception as run_err:
                        st.error("❌ Extraction failed.")
                        st.exception(run_err)

        else:
            error_msg = getattr(upload_res.error, "message", "Unknown error")
            st.error(f"❌ Upload failed: {error_msg}")


# Make the page render when Streamlit loads this module
run_receipt_scanner()
