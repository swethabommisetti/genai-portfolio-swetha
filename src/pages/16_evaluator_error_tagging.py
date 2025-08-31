import streamlit as st
from utils.evals_repo import (
    list_recent_receipts,
    save_error_tags,
    receipt_image_url,   # 👈 add this
)

st.title("🏷️ Evaluator — Error Tagging")

rows = list_recent_receipts(limit=25)
if not rows:
    st.info("No receipts yet. Run the extractor first.")
    st.stop()

labels = [f"#{r['id']} • {r.get('store_name') or '—'} • ${r.get('total') or '—'}" for r in rows]
idx = st.selectbox("Pick a receipt to tag errors", list(range(len(rows))), format_func=lambda i: labels[i])
rec = rows[idx]
rid = rec.get("receipt_file_id")
st.caption(f"receipt_file_id: `{rid}`")

# 👇 NEW: visual context
with st.container(border=True):
    st.subheader("Receipt image")
    url = receipt_image_url(rid)
    if url:
        st.image(url, use_container_width=True)
    else:
        st.info("Could not generate a signed URL for this image.")
st.caption(f"receipt_file_id: `{rid}`")

TAGS = [
    "Missed item(s)",
    "Hallucinated item(s)",
    "Wrong total",
    "Wrong store name",
    "Wrong date",
    "OCR garble / unreadable",
    "Currency/format issue",
    "Duplicated lines",
]

choices = st.multiselect("Select error types", TAGS)
comment = st.text_area("Notes (optional)")

if st.button("Save tags"):
    res = save_error_tags(
        receipt_file_id=rid,
        tags=choices,
        comment=comment,
        evaluator_email=st.session_state.get("user_email"),
    )
    if res.get("ok"):
        st.success("Saved tags.")
    else:
        st.error(f"Save failed: {res}")
