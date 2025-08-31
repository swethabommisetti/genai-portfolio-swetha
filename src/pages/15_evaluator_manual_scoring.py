# src/pages/15_evaluator_manual_scoring.py
import streamlit as st
import utils.evals_repo as evals  # namespaced import avoids shadowing

st.title("📝 Evaluator — Manual Scoring")

# ---- Load recent receipts -----------------------------------------------------
rows = evals.list_recent_receipts(limit=25)
if not rows:
    st.info("No receipts yet. Run the extractor first.")
    st.stop()

# Build stable option tuples: (dtl_id, receipt_file_id, label)
options = []
for r in rows:
    label = f"#{r.get('id')} • {r.get('store_name') or '—'} • ${r.get('total') or '—'}"
    options.append((r.get("id"), r.get("receipt_file_id"), label))

# ---- Select a receipt (returns the tuple) ------------------------------------
choice = st.selectbox(
    "Pick a receipt to score",
    options,
    format_func=lambda t: t[2],
    key="eval_receipt_choice",
)
dtl_id, rid, _ = choice  # exact ids selected

# ---- Two-column review: image | extracted values ------------------------------
left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("Receipt image")
    url = evals.receipt_image_url(rid)
    if url:
        st.image(url, use_container_width=True)
    else:
        st.info("Could not generate a signed URL for this image.")

with right:
    st.subheader("Extracted values (model output)")
    data = evals.get_extracted_values(receipt_file_id=rid, dtl_id=dtl_id)
    hdr = data.get("header") or {}
    items = data.get("items") or []
    totals = data.get("totals") or {}

    # Header summary (fallback to derived total)
    derived_total = totals.get("grand_total")
    total_val = hdr.get("total")
    currency = hdr.get("currency", "")
    total_txt = (
        f"{total_val} {currency}".strip()
        if total_val
        else (f"{derived_total} {currency}".strip() if derived_total is not None else "None")
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Store:** {hdr.get('store_name') or 'None'}")
        st.markdown(f"**Total:** {total_txt}")
    with c2:
        st.markdown(f"**Date:** {hdr.get('purchase_datetime') or 'None'}")
        st.markdown(f"**receipt_file_id:** `{rid}`")

    if items:
        # Pretty display
        def fmt(v):
            if isinstance(v, (int, float)):
                return f"{v:,.2f}"
            return v if v is not None else "—"

        view = [
            {
                "Line": it.get("line_no"),
                "Item": it.get("name"),
                "Qty": it.get("qty"),
                "Unit": fmt(it.get("unit_price")),
                "Line Total": fmt(it.get("line_total")),
                "Store": it.get("store"),
                "Address": it.get("address"),
            }
            for it in items
        ]
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.caption("No line items found for this receipt.")

st.divider()

# ---- Scoring form --------------------------------------------------------------
with st.form("score_form", clear_on_submit=True):
    st.subheader("Field-level correctness")
    c1, c2, c3 = st.columns(3)
    with c1:
        store_ok = st.selectbox("Store name correct?", ["—", "Yes", "No"], key="store_ok")
    with c2:
        total_ok = st.selectbox("Total correct?", ["—", "Yes", "No"], key="total_ok")
    with c3:
        date_ok  = st.selectbox("Date correct?",  ["—", "Yes", "No"], key="date_ok")

    overall = st.slider(
        "Overall quality score (1 bad – 5 great)",
        min_value=1,
        max_value=5,
        value=3,
        key="overall_score",
    )
    comment = st.text_area("Comments (what was wrong / how to improve)", key="eval_comment")

    submitted = st.form_submit_button("Save evaluation")
    if submitted:
        def to_bool(val: str):
            return None if val == "—" else (val == "Yes")

        res = evals.save_manual_scores(
            receipt_file_id=rid,
            store_correct=to_bool(store_ok),
            total_correct=to_bool(total_ok),
            date_correct=to_bool(date_ok),
            overall_score=overall,
            comment=comment,
            evaluator_email=st.session_state.get("user_email"),
        )
        if res.get("ok"):
            st.success("Saved. Thank you!")
        else:
            st.error(f"Save failed: {res}")
