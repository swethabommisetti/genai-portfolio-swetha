# src/pages/13_receiptscanner_analytics.py
import streamlit as st
from collections import Counter

from utils.supabase_utils import get_supabase_client
from utils.evals_repo import metrics_summary  # from the evaluator feature

st.title("📊 ReceiptScanner — Analytics")

# --- Top toggle (button bar) ---
try:
    mode = st.segmented_control("View", ["Usage", "Evaluation"], default="Usage")  # Streamlit ≥1.38
except Exception:
    # Fallback for older versions
    mode = st.radio("View", ["Usage", "Evaluation"], horizontal=True, label_visibility="collapsed")

sb = get_supabase_client()

# -------------------------
# USAGE ANALYTICS (agent usage, data produced)
# -------------------------
if mode == "Usage":
    col1, col2, col3 = st.columns(3)
    try:
        total_receipts = sb.table("receipts_dtl").select("id", count="exact").execute().count or 0
    except Exception:
        total_receipts = "—"

    try:
        # how many items extracted across all receipts (if you have receipt_items)
        total_items = sb.table("receipt_items").select("id", count="exact").execute().count or 0
    except Exception:
        total_items = "—"

    try:
        recent_files = sb.table("receipt_files").select("id", count="exact").execute().count or 0
    except Exception:
        recent_files = "—"

    col1.metric("Receipts processed", total_receipts)
    col2.metric("Line items extracted", total_items)
    col3.metric("Files stored", recent_files)

    st.subheader("Latest receipts (10)")
    try:
        latest = (
            sb.table("receipts_dtl")
            .select("id, receipt_file_id, store_name, total, purchase_datetime")
            .order("id", desc=True)
            .limit(10)
            .execute()
        )
        rows = getattr(latest, "data", []) or []
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No receipts yet.")
    except Exception as e:
        st.warning(f"Could not load latest receipts: {e}")

    st.subheader("Top stores (quick view)")
    try:
        recs = (
            sb.table("receipts_dtl")
            .select("store_name")
            .order("id", desc=True)
            .limit(1000)   # sample cap
            .execute()
        )
        names = [r.get("store_name") or "—" for r in getattr(recs, "data", []) or []]
        counts = Counter(names)
        if counts:
            st.bar_chart({"count": list(counts.values())}, x=list(counts.keys()))
        else:
            st.info("No data to aggregate yet.")
    except Exception:
        st.info("Unable to load top stores.")

# -------------------------
# EVALUATION ANALYTICS (model quality, human scoring)
# -------------------------
else:
    st.caption("Evaluator metrics come from the **evaluations** and **evaluation_errors** tables you log via the Evaluator pages.")
    m = metrics_summary()

    col1, col2, col3 = st.columns(3)
    col1.metric("Receipts", m.get("counts", {}).get("receipts", "—"))
    col2.metric("Evaluations", m.get("counts", {}).get("evaluations", "—"))
    col3.metric("Error tags", m.get("counts", {}).get("errors", "—"))

    st.subheader("Accuracy (simple)")
    acc = m.get("accuracy", {})
    if acc:
        st.write(acc)
    else:
        st.info("No evaluations yet. Use **Evaluator → Manual Scoring** to add some.")

    st.subheader("Top error types")
    try:
        data = sb.table("evaluation_errors").select("tag").execute()
        tags = [d["tag"] for d in getattr(data, "data", []) or []]
        cnt = Counter(tags)
        if cnt:
            st.bar_chart({"count": list(cnt.values())}, x=list(cnt.keys()))
        else:
            st.info("No error tags yet. Use **Evaluator → Error Tagging**.")
    except Exception:
        st.info("Unable to load error tags right now.")
