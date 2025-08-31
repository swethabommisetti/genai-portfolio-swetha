# src/pages/15_evaluator_manual_scoring.py
import streamlit as st
from datetime import datetime, date
from utils.supabase_utils import get_supabase_client
import utils.evals_repo as evals  # namespaced import avoids shadowing

st.title("📝 Evaluator — Manual Scoring")

supabase = get_supabase_client()

# ---------- SCD-2 GOLD HELPERS -------------------------------------------------
def ensure_current_gold(receipt_id: str, reviewer: str | None = None):
    """
    If no active SCD-2 gold row exists for this receipt, create a placeholder.
    Called when a user selects a receipt from the dropdown.
    """
    if not receipt_id:
        return
    existing = (
        supabase.table("receipts_gold")
        .select("id")
        .eq("receipt_id", receipt_id)
        .eq("curr_rec_ind", True)
        .limit(1)
        .execute()
        .data
    )
    if not existing:
        supabase.table("receipts_gold").insert({
            "receipt_id": receipt_id,
            "rec_eff_start": datetime.utcnow().isoformat(),
            "rec_eff_end": None,
            "curr_rec_ind": True,
            # placeholders until user confirms/promotes
            "store_name_gold": None,
            "total_gold": None,
            "purchase_date_gold": None,
            "reviewer": reviewer or st.session_state.get("user_email"),
        }).execute()

def upsert_gold_record(receipt_id: str, gold: dict, reviewer: str | None = None):
    """
    SCD-2 update: expire the current record (if any) and insert a new active one
    with the verified (gold) values.
    """
    now = datetime.utcnow().isoformat()

    # expire current
    supabase.table("receipts_gold").update({
        "rec_eff_end": now,
        "curr_rec_ind": False
    }).eq("receipt_id", receipt_id).eq("curr_rec_ind", True).execute()

    # insert new active
    row = {
        "receipt_id": receipt_id,
        "rec_eff_start": now,
        "rec_eff_end": None,
        "curr_rec_ind": True,
        "store_name_gold": gold.get("store_name_gold"),
        "total_gold": gold.get("total_gold"),
        "purchase_date_gold": gold.get("purchase_date_gold"),
        "reviewer": reviewer or st.session_state.get("user_email"),
    }
    supabase.table("receipts_gold").insert(row).execute()

# ---------- LOAD RECENT RECEIPTS -----------------------------------------------
rows = evals.list_recent_receipts(limit=25)
if not rows:
    st.info("No receipts yet. Run the extractor first.")
    st.stop()

# Build stable option tuples: (dtl_id, receipt_file_id, label)
options = []
for r in rows:
    label = f"#{r.get('id')} • {r.get('store_name') or '—'} • ${r.get('total') or '—'}"
    options.append((r.get("id"), r.get("receipt_file_id"), label))

# ---------- SELECT RECEIPT -----------------------------------------------------
choice = st.selectbox(
    "Pick a receipt to score",
    options,
    format_func=lambda t: t[2],
    key="eval_receipt_choice",
)
dtl_id, rid, _ = choice  # exact ids selected

# Ensure a current gold placeholder when a selection changes
if dtl_id and st.session_state.get("last_selected_receipt") != dtl_id:
    ensure_current_gold(dtl_id, reviewer=st.session_state.get("user_email"))
    st.session_state["last_selected_receipt"] = dtl_id
    st.toast("Gold snapshot created/ensured for this receipt.", icon="✅")

# ---------- TWO-COLUMN REVIEW: IMAGE | EXTRACTED VALUES ------------------------
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
        # show only date portion if datetime
        raw_dt = hdr.get("purchase_datetime")
        date_only = None
        if isinstance(raw_dt, str) and raw_dt:
            try:
                date_only = datetime.fromisoformat(raw_dt.replace("Z","")).date()
            except Exception:
                # try YYYY-MM-DD substring fallback
                date_only = raw_dt[:10]
        st.markdown(f"**Date:** {date_only or raw_dt or 'None'}")
        st.markdown(f"**receipt_file_id:** `{rid}`")

    if items:
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

# ---------- SCORING FORM -------------------------------------------------------
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

    st.markdown("---")
    promote_to_gold = st.checkbox("Also promote current model output to **Gold** (SCD-2)", value=False,
                                  help="This will version the receipts_gold record and set the current values as gold.")

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

        # If requested, promote the shown model output to Gold (SCD-2)
        if promote_to_gold:
            # choose total: explicit header total first, else derived grand_total
            chosen_total = None
            try:
                chosen_total = float(total_val) if total_val is not None else (
                    float(derived_total) if derived_total is not None else None
                )
            except Exception:
                chosen_total = None

            # normalize date to date object if possible
            gold_date = None
            raw_dt = hdr.get("purchase_datetime")
            if isinstance(raw_dt, str) and raw_dt:
                try:
                    gold_date = datetime.fromisoformat(raw_dt.replace("Z","")).date()
                except Exception:
                    # fallback: YYYY-MM-DD substring if it looks like a date
                    try:
                        gold_date = datetime.strptime(raw_dt[:10], "%Y-%m-%d").date()
                    except Exception:
                        gold_date = None

            upsert_gold_record(
                receipt_id=dtl_id,
                gold={
                    "store_name_gold": hdr.get("store_name"),
                    "total_gold": chosen_total,
                    "purchase_date_gold": gold_date.isoformat() if isinstance(gold_date, date) else None,
                },
                reviewer=st.session_state.get("user_email"),
            )

        if res.get("ok"):
            st.success("Saved. Thank you!")
        else:
            st.error(f"Save failed: {res}")
