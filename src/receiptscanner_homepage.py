import streamlit as st

st.set_page_config(page_title="Receipt Scanner Agent", layout="wide")

# ---- Title and tagline ----
st.title("🧾 Receipt Scanner Agent")
st.caption("Digitize and validate receipts with GenAI + Human-in-the-Loop evaluations.")

# ---- Short intro (adds recruiter context) ----
st.markdown(
    """
This agent extracts structured purchase data from scanned receipts across vendors using **OCR and LangGraph pipelines**.
It was first built to solve a real household need after Mint.com’s shutdown — and later evolved into a
**production-style GenAI evaluation project** to measure accuracy, latency, and consistency.
    """
)

# ---- Primary CTA ----
if st.button("▶️ Run the Scan", use_container_width=False):
    st.switch_page("pages/1_receiptscanner_run.py")

st.divider()

# ---- Problem and Solution sections (vertical layout, easier to read) ----
st.subheader("💡 Problem Statement")
st.markdown(
    """
- After **Mint.com** was discontinued, tracking expenses across multiple credit cards became tedious.
- Needed a simple way to **digitize household receipts** and prevent duplicate purchases.
    """
)

st.subheader("🚀 What This Tool Solves")
st.markdown(
    """
- Automates receipt digitization and validation across vendors using **GenAI and Supabase**.
- Demonstrates **agentic workflows** with evaluation loops for accuracy, latency, and robustness.
- Bridges a real-world household need with **enterprise-quality AI evaluation design**.
    """
)

st.divider()

# ---- Action buttons ----
st.markdown("### 🔍 Explore Tool Sections")
cols = st.columns(4)

with cols[0]:
    if st.button("📤 Upload"):
        st.switch_page("pages/1_receiptscanner_run.py")

with cols[1]:
    if st.button("⚙️ Extract"):
        st.switch_page("pages/12_receiptscanner_documentation.py")  # update if your Extract UI is elsewhere

with cols[2]:
    if st.button("📊 Analytics"):
        st.switch_page("pages/13_receiptscanner_analytics.py")

with cols[3]:
    if st.button("💬 Chat"):
        st.switch_page("pages/10_receiptscanner_chat.py")

st.divider()

# ---- Optional: small summary footer ----
st.caption(
    "Built with Streamlit • Evaluated using LangSmith • Data stored in Supabase • OCR powered by Mistral/LangGraph"
)
