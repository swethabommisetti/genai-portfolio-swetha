import streamlit as st

st.set_page_config(page_title="ReceiptScanner", layout="wide")

st.title("Receipt Scanner Agent")
st.caption("Digitize Day to Day Receipts")

# CTA → go to pages/1_receiptscanner_run.py
if st.button("Run the Scan", use_container_width=False):
    st.switch_page("pages/1_receiptscanner_run.py")


# (Your cards/sections)
left, right = st.columns(2)
with left:
    st.subheader("Problem Statement")
    st.markdown(
        "- Digitizing Receipts Across Vendors when using different credit cards is tough after mint.com is not available anymore \n"
        "- As a homemaker have to make sure I don't Purchase Same Item Multiple times\n"


    )
with right:
    st.subheader("What This Tool Solves")
    st.markdown(
        "- Allows me to learn Agentic AI with Evaluations \n"
        "- Solves my real world problem"
    )

c1, c2, c3,c4  = st.columns(4)
if c1.button("Upload"):
    st.switch_page("pages/1_receiptscanner_run.py")
if c2.button("Extract"):
    st.switch_page("pages/12_receiptscanner_documentation.py")  # if your extract UI is here, otherwise point to the actual extract page
if c3.button("Analytics"):
    st.switch_page("pages/13_receiptscanner_analytics.py")
if c4.button("Chat"):
    st.switch_page("pages/10_receiptscanner_chat.py")