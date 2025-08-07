import streamlit as st
from utils.tracking import log_page_visit

# Track receipt scanner page visit
log_page_visit("Receipt Scanner")

st.title("📸 Receipt Scanner Agent")
st.write("Upload a receipt image to get suggestions based on purchase history.")
st.write("AwesomeJob")

uploaded = st.file_uploader("Upload your receipt", type=["jpg", "png", "jpeg"])
if uploaded:
    st.image(uploaded, caption="Uploaded Receipt", use_column_width=True)
    st.success("This is a test upload — real logic will follow.")
