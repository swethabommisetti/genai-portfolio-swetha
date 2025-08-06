import streamlit as st

st.title("📸 Receipt Scanner Agent")
st.write("Upload a receipt image to get suggestions based on purchase history.")
st.write("AwesomeJob")

uploaded = st.file_uploader("Upload your receipt", type=["jpg", "png", "jpeg"])
if uploaded:
    st.image(uploaded, caption="Uploaded Receipt", use_column_width=True)
    st.success("This is a test upload — real logic will follow.")