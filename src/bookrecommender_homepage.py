import streamlit as st

st.set_page_config(page_title="BookRecommender", layout="wide")

st.title("BookRecommender Agent")
st.caption("??")

# CTA → go to pages/1_provision.py
if st.button("Run the Scan", use_container_width=False):
    st.switch_page("pages/2_bookrecommender_run.py")


# (Your cards/sections)
left, right = st.columns(2)
with left:
    st.subheader("Problem Statement")
    st.markdown(
        "- Not to Pick the Same Book We Already Read/Checked Out Previously\n"

    )
with right:
    st.subheader("What This Tool Solves")
    st.markdown(
        "- One-click agent to pick a book that we have NOT Picked before \n"

    )



