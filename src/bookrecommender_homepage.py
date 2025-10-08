import streamlit as st

st.set_page_config(page_title="BookRecommender", layout="wide")

st.title("Book Recommender Agent")
st.caption("")

# CTA → go to pages/1_provision.py
if st.button("RUN the Scan", use_container_width=False):
    st.switch_page("pages/2_bookrecommender_run.py")


# (Your cards/sections)
left, right = st.columns(2)
with left:
    st.subheader("Problem Statement")
    st.markdown(
        "- Suggest New Books\n"

    )
with right:
    st.subheader("What This Tool Solves")
    st.markdown(
        "- One-click agent to pick a book that we have NOT Picked before- WIP \n"

    )



