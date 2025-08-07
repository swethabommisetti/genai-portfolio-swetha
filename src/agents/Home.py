# src/agents/Home.py

import streamlit as st
from streamlit_analytics import TrackPage

with TrackPage():
    st.title("📸 Receipt Scanner")
    
st.set_page_config(page_title="Swetha's GenAI Portfolio", layout="centered")

st.title("👋 Welcome to Swetha's GenAI Agent Suite")
st.markdown("""
Use the sidebar to access the available GenAI agents.

- 📸 **Receipt Scanner** — See what you've already purchased
- 📚 **Book Recommender** — Find books for your kids based on themes
- This is Awesome Job
""")
