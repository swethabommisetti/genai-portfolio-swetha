import streamlit as st
from streamlit_analytics import start_tracking

st.set_page_config(page_title="Swetha's GenAI Portfolio", layout="centered")

# ✅ Start tracking (no `with`)
start_tracking()

# 🎯 Your UI
st.title("👋 Welcome to Swetha's GenAI Agent Suite")
st.markdown("""
Use the sidebar to access the available GenAI agents.

- 📸 **Receipt Scanner** — See what you've already purchased  
- 📚 **Book Recommender** — Find books for your kids based on themes  
- This is Awesome Job
""")