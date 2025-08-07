import streamlit as st
from utils.tracking import log_page_visit

st.set_page_config(page_title="Swetha's GenAI Portfolio", layout="centered")

# Track home page visit
log_page_visit("Home Page")

st.title("Welcome to Swetha's GenAI Portfolio 👋")

st.markdown("""
Welcome to my Journey 

Use the sidebar to access the available GenAI agents.

- 📸 **Receipt Scanner** — See what you've already purchased  
- 📚 **Book Recommender** — Find books for your kids based on themes  

""")

