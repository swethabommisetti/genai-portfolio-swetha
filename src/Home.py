import streamlit as st
import os

from utils.email_utils import prompt_for_optional_email
from agents.pages.Receipt_Scanner import run_receipt_scanner
from utils.tracking import log_once_per_page

st.set_page_config(page_title="Swetha's GenAI Portfolio", layout="centered")

# Sidebar menu
page = st.sidebar.selectbox(
    "Choose a demo",
    ("Home", "Receipt Scanner", "Book Recommender"),
)

if page == "Receipt Scanner":
    run_receipt_scanner()

elif page == "Book Recommender":
    st.title("🚧 Coming Soon: Book Recommender Agent")

else:
    # Optional email capture for the Home page
    prompt_for_optional_email()

    # Log visit once user has provided an email
    if "user_email" in st.session_state and st.session_state["user_email"]:
        log_once_per_page("Home Page")

    st.title("Welcome to Swetha's GenAI Portfolio 👋")
    st.markdown(
        """
        Welcome to my Journey

        Use the sidebar to access the available GenAI agents.

        - 📸 **Receipt Scanner** — See what you've already purchased
        - 📚 **Book Recommender** — Find books for your kids based on themes
        """
    )