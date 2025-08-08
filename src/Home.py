import streamlit as st
import os

#from utils.tracking import log_page_visit
from utils.email_utils import prompt_for_optional_email
from agents.pages.Receipt_Scanner import run_receipt_scanner
from utils.tracking import log_once_per_page

# Optional email capture
email = prompt_for_optional_email()

st.set_page_config(page_title="Swetha's GenAI Portfolio", layout="centered")

# Log visit once user has interacted with the email box
if "user_email" in st.session_state and st.session_state["user_email"]:
    log_once_per_page("Home Page")

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
    st.title("Welcome to Swetha's GenAI Portfolio 👋")
    st.markdown(
        """
        Welcome to my Journey

        Use the sidebar to access the available GenAI agents.

        - 📸 **Receipt Scanner** — See what you've already purchased
        - 📚 **Book Recommender** — Find books for your kids based on themes
        """
    )
