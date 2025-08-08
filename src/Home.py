import streamlit as st
import runpy
from utils.tracking import log_page_visit
from utils.email_utils import prompt_for_optional_email

# Optional email capture
email = prompt_for_optional_email()

st.set_page_config(page_title="Swetha's GenAI Portfolio", layout="centered")

# Log visit once user has interacted with the email box
if "user_email" in st.session_state:
    log_page_visit("Home Page")

# Sidebar menu
page = st.sidebar.selectbox(
    "Choose a demo",
    ("Home", "Receipt Scanner", "Book Recommender"),
)

if page == "Receipt Scanner":
    runpy.run_path("/agents/pages/Receipt_Scanner.py", run_name="__main__")
elif page == "Book Recommender":
    runpy.run_path("/agents\pages\Book_Recommender.py", run_name="__main__")
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
