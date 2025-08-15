import streamlit as st
import os

from utils.email_utils import prompt_for_optional_email
from agents.pages.Receipt_Scanner import run_receipt_scanner
from utils.tracking import log_once_per_page

def doppler_bootstrap():
    DT = st.secrets.get("DOPPLER_TOKEN")
    DP = st.secrets.get("DOPPLER_PROJECT")
    DC = st.secrets.get("DOPPLER_CONFIG")
    if DT and DP and DC:
        import requests  # make sure 'requests' is in requirements.txt
        r = requests.get(
            "https://api.doppler.com/v3/configs/config/secrets",
            params={"project": DP, "config": DC},
            headers={"Authorization": f"Bearer {DT}"},
            timeout=10,
        )
        r.raise_for_status()
        for k, v in r.json().get("secrets", {}).items():
            val = v.get("computed") if isinstance(v, dict) else v
            if val and not os.getenv(k):
                os.environ[k] = str(val)
doppler_bootstrap()


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