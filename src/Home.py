import streamlit as st
import sys,os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))          # /app/src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # /app

from utils.email_utils import prompt_for_optional_email
from agents.pages.Receipt_Scanner import run_receipt_scanner
from utils.tracking import log_once_per_page


def doppler_bootstrap():
    DT = st.secrets.get("DOPPLER_TOKEN")
    DP = st.secrets.get("DOPPLER_PROJECT")
    DC = st.secrets.get("DOPPLER_CONFIG")
    if not (DT and DP and DC):
        return  # Safe no-op locally or if Doppler not configured

    import requests, os
    r = requests.get(
        "https://api.doppler.com/v3/configs/config/secrets",
        params={"project": DP, "config": DC},
        headers={"Authorization": f"Bearer {DT}"},
        timeout=10,
    )
    r.raise_for_status()
    secrets = r.json().get("secrets", {})

    # Map Doppler names -> standard names your app expects
    ALIASES = {
        # LLMs
        "MISTRAL__API__KEY": "MISTRAL_API_KEY",
        "MISTRAL__API_KEY":  "MISTRAL_API_KEY",
        "MISTRAL_API_KEY":   "MISTRAL_API_KEY",
        "GROQ__API__KEY":    "GROQ_API_KEY",
        "GROQ__API_KEY":     "GROQ_API_KEY",
        "GROQ_API_KEY":      "GROQ_API_KEY",
        # Supabase
        "SUPABASE__URL":                      "SUPABASE_URL",
        "SUPABASE_URL":                       "SUPABASE_URL",
        "SUPABASE__SUPABASE_SERVICE_KEY":     "SUPABASE_SERVICE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY":          "SUPABASE_SERVICE_KEY",
        "SUPABASE_SERVICE_KEY":               "SUPABASE_SERVICE_KEY",
        "SUPABASE_KEY":                       "SUPABASE_SERVICE_KEY",
    }

    for k, v in secrets.items():
        val = v.get("computed") if isinstance(v, dict) else v
        if not val:
            continue
        target = ALIASES.get(k, k)  # normalize when we know the alias
        if not os.getenv(target):   # <-- do NOT clobber existing env
            os.environ[target] = str(val)

doppler_bootstrap()


# Export LangSmith secrets to env so SDKs pick them up
for key in ["LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"]:
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = str(st.secrets[key])

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
        Hi, and thanks for stopping by!

I began my career as a Data Analyst, I’ve been building on that foundation by learning and experimenting with Generative AI. This portfolio is where I share my journey — the projects I’m working on, the tools I’m exploring, and the ways I’m applying AI to solve everyday problems.

My focus right now is on creating practical, hands-on AI agents that can automate tasks, surface insights, and make life a little easier. Each project here reflects something I’ve learned, and my goal is to keep improving and growing as I step back into my career.

Featured Projects

🧾 Receipt Scanner — Extracts and organizes purchase details automatically

📚 Book Recommender — Suggests children’s books based on themes and interests

I’m excited about what’s ahead and open to connecting with people who are exploring or hiring in the GenAI space.
        """
    )