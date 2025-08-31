# src/portfolio_homepage.py
import sys, asyncio
import streamlit as st

# --- Windows async policy (keep) ---
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Optional theme (KPI look) ---
try:
    from provisioning.theme import inject_theme
    inject_theme()
except Exception:
    pass  # OK if you don't have custom theme

# --- Optional: Doppler & LangSmith setup (only if you use them) ---
def doppler_bootstrap():
    DT = st.secrets.get("DOPPLER_TOKEN")
    DP = st.secrets.get("DOPPLER_PROJECT")
    DC = st.secrets.get("DOPPLER_CONFIG")
    if not (DT and DP and DC):
        return
    import os, requests
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

USE_DOPPLER = bool(st.secrets.get("DOPPLER_TOKEN"))
if USE_DOPPLER:
    doppler_bootstrap()

# Export LangSmith keys if you actually use LangSmith
if any(k in st.secrets for k in ("LANGCHAIN_API_KEY","LANGCHAIN_TRACING_V2","LANGCHAIN_PROJECT")):
    import os
    for key in ("LANGCHAIN_API_KEY","LANGCHAIN_TRACING_V2","LANGCHAIN_PROJECT"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])

# --- Page config: keep only here ---
st.set_page_config(page_title="Welcome to Swetha's GenAI Portfolio 👋", layout="wide")

# --- Optional: email + visit logging for Landing page ---
from utils.email_utils import prompt_for_optional_email
from utils.tracking import log_once_per_page

def landing():
    st.title("AI AGENTS")
    st.subheader("About Me")
    # Recruiter email (optional)
    prompt_for_optional_email()
    if st.session_state.get("user_email"):
        log_once_per_page("Portfolio Landing")

    st.markdown(
        """
Hi, and thanks for stopping by!

I began my career as a Data Analyst and I’m now building practical, hands-on **GenAI agents**.

**Featured Projects**
- 🧾 **Receipt Scanner** — Extracts & organizes purchase details automatically  
- 📚 **Book Recommender** — Suggests children’s books by theme & interest
        """
    )

# ---- Pages / Nav (your existing tree) ----
home = st.Page(landing, title="Landing Page", icon="🏠", default=True)

receiptscanner_agent = st.Page("receiptscanner_homepage.py", title="ReceiptScanner", icon="🧾")
receiptscanner_run = st.Page("pages/1_receiptscanner_run.py", title="Execute", icon="▶️")
receiptscanner_documentation = st.Page("pages/12_receiptscanner_documentation.py", title="Documentation", icon="📑")
receiptscanner_analytics = st.Page("pages/13_receiptscanner_analytics.py", title="Analytics", icon="📊")
# Evaluator (ReceiptScanner) pages
receiptscanner_evaluator_home=st.Page("pages/14_evaluator_home.py",title="Evaluator")
receiptscanner_evaluator_manual = st.Page("pages/15_evaluator_manual_scoring.py", title="Evaluator – Gold Dataset Creation", icon="📝")
receiptscanner_evaluator_errors = st.Page("pages/16_evaluator_error_tagging.py", title="Evaluator – Error Tagging", icon="🏷️")
receiptscanner_evaluator_header_accuracy=st.Page("pages/18_evaluator_header_accuracy.py",title="Evaluator - Header Accuracy", icon="🧮")
receiptscanner_evaluator_altered_images_accuracy=st.Page("pages/20_evaluator_altered_images.py",title="Evaluator - Images Accuracy-Altered Images", icon="🧪️")
receiptscanner_evaluator_latency=st.Page("pages/21_evaluator_latency.py",title="Evaluator - Latency", icon="📝")
receiptscanner_evaluator_consistency=st.Page("pages/21_evaluator_consistency.py",title="Evaluator - Consistency", icon="📝")

# Utils (ReceiptScanner) pages
receiptscanner_utils_home=st.Page("pages/17_utils_home.py",title="Gold Dataset Ingest")
receiptscanner_utils_pertubed=st.Page("pages/19_utils_altered_images.py",title="Altered Images Creation")




bookrecommender_agent = st.Page("bookrecommender_homepage.py", title="BookRecommender", icon="📚")
bookrecommender_run = st.Page("pages/3_bookrecommender_run.py", title="Execute", icon="🚀")
bookrecommender_documentation = st.Page("pages/32_bookrecommender_documentation.py", title="Documentation", icon="📑")
bookrecommender_analytics = st.Page("pages/33_bookrecommender_analytics.py", title="Analytics", icon="📊")

nav = st.navigation(
    {
        "Home": [home],
        "ReceiptScanner": [
            receiptscanner_agent,
            receiptscanner_run,
            receiptscanner_documentation,
            receiptscanner_analytics
        ],
        "ReceiptScanner-Evaluator":[

            receiptscanner_evaluator_home,
            receiptscanner_evaluator_manual,
            receiptscanner_evaluator_errors,
            receiptscanner_evaluator_header_accuracy,
            receiptscanner_evaluator_altered_images_accuracy,
            receiptscanner_evaluator_latency,
            receiptscanner_evaluator_consistency,
        ],
        "ReceiptScanner-Utils":[

            receiptscanner_utils_home,
            receiptscanner_utils_pertubed,
        ],
        "BookRecommender": [
            bookrecommender_agent,
            bookrecommender_run,
            bookrecommender_documentation,
            bookrecommender_analytics,
            # Later you can nest its own "Evaluator" children the same way
        ],
    },
    position="sidebar",
)
nav.run()
