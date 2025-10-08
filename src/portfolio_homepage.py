# src/portfolio_homepage.py
import sys, asyncio, os
import streamlit as st

# --- Windows async policy (keep) ---
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Optional theme (safe to skip if not present) ---
try:
    from provisioning.theme import inject_theme
    inject_theme()
except Exception:
    pass

# --- Optional: Doppler & LangSmith bootstrap ---
def doppler_bootstrap():
    DT = st.secrets.get("DOPPLER_TOKEN")
    DP = st.secrets.get("DOPPLER_PROJECT")
    DC = st.secrets.get("DOPPLER_CONFIG")
    if not (DT and DP and DC):
        return
    import requests
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

if st.secrets.get("DOPPLER_TOKEN"):
    doppler_bootstrap()

if any(k in st.secrets for k in ("LANGCHAIN_API_KEY","LANGCHAIN_TRACING_V2","LANGCHAIN_PROJECT")):
    for key in ("LANGCHAIN_API_KEY","LANGCHAIN_TRACING_V2","LANGCHAIN_PROJECT"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])

# --- Page config ---
st.set_page_config(page_title="Swetha Bommisetti — GenAI Portfolio", layout="wide")

# --- Optional: email logging ---
from utils.email_utils import prompt_for_optional_email
from utils.tracking import log_once_per_page

# ---------------- Landing content (compact) ----------------
def landing():
    st.title("Human-trusted AI, not just AI.")
    st.markdown(
        "I build and evaluate GenAI agents with **human-in-the-loop** workflows, "
        "**gold datasets**, and **LangSmith-backed** metrics — so outputs are accurate, fast, and ready for hand-off."
    )

    # CTAs (native Streamlit)
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.page_link("receiptscanner_homepage.py", label="🧾 Explore ReceiptScanner", icon="↗")
    with c2:
        st.page_link("bookrecommender_homepage.py", label="📚 Explore BookRecommender", icon="↗")

    with st.expander("Recruiters & collaborators — leave your email (optional)", expanded=False):
        prompt_for_optional_email()
        if st.session_state.get("user_email"):
            log_once_per_page("Portfolio Landing")

    st.divider()

    # Value pillars (short)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.subheader("🧪 Evidence-driven")
        st.caption("Evaluators for accuracy, latency, and robustness so you see progress, not promises.")
    with p2:
        st.subheader("🧩 End-to-end")
        st.caption("Ingest → extraction → storage → analytics. Wired to Supabase and observable.")
    with p3:
        st.subheader("📕 Hand-off ready")
        st.caption("Each page doubles as internal documentation: decisions, trade-offs, and next steps.")

    st.divider()

    # ---------- Featured Projects (NOW TABS to reduce scroll) ----------
    st.header("Featured Projects")
    tabs = st.tabs(["🧾 ReceiptScanner", "📚 BookRecommender"])

    # --- TAB 1: ReceiptScanner ---
    with tabs[0]:
        left, right = st.columns([1.4, 0.6])
        with left:
            st.subheader("ReceiptScanner")
            st.write(
                "- **Purpose:** Built and evaluated a GenAI receipt extraction agent to digitize household and multi-vendor expenses after Mint.com’s shutdown — ensuring reliable data capture and measurable robustness across layouts."
            )
            # Quick actions
            a1, a2, a3 = st.columns(3)
            with a1:
                st.page_link("receiptscanner_homepage.py", label="Live Demo", icon="▶️")
            with a2:
                st.page_link("pages/12_receiptscanner_documentation.py", label="Docs", icon="📑")
            with a3:
                st.page_link("pages/14_evaluator_home.py", label="Evaluations", icon="🧮")

            # Details collapse to save space
            with st.expander("Show details (Evaluator Persona, Stack)", expanded=False):
                st.write(
                    "- **Evaluator Persona:** Validates extraction accuracy, labeling robustness, and consistency using gold datasets.\n"
                    "- **Stack:** Streamlit • Supabase (Postgres + Storage) • LangGraph • Groq Llama-3.1"
                )
        with right:
            st.caption("**Quality focus**")
            st.write("Accuracy, latency & robustness benchmarking via LangSmith.")

    # --- TAB 2: BookRecommender ---
    with tabs[1]:
        left, right = st.columns([1.4, 0.6])
        with left:
            st.subheader("BookRecommender")
            st.write(
                "- **Purpose:** Designed and evaluated a GenAI book recommender to help parents find age-appropriate and theme-based books for their children — ensuring factual, unbiased, and engaging suggestions."
            )
            a1, a2, a3 = st.columns(3)
            with a1:
                st.page_link("bookrecommender_homepage.py", label="Live Demo", icon="▶️")
            with a2:
                st.page_link("pages/32_bookrecommender_documentation.py", label="Docs", icon="📑")
            with a3:
                st.page_link("pages/33_bookrecommender_analytics.py", label="Analytics", icon="📊")

            with st.expander("Show details (Evaluator Persona, Stack, Note)", expanded=False):
                st.write(
                    "- **Evaluator Persona:** Assesses prompt quality, bias, and latency through structured test sets.\n"
                    "- **Stack:** Streamlit UI • simple vector search • structured prompts\n\n"
                    "_Note: Inspired by a real challenge at home — finding meaningful books for growing kids. Once validation is complete, it will be deployed for real family use._"
                )
        with right:
            st.caption("**Quality focus**")
            st.write("Recommendation diversity and response-time transparency.")

    st.divider()

    # ---------- Explore table (collapsed by default to save height) ----------
    with st.expander("🧭 Explore My Projects (what each page contains)", expanded=False):
        st.markdown(
            """
Each project includes dedicated pages so you can **run, understand, and measure** it like a real production system.

| Section | What you’ll find |
|----------|------------------|
| **Execute** | Try the agent live — upload receipts or request recommendations. |
| **Documentation** | Design choices, architecture decisions, and tech stack explained clearly. |
| **Analytics** | View simple KPIs and latency metrics. |
| **Evaluator** | Create gold datasets, tag errors, and measure accuracy, latency, and robustness. |
| **Utils** | Import public gold data or generate altered images for stress testing. |
            """
        )

    st.caption("Built with Streamlit • Secrets stored in Doppler or Streamlit’s vault • Hosted on Streamlit Community Cloud")

# ---------------- Nav (kept intact) ----------------
home = st.Page(landing, title="Landing Page", icon="🏠", default=True)

receiptscanner_agent = st.Page("receiptscanner_homepage.py", title="ReceiptScanner", icon="🧾")
receiptscanner_run = st.Page("pages/1_receiptscanner_run.py", title="Execute", icon="▶️")
receiptscanner_documentation = st.Page("pages/12_receiptscanner_documentation.py", title="Documentation", icon="📑")
receiptscanner_analytics = st.Page("pages/13_receiptscanner_analytics.py", title="Analytics", icon="📊")
receiptscanner_chat = st.Page("pages/10_receiptscanner_chat.py", title="Chat", icon="💬")

# Evaluator pages
receiptscanner_evaluator_home = st.Page("pages/14_evaluator_home.py", title="Evaluator")
receiptscanner_evaluator_manual = st.Page("pages/15_evaluator_manual_scoring.py", title="Evaluator – Gold Dataset Creation", icon="📝")
receiptscanner_evaluator_errors = st.Page("pages/16_evaluator_error_tagging.py", title="Evaluator – Error Tagging", icon="🏷️")
receiptscanner_evaluator_header_accuracy = st.Page("pages/18_evaluator_header_accuracy.py", title="Evaluator – Header Accuracy", icon="🧮")
receiptscanner_evaluator_altered_images_accuracy = st.Page("pages/20_evaluator_altered_images.py", title="Evaluator – Images Accuracy (Altered Images)", icon="🧪")
receiptscanner_evaluator_latency = st.Page("pages/21_evaluator_latency.py", title="Evaluator – Latency", icon="⏱️")
receiptscanner_evaluator_consistency = st.Page("pages/21_evaluator_consistency.py", title="Evaluator – Consistency", icon="🧭")

# Utils
receiptscanner_utils_home = st.Page("pages/17_utils_home.py", title="Gold Dataset Ingest")
receiptscanner_utils_pertubed = st.Page("pages/19_utils_altered_images.py", title="Altered Images Creation (Perturbed)")

# BookRecommender
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
            receiptscanner_analytics,
            receiptscanner_chat,
        ],
        "ReceiptScanner-Evaluator": [
            receiptscanner_evaluator_home,
            receiptscanner_evaluator_manual,
            receiptscanner_evaluator_errors,
            receiptscanner_evaluator_header_accuracy,
            receiptscanner_evaluator_altered_images_accuracy,
            receiptscanner_evaluator_latency,
            receiptscanner_evaluator_consistency,
        ],
        "ReceiptScanner-Utils": [
            receiptscanner_utils_home,
            receiptscanner_utils_pertubed,
        ],
        "BookRecommender": [
            bookrecommender_agent,
            bookrecommender_run,
            bookrecommender_documentation,
            bookrecommender_analytics,
        ],
    },
    position="sidebar",
)

nav.run()
