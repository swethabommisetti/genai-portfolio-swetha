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
    # ===== Hero =====
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:12px">
          <span style="font-size:42px">🤝</span>
          <h1 style="margin:0">AI Agents — Built like real products</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Hands-on projects, measurable evals, and clear roadmaps. Built and documented by **Swetha Bommisetti**.")

    # Recruiter email (optional)
    with st.expander("Recruiters & collaborators — leave your email (optional)", expanded=False):
        prompt_for_optional_email()
        if st.session_state.get("user_email"):
            log_once_per_page("Portfolio Landing")

    st.divider()
    st.markdown(
        """
**Hi, I'm Swetha — a Data & BI Analyst who builds practical, hands-on GenAI agents.**  
This portfolio shows that even as a newcomer to AI, I can **design, build, and evaluate** production-style agents end-to-end:
- clean UI with Streamlit  
- data + storage with Supabase  
- structured pipelines with LangGraph 
- measurable quality via custom **Evaluators** and **Gold Datasets**
- Secrets&Keys
    - to keep things safe, the project never hard-codes senstive keys. instead
         - On my laptop, they live in a small private file so i can run everything without exposing passwords
         - On the cloud, secrets are stored securely in Doppler or Streamlit's built-in valut. When the app runs,
             the app pulls the keys from doppler via github actions.
        """
    )
    st.divider()
    # ===== Value props =====
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🧪 Evidence-driven")
        st.write("Every agent ships with **evaluators** (accuracy, latency, robustness) so you can see progress, not promises.")
    with c2:
        st.markdown("### 🧩 End-to-end")
        st.write("From **ingest → extraction → storage → analytics**, each app is wired to Supabase and observable.")
    with c3:
        st.markdown("### 📕 Documented for hand-off")
        st.write("Each page reads like an internal doc: what it does, decisions made, trade-offs, and next steps.")

    st.divider()

    # ===== Featured projects =====
    st.markdown("## Featured Projects")
    f1, f2 = st.columns([1.2, 1])
    with f1:
        st.markdown("#### 🧾 ReceiptScanner")
        st.write(
            "- Extracts store, total, date & line items from receipt images.\n"
            "- Public **gold datasets** can be imported; users can create gold labels.\n"
            "- Robustness tests with **altered images** (rotate/blur/etc.)."
        )
        st.markdown("**Evaluators:** Header Accuracy, Altered-Image Robustness, Latency, Consistency")
        st.markdown("**Stack:** Streamlit, Supabase (Postgres + Storage), LangGraph, Groq Llama-3.1")
    with f2:
        st.metric("Header Accuracy (demo set)", "—", help="Shown live in the Evaluator pages")
        st.metric("p50 Latency", "—")
        st.metric("Robustness (rotate fixed)", "—")

    st.markdown("---")

    g1, g2 = st.columns([1.2, 1])
    with g1:
        st.markdown("#### 📚 BookRecommender")
        st.write(
            "- Suggests children’s books by **theme** & **reading level**.\n"
            "- Shows **trace/latency** and **prompt iterations** for transparency."
        )
        st.markdown("**Stack:** Streamlit UI, simple vector search, structured prompts")
    with g2:
        st.metric("Recommendation latency", "—")
        st.metric("User saves to list", "—")

    st.divider()

    # ===== What each section does (quick map) =====
    st.markdown("## Site Map — What you’ll find")
    l, r = st.columns(2)
    with l:
        st.markdown("### ReceiptScanner")
        st.write("**Execute** – upload a receipt and see extraction.")
        st.write("**Documentation** – tech choices, repo structure, and workflow.")
        st.write("**Analytics** – simple KPIs and trends.")
        st.write("**Evaluator** – gold creation, error tagging, accuracy/latency/robustness/consistency.")
        st.write("**Utils** – import public gold datasets; create altered images for stress tests.")
    with r:
        st.markdown("### BookRecommender")
        st.write("**Execute** – try the recommender.")
        st.write("**Documentation** – design and future roadmap.")
        st.write("**Analytics** – request/latency view.")

    st.divider()

    # ===== Roadmap & CTA =====
    st.markdown("## What I’m building next")
    st.write(
        "- **Email-to-Summary Agent** for parents of multiple kids across schools/day-care apps.\n"
        "  Aggregates daily emails into one **smart summary** with highlights and trends."
    )
    #st.info("If you’d like a short walkthrough or a sandbox account, drop your email above. Happy to demo live.")

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
        ],
    },
    position="sidebar",
)

nav.run()
