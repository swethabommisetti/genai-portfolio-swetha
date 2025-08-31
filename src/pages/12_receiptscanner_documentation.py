# src/pages/12_receiptscanner_documentation.py
import streamlit as st

st.set_page_config(page_title="ReceiptScanner — Docs", layout="wide")

# ---------- tiny CSS polish ----------
st.markdown(
    """
<style>
/* page polish */
:root { --accent:#7c3aed; --muted:#9aa3af; }
html, body, [class*="css"]  { font-feature-settings: "ss01" on, "ss02" on; }
h1 { font-weight: 800; letter-spacing: .2px; }
h2 { 
  font-weight: 800; 
  margin: 1.75rem 0 .75rem; 
  padding-left: .65rem; 
  border-left: 6px solid var(--accent);
}
h3 { font-weight: 700; margin: 1.25rem 0 .25rem; }
ul { line-height: 1.6; margin-top: .25rem; }
li + li { margin-top: .25rem; }
small, .muted { color: var(--muted); }
code, pre { font-size: .95rem; border-radius: 8px; }
.kbd { 
  background:#0f172a; border:1px solid #1f2937; color:#e5e7eb; 
  border-radius: 8px; padding: 2px 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
.badge {
  display:inline-block; padding:6px 10px; margin:6px 8px 0 0; 
  border-radius: 999px; border:1px solid #1f2937; background:#0b1220;
  font-size:.92rem; white-space:nowrap;
}
.badge a { text-decoration: none; }
hr { border: 0; border-top: 1px solid #1f2937; margin: 1.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- header ----------
st.title("📑 ReceiptScanner — Documentation")
st.caption("Architecture, decisions, and how to run it locally & on the public internet.")

# ---------- section 1: tech stack ----------
st.header("1. Tech Stack")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(
        """
<span class="badge"><a href="https://streamlit.io" target="_blank">Streamlit (UI)</a></span>
<span class="badge"><a href="https://supabase.com" target="_blank">Supabase (Postgres + Storage)</a></span>
<span class="badge"><a href="https://python.langchain.com/docs/langgraph" target="_blank">LangGraph (Orchestration)</a></span>
""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
<span class="badge"><a href="https://groq.com" target="_blank">Groq Llama-3.1 (LLM)</a></span>
<span class="badge">Python utils (PIL / re / dotenv)</span>
<span class="badge">Custom <code>utils/</code> layer</span>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
- **Frontend/UI**: Streamlit — lightning-fast for interactive demos and recruiter-friendly UIs.  
- **Database & Storage**: Supabase — Postgres tables for metadata & JSON; Object Storage for images.  
- **Orchestration**: LangGraph — reliable multi-step agent flow (extract ➜ summarize).  
- **LLM**: Groq Llama-3.1 — low-latency, great DX for structured outputs.  
- **Utilities**: a thin `utils/` layer for Supabase, email capture, visitor tracking.
"""
)

# ---------- section 2: why I picked each tool ----------
st.header("2. Why I Picked Each Tool")
st.markdown(
    """
- **Streamlit** → Demo-ready, no frontend framework overhead.  
- **Supabase** → One platform for SQL + file storage + auth + REST/RPC APIs.  
- **LangGraph** → Clean state machines; easy to add steps or swap LLM/tools.  
- **Groq Llama-3.1** → Snappy inference for a good UX during live demos.  
- **Custom `utils/`** → Portable code for data access, logging, and email capture.
"""
)

# ---------- section 3: workflow ----------
st.header("3. Workflow of the Application")

with st.expander("Upload ➜ Extract ➜ Persist (open)", expanded=True):
    st.markdown(
        """
1) **Upload Receipt**  
   User uploads JPG/PNG ➜ stored in Supabase **Storage**. A row is inserted into `receipt_files`.

2) **Extraction Pipeline**  
   - The image path goes into a tool (OCR/LLM) behind a LangGraph flow:  
     - `extract_receipt`: parses header + line items  
     - `generate_final_answer`: small, user-friendly summary  

3) **Persistence**  
   - Normalized header ➜ `receipts_dtl`  
   - Normalized items ➜ `receipt_items`  
   - We keep short-lived **signed URLs** for preview in evaluator pages

4) **Evaluator**  
   - **Manual scoring** ➜ `evaluations`  
   - **Error tagging** ➜ `evaluation_errors`  

5) **Analytics**  
   - High-level counts & simple accuracy metrics for the portfolio story.
        """
    )

# ---------- section 4: local & public ----------
st.header("4. Local & Public Access (Dev ➜ Cloud)")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("4.1 Local Dev")
    st.markdown(
        """
- Put secrets in **`.streamlit/secrets.toml`**.  
- Run the app:  
"""
    )
    st.code("streamlit run src/portfolio_homepage.py", language="bash")

    st.markdown(
        """
- Open from other devices on your Wi-Fi:  
  1. Find your IPv4 with `ipconfig` / `ifconfig`  
  2. Visit `http://<your-local-ip>:8501` on phone/laptop  
- Optional Docker (Jupyter or Streamlit) commands you noted also work.
"""
    )

with right:
    st.subheader("4.2 Public Internet (Streamlit Community Cloud)")
    st.markdown(
        """
- Store secrets in **Doppler** (or Streamlit Cloud Secrets) and connect your GitHub repo.  
- On push to `main`, Streamlit Cloud builds & deploys.  
- Live demo:  
"""
    )
    st.code("https://genai-portfolio-swetha.streamlit.app/", language="text")
    st.markdown(
        """
<small class="muted">
Tip: keep `requirements.txt` lean and pin any LLM/LC packages to avoid surprise upgrades.
</small>
""",
        unsafe_allow_html=True,
    )

    # ---------- section 4: local & public ----------
st.header("5. What Each Menu Item Does")
st.markdown(
    """

## App Guide — What each menu item does

### ReceiptScanner
- **ReceiptScanner**  
  Upload an image (JPG/PNG). We store it in Supabase Object Storage and insert a row into `receipt_files`.
- **Execute**  
  Runs the LangGraph extraction pipeline (OCR → normalize → structure). Writes header + line items to `receipts_dtl`.
- **Documentation**  
  The page you’re reading. Architecture, tech choices, and workflow.
- **Analytics**  
  Simple observability on volumes, runs, and basic quality counters.

### ReceiptScanner-Evaluator
- **Evaluator — Gold Dataset Creation**  
  Pick a receipt → confirm store/total/date → save a gold record (SCD-II style) in `receipts_gold`.
- **Evaluator — Error Tagging**  
  Tag systematic model errors (e.g., “date parsing”, “wrong currency”). Stores tags for later analysis.
- **Evaluator — Header Accuracy**  
  Compare model predictions vs current gold across store/total/date. Shows PASS/FAIL per field and overall.
- **Evaluator — Images Accuracy — Altered Images**  
  Robustness testing. Compares **baseline** vs **perturbed** images (e.g., rotate) against gold. Flags regressions.
- **Evaluator — Latency**  
  Measures p50/p95 latency and errors for the extraction pipeline across a sample.

### ReceiptScanner-Utils
- **Gold Dataset Ingest**  
  Pulls public receipt datasets (SROIE, ExpressExpense, etc.) into Supabase (each in its own table & storage path).
- **Altered Images Creation**  
  Generate perturbed images from public/personal receipts (rotate, brightness, etc.), saved to `pertubed_images/`.

### BookRecommender (demo app #2)
- Mirrors the pattern (Execute, Docs, Analytics) to show you can ship a second small AI app in the same repo.
"""
)
st.divider()
st.caption("Last updated · this page is auto-rendered by Streamlit; all links open in a new tab.")
