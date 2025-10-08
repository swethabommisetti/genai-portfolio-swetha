# src/pages/bookrecommender_documentation.py
import streamlit as st

st.set_page_config(page_title="BookRecommender — Docs", layout="wide")

# ---------- tiny CSS polish ----------
st.markdown(
    """
<style>
:root { --accent:#0ea5e9; --muted:#9aa3af; }
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
st.title("📖 BookRecommender — Documentation")
st.caption("Architecture, decisions, and how the Dual Model Runner works (side-by-side prompts, logging, and analytics).")

# ---------- section 1: tech stack ----------
st.header("1. Tech Stack")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown(
        """
<span class="badge"><a href="https://streamlit.io" target="_blank">Streamlit (UI)</a></span>
<span class="badge"><a href="https://supabase.com" target="_blank">Supabase (Postgres)</a></span>
<span class="badge"><a href="https://groq.com" target="_blank">Groq (Llama-3.1-70B)</a></span>
""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
<span class="badge"><a href="https://platform.openai.com" target="_blank">OpenAI (gpt-4o-mini)</a></span>
<span class="badge"><a href="https://huggingface.co" target="_blank">HF Inference (Mixtral)</a></span>
<span class="badge">Custom <code>utils/</code> layer (Supabase client)</span>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
- **Frontend/UI**: Streamlit — quick to build A/B demo UIs with session state.  
- **Database**: Supabase — `br_runs` and `br_ratings` tables for run logs & human feedback.  
- **LLMs**: Pluggable providers (Groq, OpenAI, Hugging Face) to compare quality/latency.  
- **Utilities**: `utils.supabase_utils.get_supabase_client()` centralizes secrets & connection.
"""
)

# ---------- section 2: why these tools ----------
st.header("2. Why These Tools")
st.markdown(
    """
- **Side-by-side experiments** need fast UI tweaks → Streamlit is ideal.  
- **Traceability** for recruiters → every run is logged with inputs, provider, latency, results JSON.  
- **Vendor comparison** → same prompt pattern, multiple providers; fair A/B scoring UI.  
- **Single source of truth** for DB access → project-standard `get_supabase_client()` avoids secret drift.
"""
)

# ---------- section 3: page workflow ----------
st.header("3. Workflow of `bookrecommender_run.py`")

with st.expander("Inputs ➜ Prompt Build ➜ Model Call ➜ Parse ➜ Persist ➜ Rate ➜ Review", expanded=True):
    st.markdown(
        """
1) **User Inputs**  
   - Age, reading level, interests, diversity preference  
   - Tone/persona per panel (A/B), plus prompt style (A=minimal, B=structured+diversity)

2) **Prompt Builder**  
   - Produces a JSON-friendly instruction string.  
   - Style A: minimal; Style B: adds warm tone + diversity rule.

3) **Model Call**  
   - Provider per panel (Groq/OpenAI/HF).  
   - Returns raw text; we extract the first top-level JSON list.  
   - If parsing fails, we **fallback** to a safe list of 2-3 books.

4) **Persistence**  
   - Insert one row in **`br_runs`** per panel execution with: inputs, provider, latency, and **`results_json`**.

5) **Human Rating**  
   - Side selector (“Panel A/B”), 1-5 slider.  
   - Writes to **`br_ratings`** with provider + prompt style.

6) **Summary Expander**  
   - Fetches last 50 rows from `br_runs` and shows basic counts by provider.

7) **Sticky Results**  
   - Uses `st.session_state["br_state"]` so results **stay on screen** until you click **“🧹 Clear results”**.
        """
    )

# ---------- section 4: required secrets & env ----------
st.header("4. Secrets & Environment")

st.markdown(
    """
Add these in **`.streamlit/secrets.toml`** (local) or **Streamlit Cloud → Settings → Secrets**:
"""
)
st.code(
    """
[supabase]
url = "https://<YOUR-PROJECT>.supabase.co"
key = "<SERVICE_ROLE_OR_ANON_KEY_WITH_RLS>"

# Optional model providers (set as environment variables in Cloud)
# GROQ_API_KEY="..."
# OPENAI_API_KEY="..."
# HF_API_KEY="..."
""",
    language="toml",
)
st.markdown(
    """
- This page **does not** read `st.secrets` directly; it uses `utils.supabase_utils.get_supabase_client()`.  
- For providers, unset keys simply make that provider use the **fallback** results (still logs runs).
"""
)

# ---------- section 5: data model ----------
st.header("5. Data Model (Supabase)")

st.subheader("5.1 DDL — `br_runs`")
st.code(
    """
create table if not exists public.br_runs (
  id            uuid primary key default gen_random_uuid(),
  run_ts        timestamptz not null default now(),
  user_email    text,
  age           int,
  reading_level text,
  interests     jsonb,
  diversity     text,
  panel         text check (panel in ('A','B')),
  provider      text,               -- 'groq' | 'openai' | 'hf'
  tone          text,
  prompt_style  text,               -- 'A' | 'B'
  latency_ms    int,
  result_count  int,
  results_json  jsonb               -- array of {title, author, age_fit, why, tags}
);

-- helpful indexes
create index if not exists br_runs_ts_idx on public.br_runs (run_ts desc);
create index if not exists br_runs_provider_idx on public.br_runs (provider);
create index if not exists br_runs_panel_idx on public.br_runs (panel);
""",
    language="sql",
)

st.subheader("5.2 DDL — `br_ratings`")
st.code(
    """
create table if not exists public.br_ratings (
  id            uuid primary key default gen_random_uuid(),
  rating_ts     timestamptz not null default now(),
  user_email    text,
  panel         text check (panel in ('A','B')),
  score         int check (score between 1 and 5),
  provider      text,
  prompt_style  text,       -- 'A' | 'B'
  age           int,
  interests     jsonb
);

create index if not exists br_ratings_ts_idx on public.br_ratings (rating_ts desc);
""",
    language="sql",
)

st.markdown(
    """
**RLS** (optional): if you enable RLS, allow inserts for anonymous users or gate by a session email claim.  
For portfolio demos, many teams keep RLS off on these two tables or permit broad inserts.
"""
)

# ---------- section 6: code highlights ----------
st.header("6. Code Highlights")

st.subheader("6.1 Project-standard Supabase client")
st.code(
    """
from utils.supabase_utils import get_supabase_client
supabase = get_supabase_client()
rows = supabase.table("br_runs").select("*").limit(1).execute().data
""",
    language="python",
)

st.subheader("6.2 Prompt construction (A vs B)")
st.code(
    r'''
def build_prompt(age, level, interests, tone, style="A", diversity="No preference"):
    base = (
        f"You are a children's librarian. Recommend 2–3 books for a {age}-year-old.\n"
        f"Reading level: {level}. Interests: {', '.join(interests) or 'general'}.\n"
    )
    if style == "A":
        rules = "Return JSON list: [{title, author, age_fit, why, tags}]. Keep blurbs one sentence."
    else:
        rules = ("Use a warm, encouraging tone. Include at least one contemporary and one classic. "
                 "Return JSON list: [{title, author, age_fit, why, tags}]. Tags should include interest keywords.")
    diversity_rule = " Include at least one diverse author." if diversity == "Include diverse authors" else ""
    return f"{base}Tone/persona: {tone}.\n{rules}{diversity_rule}"
''',
    language="python",
)

st.subheader("6.3 Sticky results via `session_state`")
st.code(
    """
if "br_state" not in st.session_state:
    st.session_state["br_state"] = {"A": None, "B": None, "email": ""}

# on Run A:
st.session_state["br_state"]["A"] = {
  "provider": provA, "tone": toneA, "style": styleA,
  "prompt": promptA, "results": resA, "latency_ms": latencyA,
  "age": age, "level": level, "interests": interests, "diversity": diversity,
}

# render persists until explicit clear:
if st.button("🧹 Clear results"):
    st.session_state["br_state"]["A"] = None
    st.session_state["br_state"]["B"] = None
    st.rerun()
""",
    language="python",
)

st.subheader("6.4 Run logging & ratings")
st.code(
    """
# br_runs insert
supabase.table("br_runs").insert({
    "user_email": st.session_state.get("email"),
    "age": age,
    "reading_level": level,
    "interests": interests,
    "diversity": diversity,
    "panel": "A",
    "provider": provA.split()[0].lower(),
    "tone": toneA,
    "prompt_style": "A" if styleA.startswith("A") else "B",
    "latency_ms": latencyA,
    "result_count": len(resA or []),
    "results_json": resA or [],
}).execute()

# br_ratings insert
supabase.table("br_ratings").insert({
    "user_email": st.session_state.get("email"),
    "panel": "A",
    "score": score,                # 1..5
    "provider": provA.split()[0].lower(),
    "prompt_style": "A",
    "age": age,
    "interests": interests,
}).execute()
""",
    language="python",
)

# ---------- section 7: local & cloud ----------
st.header("7. Local & Cloud")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("7.1 Local Dev")
    st.markdown("- Add secrets to **`.streamlit/secrets.toml`**.")
    st.code("streamlit run src/portfolio_homepage.py", language="bash")
    st.markdown(
        """
Open from other devices on your Wi-Fi:
1. Find your IPv4 with <span class="kbd">ipconfig</span> / <span class="kbd">ifconfig</span>  
2. Visit <span class="kbd">http://&lt;your-ip&gt;:8501</span>
""",
        unsafe_allow_html=True,
    )

with right:
    st.subheader("7.2 Streamlit Community Cloud")
    st.markdown(
        """
- Paste the same TOML block into **Settings → Secrets**.  
- Push to `main` → auto build/deploy.  
- Keep `requirements.txt` minimal & pin LLM libs to avoid surprise upgrades.
"""
    )

# ---------- section 8: troubleshooting ----------
st.header("8. Troubleshooting")

st.markdown(
    """
- **`st.secrets has no key "supabase"`**  
  You’re likely bypassing the standard helper. Confirm `bookrecommender_run.py` imports  
  `from utils.supabase_utils import get_supabase_client` and **do not** access `st.secrets` directly.

- **No results / parsing errors**  
  Providers sometimes return extra text; we slice the first `[...]` block. If JSON parsing fails, we use a **safe fallback** list so the demo never blocks.

- **Inserts fail**  
  Check table names & DDL above, and verify your key has permissions (RLS off or permissive policy).

- **Latency field is 0**  
  Ensure `latency_ms` is computed around the provider call and passed into the insert.

- **Results disappear after any UI change**  
  Make sure you’re on the version that uses `st.session_state["br_state"]` and a dedicated **“🧹 Clear results”** button.
"""
)

st.divider()
st.caption("Last updated · BookRecommender documentation. Links open in new tabs.")
