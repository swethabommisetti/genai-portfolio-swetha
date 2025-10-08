# src/pages/bookrecommender_run.py
import os, json, time, requests
import streamlit as st
from utils.supabase_utils import get_supabase_client  # ✅ project-standard helper

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Book Recommender – Dual Model Runner",
                   page_icon="📚", layout="wide")
st.title("📚 Book Recommender — Dual Model Runner")
st.caption("Compare prompt styles and providers side-by-side, then log results and ratings to Supabase.")

# ──────────────────────────────────────────────────────────────────────────────
# Session state for persistent results
# ──────────────────────────────────────────────────────────────────────────────
if "br_state" not in st.session_state:
    st.session_state["br_state"] = {
        "A": None,   # will hold dict with {provider, tone, style, prompt, results, latency_ms}
        "B": None,
        "email": "", # optional user email for logging
    }
br_state = st.session_state["br_state"]

# optional email capture (kept in session)
with st.expander("Optional: add your email for logging"):
    _email = st.text_input("Email", value=br_state.get("email", ""), placeholder="you@example.com", key="br_email_input")
    if st.button("Save email", key="save_email_btn"):
        if _email and "@" in _email:
            br_state["email"] = _email.strip()
            st.success("Saved for this session.")
        elif _email:
            st.warning("Please enter a valid email.")

# ──────────────────────────────────────────────────────────────────────────────
# Supabase (project standard)
# ──────────────────────────────────────────────────────────────────────────────
supabase = get_supabase_client()
try:
    supabase.table("br_runs").select("id").limit(1).execute()
    st.caption("🔌 Supabase connected.")
except Exception as e:
    st.warning(f"Supabase check failed (still usable after tables exist): {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(age, level, interests, tone, style="A", diversity="No preference"):
    base = (
        f"You are a children's librarian. Recommend 2–3 books for a {age}-year-old.\n"
        f"Reading level: {level}. Interests: {', '.join(interests) or 'general'}.\n"
    )
    if style == "A":
        rules = "Return JSON list: [{title, author, age_fit, why, tags}]. Keep blurbs one sentence."
    else:
        rules = (
            "Use a warm, encouraging tone. Include at least one contemporary and one classic. "
            "Return JSON list: [{title, author, age_fit, why, tags}]. Tags should include interest keywords."
        )
    diversity_rule = " Include at least one diverse author." if diversity == "Include diverse authors" else ""
    persona = f"Tone/persona: {tone}."
    return f"{base}{persona}\n{rules}{diversity_rule}"

# ──────────────────────────────────────────────────────────────────────────────
# Model caller
# ──────────────────────────────────────────────────────────────────────────────
def call_model(provider: str, prompt: str, temperature: float = 0.4, timeout_s: int = 10):
    fallback = [
        {"title":"The Wild Robot","author":"Peter Brown","age_fit":"8+",
         "why":"Adventure with empathy and nature.","tags":["adventure","empathy"]},
        {"title":"A Wrinkle in Time","author":"Madeleine L'Engle","age_fit":"10+",
         "why":"Time-travel quest about courage and family.","tags":["science","classic"]},
        {"title":"The Land of Stories","author":"Chris Colfer","age_fit":"9+",
         "why":"Fantasy that blends fairy tales with modern humor.","tags":["fantasy","humor"]},
    ]
    try:
        if provider.startswith("Groq"):
            key = os.getenv("GROQ_API_KEY")
            if not key: return fallback
            url = "https://api.groq.com/openai/v1/chat/completions"
            payload = {
                "model": "llama-3.1-70b-versatile",
                "messages": [
                    {"role":"system","content":"Return only valid JSON."},
                    {"role":"user","content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 600,
            }
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout_s)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]

        elif provider.startswith("OpenAI"):
            import openai  # lazy import
            key = os.getenv("OPENAI_API_KEY")
            if not key: return fallback
            client = openai.OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"Return only valid JSON."},
                    {"role":"user","content": prompt}
                ],
                temperature=temperature,
                max_tokens=600,
            )
            txt = resp.choices[0].message.content

        else:
            key = os.getenv("HF_API_KEY")
            if not key: return fallback
            url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
            headers = {"Authorization": f"Bearer {key}"}
            r = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                txt = data[0]["generated_text"]
            else:
                txt = json.dumps(fallback)

        # Parse JSON safely (extract the first top-level list)
        start, end = txt.find("["), txt.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(txt[start:end+1])
                if isinstance(parsed, list) and parsed:
                    return parsed[:3]
            except Exception:
                pass
        return fallback
    except Exception:
        return fallback

# ──────────────────────────────────────────────────────────────────────────────
# DB logging helpers
# ──────────────────────────────────────────────────────────────────────────────
def insert_run(panel, provider, tone, prompt_style, age, level, interests, diversity, results, latency_ms):
    try:
        data = {
            "user_email": br_state.get("email") or st.session_state.get("email"),
            "age": age,
            "reading_level": level,
            "interests": interests,
            "diversity": diversity,
            "panel": panel,
            "provider": provider.lower().split()[0],
            "tone": tone,
            "prompt_style": "A" if prompt_style.startswith("A") else "B",
            "latency_ms": int(latency_ms or 0),
            "result_count": len(results or []),
            "results_json": results or [],
        }
        supabase.table("br_runs").insert(data).execute()
    except Exception as e:
        st.warning(f"(⚠️ br_runs insert failed: {e})")

def insert_rating(panel, score, provider=None, prompt_style=None, age=None, interests=None):
    try:
        data = {
            "user_email": br_state.get("email") or st.session_state.get("email"),
            "age": age,
            "interests": interests,
            "panel": panel,
            "score": int(score),
            "provider": provider,
            "prompt_style": "A" if (prompt_style and prompt_style.startswith('A')) else "B",
        }
        supabase.table("br_ratings").insert(data).execute()
    except Exception as e:
        st.warning(f"(⚠️ br_ratings insert failed: {e})")

# ──────────────────────────────────────────────────────────────────────────────
# UI inputs (these do NOT clear results anymore)
# ──────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1,1,2,1])
age = c1.slider("Reader age", 3, 18, 10, key="br_age")
level = c2.selectbox("Reading level", ["Early reader","Middle grade","Young adult"], index=1, key="br_level")
interests = c3.multiselect("Interests", ["adventure","friendship","kindness","science","mystery","fantasy"],
                           default=["adventure"], key="br_interests")
diversity = c4.selectbox("Diversity preference", ["No preference","Include diverse authors"], index=1, key="br_diversity")

# Top-right control bar
ctrl_left, ctrl_right = st.columns([6, 1])
with ctrl_right:
    if st.button("🧹 Clear results", use_container_width=True, key="clear_results_btn"):
        br_state["A"] = None
        br_state["B"] = None
        st.rerun()

left, right = st.columns(2)

# ──────────────────────────────────────────────────────────────────────────────
# Panel A
# ──────────────────────────────────────────────────────────────────────────────
with left:
    st.markdown("### ◀️ Panel A")
    provA  = st.selectbox("Provider A", ["Groq (Llama 3.1-70B)","OpenAI (gpt-4o-mini)","HF Inference (Mixtral)"], index=0, key="provA")
    toneA  = st.selectbox("Tone / Persona (A)", ["friendly librarian","expert coach","fun & playful"], index=0, key="toneA")
    styleA = st.radio("Prompt style (A)", ["A (minimal)","B (structured & diversity)"], horizontal=True, index=1, key="styleA")

    runA = st.button("Run A", key="runA_btn")
    if runA:
        promptA = build_prompt(age, level, interests, toneA, style="A" if styleA.startswith("A") else "B", diversity=diversity)
        t0 = time.time()
        resA = call_model(provA, promptA)
        latencyA = int((time.time()-t0)*1000)
        # 🔒 persist result
        br_state["A"] = {
            "provider": provA, "tone": toneA, "style": styleA,
            "prompt": promptA, "results": resA, "latency_ms": latencyA,
            "age": age, "level": level, "interests": interests, "diversity": diversity,
        }
        insert_run("A", provA, toneA, styleA, age, level, interests, diversity, resA, latencyA)

    # Render current A (if present)
    if br_state["A"]:
        _A = br_state["A"]
        st.success(f"Returned {len(_A['results'])} recs in {_A['latency_ms']/1000:.1f}s ({_A['provider'].split()[0]})")
        st.caption(f"Tone: {_A['tone']} • Style: {_A['style']} • Diversity: {_A['diversity']}")
        st.code(_A["prompt"], language="markdown")
        st.subheader("Recommendations — Panel A")
        for i, r in enumerate(_A["results"], 1):
            st.markdown(f"**{i}. {r.get('title','')}** — {r.get('author','')}")
            tags = r.get("tags", [])
            st.caption(f"Age fit: {r.get('age_fit','')} | Tags: {', '.join(tags) if tags else '—'}")
            st.write(r.get("why",""))
            st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Panel B
# ──────────────────────────────────────────────────────────────────────────────
with right:
    st.markdown("### ▶️ Panel B")
    provB  = st.selectbox("Provider B", ["OpenAI (gpt-4o-mini)","Groq (Llama 3.1-70B)","HF Inference (Mixtral)"], index=1, key="provB")
    toneB  = st.selectbox("Tone / Persona (B)", ["friendly librarian","expert coach","fun & playful"], index=2, key="toneB")
    styleB = st.radio("Prompt style (B)", ["A (minimal)","B (structured & diversity)"], horizontal=True, index=0, key="styleB")

    runB = st.button("Run B", key="runB_btn")
    if runB:
        promptB = build_prompt(age, level, interests, toneB, style="A" if styleB.startswith("A") else "B", diversity=diversity)
        t0 = time.time()
        resB = call_model(provB, promptB)
        latencyB = int((time.time()-t0)*1000)
        # 🔒 persist result
        br_state["B"] = {
            "provider": provB, "tone": toneB, "style": styleB,
            "prompt": promptB, "results": resB, "latency_ms": latencyB,
            "age": age, "level": level, "interests": interests, "diversity": diversity,
        }
        insert_run("B", provB, toneB, styleB, age, level, interests, diversity, resB, latencyB)

    # Render current B (if present)
    if br_state["B"]:
        _B = br_state["B"]
        st.success(f"Returned {len(_B['results'])} recs in {_B['latency_ms']/1000:.1f}s ({_B['provider'].split()[0]})")
        st.caption(f"Tone: {_B['tone']} • Style: {_B['style']} • Diversity: {_B['diversity']}")
        st.code(_B["prompt"], language="markdown")
        st.subheader("Recommendations — Panel B")
        for i, r in enumerate(_B["results"], 1):
            st.markdown(f"**{i}. {r.get('title','')}** — {r.get('author','')}")
            tags = r.get("tags", [])
            st.caption(f"Age fit: {r.get('age_fit','')} | Tags: {', '.join(tags) if tags else '—'}")
            st.write(r.get("why",""))
            st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Rating UI (uses persisted runs if available)
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("⭐ Rate the results (optional)"):
    side = st.selectbox("Which panel are you rating?", ["Panel A","Panel B"], key="rate_side")
    score = st.slider("How relevant were the suggestions?", 1, 5, 4, key="rate_score")
    if st.button("Save rating", key="save_rating_btn"):
        curr = br_state["A"] if side == "Panel A" else br_state["B"]
        if not curr:
            st.warning("Run the selected panel first.")
        else:
            insert_rating(
                panel="A" if side == "Panel A" else "B",
                score=score,
                provider=curr["provider"].split()[0].lower(),
                prompt_style=("A" if curr["style"].startswith("A") else "B"),
                age=curr["age"],
                interests=curr["interests"],
            )
            st.success("✅ Rating logged to Supabase")

# ──────────────────────────────────────────────────────────────────────────────
# Summary view (last 50 runs)
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📊 Recent run summary (last 50)"):
    try:
        runs = supabase.table("br_runs").select("*").order("run_ts", desc=True).limit(50).execute().data
        if runs:
            st.write(f"Total runs: {len(runs)}")
            prov_count = {}
            for r in runs:
                prov = (r.get("provider") or "").lower()
                prov_count[prov] = prov_count.get(prov, 0) + 1
            st.write("By provider:", prov_count)
        else:
            st.caption("No runs yet.")
    except Exception as e:
        st.warning(f"(summary failed: {e})")
