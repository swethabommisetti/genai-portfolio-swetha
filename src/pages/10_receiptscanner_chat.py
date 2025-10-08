# src/pages/10_receiptscanner_chat.py
import os, datetime as dt
import pandas as pd
import streamlit as st
from utils.supabase_utils import get_supabase_client

st.set_page_config(page_title="Receipt Scanner Chat + Prompt A/B", page_icon="🧾", layout="wide")
st.title("🧾 Receipt Scanner — Chat over Supabase (with Prompt A/B)")

# --- tiny CSS so results always wrap & stay boxed ---
st.markdown(
    """
    <style>
      .result-box {
        border: 1px solid var(--secondary-background-color, #333);
        border-radius: .6rem;
        padding: .85rem 1rem;
        margin-top: .35rem;
        background: rgba(255,255,255,.03);
        overflow-wrap: anywhere;
        word-break: break-word;
        white-space: pre-wrap;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Supabase client / secrets ---
sb = get_supabase_client()
def get_secret(name: str, default=None):
    return st.secrets.get(name, os.getenv(name, default))
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# --- helpers for dates/months ---
def month_start(d=None):
    d = d or dt.date.today()
    return d.replace(day=1)

def to_date_series(s):
    """Coerce a pandas Series to datetime.date (safe)."""
    if s is None:
        return s
    try:
        return pd.to_datetime(s, errors="coerce").dt.date
    except Exception:
        return s  # leave as-is if cannot convert

HELP = [
  "How much did I spend in the last 30 days?",
  "How much did I spend this month?",
  "Top 5 merchants this month",
  "Show month-by-month spend (last 6 months)",
  "Walmart trend by month",
  "Average receipt value this month",
  "How many receipts this month?",
  "Average items per receipt this month",
  "Most purchased items (last 60 days)",
  "Largest receipts (last 90 days)",
]

# --- Layout ---
main, aside = st.columns((7, 3), gap="large")
with aside:
    st.markdown("### 🔒 Demo scope")
    st.info(
        "This is a **demo environment**.\n\n"
        "You can ask exactly these **10 questions**:\n\n"
        + "\n".join([f"- {q}" for q in HELP]) +
        "\n\n_Data is live from **Supabase** via read-only SQL **views**._"
    )

with main:
    # -------- small helpers --------
    def classify(q: str):
        ql = q.lower()
        if "last" in ql and "30" in ql and "spend" in ql: return "spend_30d"
        if "this month" in ql and "spend" in ql: return "spend_this_month"
        if "top" in ql and ("merchant" in ql or "store" in ql): return "top_merchants_this_month"
        if "month-by-month" in ql or ("trend" in ql and "overall" in ql): return "monthly_spend_series"
        if "trend" in ql and ("walmart" in ql or "merchant" in ql): return "merchant_trend"
        if "average" in ql and ("receipt" in ql or "ticket" in ql) and "this month" in ql: return "avg_ticket_this_month"
        if ("how many" in ql or "count" in ql) and "receipt" in ql and "this month" in ql: return "count_receipts_this_month"
        if "average" in ql and "items" in ql and "this month" in ql: return "avg_items_this_month"
        if "most purchased" in ql or "top items" in ql: return "top_items_60d"
        if "largest" in ql and "receipt" in ql: return "max_receipts_90d"
        return "help"

    def _latest_month():
        r = sb.table("rx_v_spend_monthly").select("month").order("month", desc=True).limit(1).execute()
        return r.data[0]["month"] if r.data else None

    def has_data(payload) -> bool:
        """Return True if the returned payload actually contains usable rows/numbers."""
        if isinstance(payload, list):
            return len(payload) > 0
        if isinstance(payload, dict):
            if "daily" in payload:
                return bool(payload.get("daily")) or float(payload.get("total_spend_last_30d", 0)) > 0
            for key in ("spend", "avg_ticket", "receipt_count", "avg_items_per_receipt"):
                if key in payload and float(payload.get(key) or 0) > 0:
                    return True
            return any(bool(v) for v in payload.values())
        return bool(payload)

    # -------- queries (Supabase views) --------
    def q_spend_30d():
        r = sb.table("rx_v_spend_daily").select("*").order("day", desc=False).execute()
        df = pd.DataFrame(r.data or [])
        if df.empty:
            return {"total_spend_last_30d": 0.0, "daily": []}
        # FIX: coerce 'day' -> date for comparison
        df["day"] = to_date_series(df.get("day"))
        cutoff = dt.date.today() - dt.timedelta(days=30)
        mask = df["day"] >= cutoff
        total30 = float(df.loc[mask, "spend"].astype(float).sum()) if "spend" in df.columns else 0.0
        last30 = df.loc[mask, ["day", "spend"]] if {"day","spend"} <= set(df.columns) else df
        return {"total_spend_last_30d": total30, "daily": last30.to_dict("records")}

    def q_spend_this_month():
        m = str(month_start())
        r = sb.table("rx_v_spend_monthly").select("*").eq("month", m).execute()
        if not r.data:
            lm = _latest_month()
            if lm:
                r = sb.table("rx_v_spend_monthly").select("*").eq("month", lm).execute()
                st.caption(f"ℹ️ No data for this month; showing the latest month: {lm}")
                m = lm
        val = float(r.data[0]["spend"]) if r.data else 0.0
        return {"month": m, "spend": val}

    def q_top_merchants_this_month():
        m = str(month_start())
        r = (sb.table("rx_v_merchant_monthly")
             .select("*").eq("month", m)
             .order("spend", desc=True).limit(5).execute())
        rows = r.data or []
        if not rows:
            lm = _latest_month()
            if lm:
                r = (sb.table("rx_v_merchant_monthly")
                     .select("*").eq("month", lm)
                     .order("spend", desc=True).limit(5).execute())
                rows = r.data or []
                if rows:
                    st.caption(f"ℹ️ No data for this month; showing top merchants for {lm}")
        return rows

    def q_monthly_spend_series():
        r = sb.table("rx_v_spend_monthly").select("*").order("month", desc=False).execute()
        return (r.data or [])[-6:]

    def extract_merchant(text: str):
        for token in text.split():
            t = token.strip(",.!? ").lower()
            if t in ("walmart","target","kroger","aldi","costco","amazon","safeway","heb"):
                return t
        return None

    def q_merchant_trend(q: str):
        mname = extract_merchant(q) or "walmart"
        r = (sb.table("rx_v_merchant_monthly")
             .select("*")
             .ilike("merchant_name", f"%{mname}%")
             .order("month", desc=False).execute())
        return r.data or []

    def q_avg_ticket_this_month():
        m = str(month_start())
        r = sb.table("rx_v_receipt_stats_monthly").select("*").eq("month", m).execute()
        if not r.data:
            lm = _latest_month()
            if lm:
                r = sb.table("rx_v_receipt_stats_monthly").select("*").eq("month", lm).execute()
                st.caption(f"ℹ️ No data for this month; avg ticket for latest month: {lm}")
                m = lm
        avgv = float(r.data[0]["avg_ticket"]) if r.data else 0.0
        return {"month": m, "avg_ticket": avgv}

    def q_count_receipts_this_month():
        m = str(month_start())
        r = sb.table("rx_v_receipt_stats_monthly").select("*").eq("month", m).execute()
        if not r.data:
            lm = _latest_month()
            if lm:
                r = sb.table("rx_v_receipt_stats_monthly").select("*").eq("month", lm).execute()
                st.caption(f"ℹ️ No data for this month; count for latest month: {lm}")
                m = lm
        cnt = int(r.data[0]["receipt_count"]) if r.data else 0
        return {"month": m, "receipt_count": cnt}

    def q_avg_items_this_month():
        m = str(month_start())
        r = sb.table("rx_v_items_per_receipt_monthly").select("*").eq("month", m).execute()
        avgv = float(r.data[0]["avg_items_per_receipt"]) if r.data else 0.0
        return {"month": m, "avg_items_per_receipt": avgv}

    def q_top_items_60d():
        r = sb.table("rx_v_items_60d").select("*").limit(20).execute()
        return r.data or []

    def q_max_receipts_90d():
        r = sb.table("rx_v_max_receipts_90d").select("*").limit(20).execute()
        return r.data or []

    INTENT_TO_FUNC = {
        "spend_30d": q_spend_30d,
        "spend_this_month": q_spend_this_month,
        "top_merchants_this_month": q_top_merchants_this_month,
        "monthly_spend_series": q_monthly_spend_series,
        "merchant_trend": q_merchant_trend,
        "avg_ticket_this_month": q_avg_ticket_this_month,
        "count_receipts_this_month": q_count_receipts_this_month,
        "avg_items_this_month": q_avg_items_this_month,
        "top_items_60d": q_top_items_60d,
        "max_receipts_90d": q_max_receipts_90d,
        "help": lambda: {"help": HELP},
    }

    # -------- Groq Prompt A/B --------
    def narrate(prompt: str, facts: str):
        if not GROQ_API_KEY:
            return "(LLM disabled) " + prompt[:80], 0

        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        sys = "You are a strict analyst. Use ONLY provided facts. Never invent numbers or dates."
        user = f"Facts (JSON or table):\n{facts}\n\nTask:\n{prompt}"
        t0 = dt.datetime.now()

        def _call(model_id: str):
            return client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=350,
            )

        try:
            out = _call("llama-3.3-70b-versatile")  # main model
        except Exception:
            out = _call("llama-3.1-8b-instant")     # fallback

        latency_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)
        return out.choices[0].message.content, latency_ms

    PROMPT_A = "Summarize the results in plain English. Do not add numbers not present in facts."
    PROMPT_B = (
        "Write a concise consumer-style summary:\n"
        "- Use $ for amounts and month names.\n"
        "- If a list is present, highlight the top 3 with numbers.\n"
        "- No numbers not present in facts. 4 sentences max."
    )

    # -------- session + top action row --------
    if "chat" not in st.session_state:
        st.session_state.chat = []

    top_a, _ = st.columns([1, 7])
    with top_a:
        if st.button("🧹 Clear results"):
            st.session_state.chat.clear()
            st.rerun()

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    q = st.chat_input("Ask one of the 10 supported questions…")

    if q:
        # single-turn behavior: show only the latest results
        st.session_state.chat.clear()

        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)

        intent = classify(q)
        data = INTENT_TO_FUNC[intent]() if intent != "merchant_trend" else q_merchant_trend(q)

        with st.expander("Raw data from Supabase (read-only views)", expanded=False):
            st.write(data)

        if not has_data(data):
            st.info("No data found for that question/period. Try another question.")
        else:
            facts = pd.DataFrame(data).to_json(orient="records") if isinstance(data, list) else str(data)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Prompt A")
                outA, latA = narrate(PROMPT_A, facts)
                st.markdown(f"<div class='result-box'>{outA}</div>", unsafe_allow_html=True)
                st.caption(f"Latency: {latA} ms")

            with c2:
                st.markdown("#### Prompt B (guided)")
                outB, latB = narrate(PROMPT_B, facts)
                st.markdown(f"<div class='result-box'>{outB}</div>", unsafe_allow_html=True)
                st.caption(f"Latency: {latB} ms")

            st.session_state.chat.append(("assistant", outB))

        st.divider()
        if st.button("🔄 ReRun A/B", type="primary", use_container_width=True):
            st.session_state.chat.clear()
            st.rerun()
