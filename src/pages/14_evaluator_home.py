import streamlit as st

st.title("🧪 Evaluator — Overview")
st.markdown("""
Use these tools to **assess model quality** like an AI Trainer/Evaluator:

1. **Manual Scoring** — mark fields as correct/incorrect; give an overall score and notes.  
2. **Error Tagging** — tag common failure types (missed item, hallucination, wrong total…).  
3. **Eval Analytics** — see aggregated accuracy and error frequencies.
""")

c1, c2, c3 = st.columns(3)
if c1.button("➡️ Manual Scoring"):
    st.switch_page("pages/15_evaluator_manual_scoring.py")
if c2.button("➡️ Error Tagging"):
    st.switch_page("pages/16_evaluator_error_tagging.py")
if c3.button("➡️ Eval Analytics"):
    st.switch_page("pages/13_receiptscanner_analytics.py")
