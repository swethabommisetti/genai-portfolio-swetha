import streamlit as st
from datetime import datetime
import uuid
from utils.supabase_utils import get_supabase_client
import streamlit.components.v1 as components

def log_once_per_page(page_name, extra_fields=None):
    """
    Logs a visit to Supabase only once per session for a specific page.
    Prevents repeated inserts during Streamlit reruns.
    """
    visit_key = f"visited_{page_name}"
    if st.session_state.get(visit_key):
        return  # Already logged for this page

    st.session_state[visit_key] = True  # Mark this page as logged

    # Optional: collect IP and User-Agent (if not already set)
    js = """
    <script>
      async function sendInfo() {
        let ip = "unknown";
        try {
          const res = await fetch('https://api.ipify.org?format=json');
          ip = (await res.json()).ip;
        } catch {}
        const ua = navigator.userAgent;
        const msg = JSON.stringify({ip: ip, ua: ua});
        window.parent.postMessage(msg, "*");
      }
      sendInfo();
    </script>
    """
    components.html(js, height=0)

    ip = st.session_state.get("client_ip", "unknown")
    user_agent = st.session_state.get("user_agent", "unknown")
    session_id = st.session_state.get("session_id", str(uuid.uuid4()))

    supabase = get_supabase_client()
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "page_name": page_name,
        "ip_address": ip,
        "user_agent": user_agent,
        "insrt_user_id": session_id,
        "visitor_id": st.session_state.get("visitor_id"),
        "visitor_email": st.session_state.get("user_email", None),
    }

    if extra_fields:
        row.update(extra_fields)

    try:
        supabase.table("visits").insert(row).execute()
    except Exception as e:
        st.warning(f"⚠️ Visit log failed: {e}")
