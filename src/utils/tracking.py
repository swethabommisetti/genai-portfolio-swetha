import streamlit as st
from datetime import datetime
from utils.supabase_utils import get_supabase_client
import uuid

def log_page_visit(page_name, extra_fields=None):
    """
    Logs a visit to the current page to Supabase (and optionally more fields).
    """
    # --- JavaScript: collect IP and User-Agent
    import streamlit.components.v1 as components
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

    # --- Try to grab JS result from session_state (advanced)
    # For simplicity in a serverless context, we'll fallback to "unknown"
    ip = st.session_state.get("client_ip", "unknown")
    user_agent = st.session_state.get("user_agent", "unknown")
    session_id = st.session_state.get("session_id", str(uuid.uuid4()))

    # --- Insert to Supabase
    supabase = get_supabase_client()
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "page_name": page_name,
        "ip_address": ip,
        "user_agent": user_agent,
        "insrt_user_id":session_id,
        "visitor_id": st.session_state.get("visitor_id"),  
        "visitor_email": st.session_state.get("user_email", None), 
    }
    if extra_fields:
        row.update(extra_fields)
    supabase.table("visits").insert(row).execute()
