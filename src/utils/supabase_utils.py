# src/utils/supabase_utils.py
"""
Secrets helper + Supabase client creator.

Resolution order for secrets:
1) st.secrets (Streamlit Cloud / local .streamlit/secrets.toml)
2) Environment variables (.env, Doppler CLI, GH Actions, Docker)

Aliases supported:
- SUPABASE_URL  or SUPABASE__URL
- SUPABASE_SERVICE_KEY  or SUPABASE__SUPABASE_SERVICE_KEY  or SUPABASE_KEY
- MISTRAL_API_KEY  or MISTRAL__API_KEY
- GROQ_API_KEY     or GROQ__API_KEY
"""

from __future__ import annotations

import os
import streamlit as st
from supabase import create_client, Client

# Optional: make local .env work
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def sget(*names: str) -> str | None:
    """
    Return the first non-empty value among names,
    checking Streamlit secrets first, then environment.
    """
    for n in names:
        # Streamlit secrets
        try:
            if hasattr(st, "secrets") and n in st.secrets:
                v = st.secrets[n]
                if v:
                    return str(v)
        except Exception:
            pass
        # Environment
        v = os.getenv(n)
        if v:
            return v
    return None


def _missing_msg(missing: list[str]) -> str:
    return (
        "Missing required secrets: "
        + ", ".join(missing)
        + "\nAdd them in Streamlit Cloud → Settings → Secrets (TOML) or export as env vars.\n"
        "Aliases supported for Supabase: SUPABASE__URL, SUPABASE__SUPABASE_SERVICE_KEY."
    )


@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Client:
    """Create a cached Supabase client."""
    url = sget("SUPABASE_URL", "SUPABASE__URL")
    key = sget("SUPABASE_SERVICE_KEY", "SUPABASE__SUPABASE_SERVICE_KEY", "SUPABASE_KEY")

    missing = []
    if not url:
        missing.append("SUPABASE_URL")
    if not key:
        missing.append("SUPABASE_SERVICE_KEY")
    if missing:
        raise RuntimeError(_missing_msg(missing))

    return create_client(url, key)


# --- LLM API keys -------------------------------------------------------------

def get_mistral_api_key(required: bool = True) -> str | None:
    """Returns Mistral key from secrets/env; raises if required and missing."""
    key = sget("MISTRAL_API_KEY", "MISTRAL__API_KEY")
    if required and not key:
        raise RuntimeError("Missing Mistral API key. Set MISTRAL_API_KEY in secrets or env.")
    return key


def get_groq_api_key(required: bool = True) -> str | None:
    """Returns Groq key from secrets/env; raises if required and missing."""
    key = sget("GROQ_API_KEY", "GROQ__API_KEY")
    if required and not key:
        raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY in secrets or env.")
    return key


def ensure_llm_env() -> None:
    """
    Convenience: export keys to os.environ so SDKs that auto-read env work.
    Safe no-op if already set.
    """
    m = get_mistral_api_key(required=False)
    g = get_groq_api_key(required=False)
    if m and not os.getenv("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = m
    if g and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = g
