from supabase import create_client
import streamlit as st

@st.cache_resource
def get_supabase_client():
    #url = st.secrets["SUPABASE_URL"]
    #key = st.secrets["SUPABASE_KEY"]
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["SUPABASE_SERVICE_KEY"]
    
    return create_client(url, key)
