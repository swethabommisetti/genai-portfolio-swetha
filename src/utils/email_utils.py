import streamlit as st
import re
from utils.visitor_service import fetch_or_insert_visitor_id

def is_valid_email(email):
    return re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email)

def prompt_for_optional_email():
    email = st.text_input(
        "Leave your email if you'd like me to know you visited (recruiters: please enter your work email!)"
    )
    if email and is_valid_email(email):
        st.session_state["user_email"] = email
        st.session_state["visitor_id"] = fetch_or_insert_visitor_id(email)
    else:
        st.session_state["user_email"] = ""
        st.session_state["visitor_id"] = None
    return st.session_state["user_email"]
