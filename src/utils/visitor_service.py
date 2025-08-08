from supabase import create_client
import os
from utils.supabase_utils import get_supabase_client

# Setup client
#supabase = create_client(
#    os.getenv("SUPABASE_URL"),
#    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
#)

def fetch_or_insert_visitor_id(email: str) -> str:
    """Insert email into visitors (if needed), return visitor_id"""
    if not email:
        return None

    # Step 1: Try fetch
    result = get_supabase_client.table("visitors").select("id").eq("email", email).limit(1).execute()
    if result.data:
        return result.data[0]["id"]

    # Step 2: Insert
    insert = get_supabase_client.table("visitors").insert({"email": email}).execute()
    return insert.data[0]["id"]
