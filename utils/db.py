from supabase import create_client, Client
import os

# Try loading Streamlit secrets first (for cloud)
try:
    import streamlit as st
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except (ImportError, KeyError):
    # Fallback to local .env
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path)
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client if credentials exist
if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase connected ✅")
else:
    supabase = None
    print("Supabase credentials not found! Set .env or Streamlit secrets.")

# -----------------------------
# Helper functions for database
# -----------------------------

def insert_detection_data(table_name: str, data: dict):
    """
    Insert a row into a Supabase table.
    :param table_name: str, name of the Supabase table
    :param data: dict, data to insert
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized")
    response = supabase.table(table_name).insert(data).execute()
    return response

def fetch_all_detections(table_name: str):
    """
    Fetch all rows from a Supabase table.
    :param table_name: str, name of the Supabase table
    :return: list of dicts
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized")
    response = supabase.table(table_name).select("*").execute()
    return response.data
