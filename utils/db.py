from supabase import create_client, Client
import os
import streamlit as st

# Try loading from Streamlit secrets first (Cloud)
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except (KeyError, ModuleNotFoundError):
    # Fallback to local .env
    from dotenv import load_dotenv
    import os
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path)
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Check if values exist
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found! Check your .env or Streamlit secrets.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase connected ✅")
