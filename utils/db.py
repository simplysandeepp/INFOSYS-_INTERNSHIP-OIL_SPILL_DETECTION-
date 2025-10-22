from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load .env from project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

# Read Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug prints to ensure values are loaded
print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_KEY length:", len(SUPABASE_KEY) if SUPABASE_KEY else None)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found. Check your .env file!")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Supabase connected ✅")
